"""
 @Time    : 2020/3/15 20:09
 @Author  : TaylorMei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2020_GDNet
 @File    : misc.py
 @Function:
 
"""
import os
import xlwt
import numpy as np
import pydensecrf.densecrf as dcrf
from skimage import io


################################################################
######################## Utils #################################
################################################################
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(sheetname="sheet1", cell_overwrite_ok=True)

    j = 0
    for data in datas:
        for i in range(len(data)):
            sheet1.write(i, j, data[i])
        j = j + 1

    f.save(file_path)


################################################################
######################## Train & Test ##########################
################################################################
class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# codes of this function are borrowed from https://github.com/Andrew-Qibin/dss_crf
def crf_refine(img, annos):
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')


def get_gt_mask(imgname, MASK_DIR):
    filestr = imgname[:-4]
    mask_folder = MASK_DIR
    mask_path = os.path.join(mask_folder, filestr + ".png")
    mask = io.imread(mask_path)
    mask = np.where(mask == 255, 1, 0).astype(np.float32)

    return mask


def get_normalized_predict_mask(imgname, PREDICT_MASK_DIR):
    filestr = imgname[:-4]
    mask_folder = PREDICT_MASK_DIR
    mask_path = os.path.join(mask_folder, filestr + ".png")
    if not os.path.exists(mask_path):
        print("{} has no predict mask!".format(imgname))
    mask = io.imread(mask_path).astype(np.float32)
    if np.max(mask) - np.min(mask) > 0:
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    else:
        mask = mask / 255.0
    mask = mask.astype(np.float32)

    return mask


def get_binary_predict_mask(imgname, PREDICT_MASK_DIR):
    filestr = imgname[:-4]
    mask_folder = PREDICT_MASK_DIR
    mask_path = os.path.join(mask_folder, filestr + ".png")
    if not os.path.exists(mask_path):
        print("{} has no predict mask!".format(imgname))
    mask = io.imread(mask_path).astype(np.float32)
    mask = np.where(mask >= 127.5, 1, 0).astype(np.float32)

    return mask


################################################################
######################## Evaluation ############################
################################################################
def compute_iou(predict_mask, gt_mask):
    check_size(predict_mask, gt_mask)

    if np.sum(predict_mask) == 0 or np.sum(gt_mask) == 0:
        iou_ = 0
        return iou_

    n_ii = np.sum(np.logical_and(predict_mask, gt_mask))
    t_i = np.sum(gt_mask)
    n_ij = np.sum(predict_mask)

    iou_ = n_ii / (t_i + n_ij - n_ii)

    return iou_


def compute_acc(predict_mask, gt_mask):
    # recall
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    accuracy_ = TP / N_p

    return accuracy_


def compute_acc_image(predict_mask, gt_mask):
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    accuracy_ = (TP + TN) / (N_p + N_n)

    return accuracy_


def compute_precision_recall(prediction, gt):
    assert prediction.dtype == np.float32
    assert gt.dtype == np.float32
    assert prediction.shape == gt.shape

    eps = 1e-4

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)

    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall


def compute_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure


def compute_mae(predict_mask, gt_mask):
    check_size(predict_mask, gt_mask)

    mae_ = np.mean(abs(predict_mask - gt_mask)).item()

    return mae_


def compute_ber(predict_mask, gt_mask):
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    ber_ = 100 * (1 - (1 / 2) * ((TP / N_p) + (TN / N_n)))

    return ber_


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
