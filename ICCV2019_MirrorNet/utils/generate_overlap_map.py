"""
 @Time    : 9/15/19 16:47
 @Author  : TaylorMei
 @Email   : mhy666@mail.dlut.edu.cn
 
 @Project : ICCV2019_MirrorNet
 @File    : generate_overlap_map.py
 @Function: generate overlap map of each image in test set, according to the statistic on training set.
 
"""
import os
import numpy as np
from skimage import io, transform
from config import msd_training_root, msd_testing_root, msd_results_root

train_image_path = os.path.join(msd_training_root, 'image')
test_image_path = os.path.join(msd_testing_root, 'image')
mask_path = os.path.join(msd_training_root, 'mask')
output_path = os.path.join(msd_results_root, 'Statistics')
if not os.path.exists(output_path):
    os.mkdir(output_path)

overlap = np.zeros([256, 256], dtype=np.float64)

train_imglist = os.listdir(train_image_path)
for i, imgname in enumerate(train_imglist):

    print(i, imgname)

    name = imgname.split('.')[0]

    mask = io.imread(os.path.join(mask_path, name + '.png'))

    mask = transform.resize(mask, [256, 256], order=0)
    mask = np.where(mask != 0, 1, 0).astype(np.float64)

    overlap += mask

overlap = overlap / len(train_imglist)
overlap = (overlap - np.min(overlap)) / (np.max(overlap) - np.min(overlap))

test_imglist = os.listdir(test_image_path)
for j, imgname in enumerate(test_imglist):

    print(j, imgname)

    name = imgname.split('.')[0]

    image = io.imread(os.path.join(test_image_path, imgname))

    height = image.shape[0]
    width = image.shape[1]

    mask = transform.resize(overlap, [height, width], 0)

    save_path = os.path.join(output_path, name + '.png')
    io.imsave(save_path, (mask * 255).astype(np.uint8))

print("OK!")