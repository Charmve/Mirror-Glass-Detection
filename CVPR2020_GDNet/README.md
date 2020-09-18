# CVPR2020_GDNet

## Don't Hit Me! Glass Detection in Real-world Scenes
[Haiyang Mei](https://mhaiyang.github.io/), Xin Yang, Yang Wang, Yuanyuan Liu, Shengfeng He, Qiang Zhang, Xiaopeng Wei, and Rynson W.H. Lau

[[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Mei_Dont_Hit_Me_Glass_Detection_in_Real-World_Scenes_CVPR_2020_paper.pdf)] [[Project Page](https://mhaiyang.github.io/CVPR2020_GDNet/index.html)]

### Abstract
Glass is very common in our daily life. Existing computer vision systems neglect the glass and thus might lead to severe consequence, \eg, the robot might crash into the glass wall. However, sensing the presence of the glass is not straightforward. The key challenge is that arbitrary objects/scenes can appear behind the glass and the content presented in the glass region typically similar to those outside of it. In this paper, we raise an interesting but important problem of detecting glass from a single RGB image. To address this problem, we construct a large-scale glass detection dataset (GDD) and design a glass detection network, called GDNet, by learning abundant contextual features from a global perspective with a novel large-field contextual feature integration module. Extensive experiments demonstrate the proposed method achieves superior glass detection results on our GDD test set. Particularly, we outperform state-of-the-art methods that fine-tuned for glass detection.

### Citation
If you use this code or our dataset (including test set), please cite:

```
@InProceedings{Mei_2020_CVPR,
    author = {Mei, Haiyang and Yang, Xin and Wang, Yang and Liu, Yuanyuan and He, Shengfeng and Zhang, Qiang and Wei, Xiaopeng and Lau, Rynson W.H.},
    title = {Don't Hit Me! Glass Detection in Real-World Scenes},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

### Dataset
See [Peoject Page](https://mhaiyang.github.io/CVPR2020_GDNet/index.html)

### Requirements
* PyTorch == 1.0.0
* TorchVision == 0.2.1
* CUDA 10.0  cudnn 7.2
* Setup
```
sudo pip3 install -r requirements.txt
git clone https://github.com/Mhaiyang/dss_crf.git
sudo python setup.py install
```

### Test
Download trained model `GDNet.pth` at [here](https://mhaiyang.github.io/CVPR2020_GDNet/index.html), then run `infer.py`.

<!-- ### Experimental Results -->

<!-- ##### Quantitative Results -->
<!-- <img src="https://github.com/Mhaiyang/CVPR2020_GDNet/blob/master/assets/table1.png" width="60%" height="60%"> -->


<!-- ##### Component analysis -->
<!-- <img src="https://github.com/Mhaiyang/CVPR2020_GDNet/blob/master/assets/table2.png" width="60%" height="60%"> -->


<!-- ##### Qualitative Results -->
<!-- <img src="https://github.com/Mhaiyang/CVPR2020_GDNet/blob/master/assets/results.png" width="100%" height="100%"> -->

### License
Please see `license.txt`

### Contact
E-Mail: mhy666@mail.dlut.edu.cn
