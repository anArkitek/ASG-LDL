# Towards Unbiased Label Distribution Learning for Facial Pose Estimation Using Anisotropic Spherical Gaussian 	

## Abstract

Facial pose estimation refers to the task of predicting face orientation from a single RGB image. It is an important research topic with a wide range of applications in computer vision. Label distribution learning (LDL) based methods have been recently proposed for facial pose estimation, which achieve promising results. However, there are two major issues in existing LDL methods. First, the expectations of label distributions are biased, leading to a biased pose estimation. Second, fixed distribution parameters are applied for all learning samples, severely limiting the model capability. In this paper, we propose an Anisotropic Spherical Gaussian (ASG)-based LDL approach for facial pose estimation. In particular, our approach adopts the spherical Gaussian distribution on a unit sphere which constantly generates unbiased expectation. Meanwhile, we introduce a new loss function that allows the network to learn the distribution parameter for each learning sample flexibly. Extensive experimental results show that our method sets new state-of-the-art records on AFLW2000 and BIWI datasets.

## Test on aflw2000

We have prepared a pre-trained network and the aflw2000 dataset along with the project thanks to their relatively small size. Simply process as follows:

```bash
cd data
unzip aflw2000
cd ..
sh run_test.sh
```

## Data Preparation

We follow the same data preparation procedure as [FSA-NET](https://github.com/shamangary/FSA-Net). Please refer to this [link](https://github.com/shamangary/FSA-Net#1-data-pre-processing) for the details of data preparation.

All other datasets need to be organized in the same structure just as AFLW2000 if you want to test on them.

## Train the Network

If you want to train the network on your own dataset, please organize the training dataset the same as AFLW2000, and use run_train.sh to train the network.

## Citation

```txt
@misc{https://doi.org/10.48550/arxiv.2208.09122,
  doi = {10.48550/ARXIV.2208.09122},
  
  url = {https://arxiv.org/abs/2208.09122},
  
  author = {Cao, Zhiwen and Liu, Dongfang and Wang, Qifan and Chen, Yingjie},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Towards Unbiased Label Distribution Learning for Facial Pose Estimation Using Anisotropic Spherical Gaussian},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
