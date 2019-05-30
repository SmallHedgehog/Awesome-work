Awesome-work implemented by PyTorch
========================

This repository aims to reimplement some interesting work and provides examples for self-study and customization in your 
own applications.

## Contents
- [Object detection](#Object-detection)
- [Image classification](#Image-classification)
- [Optimization](#Optimization)
- [AutoML](#AutoML)

## Object detection
1. [AAAI 2019] [Gradient Harmonized Single-stage Detector](https://arxiv.org/abs/1811.05181v1).
We modify the GHM-C Loss in this paper for multiple classification problems. [[code](https://github.com/SmallHedgehog/Awesome-work/blob/master/GHM/GHMC_Loss.py)]

## Image classification
1. [CVPR 2019] [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf).
We tested some tricks on the CIFAR10 dataset, see [Experiments on CIFAR10](#Experiments-on-CIFAR10) for more details.

2. [CVPR 2019] [Selective Kernel Networks](https://arxiv.org/abs/1903.06586). [[code](https://github.com/implus/SKNet)]. [[our impl.](https://github.com/SmallHedgehog/Awesome-work/blob/master/model/SK/SKNet.py)]

## Optimization
1. [ICLR 2019] [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). [[code](https://github.com/kashif/pytorch/blob/adamw/torch/optim/adam.py)]

## AutoML
We use [NNI(Neural Network Intelligence)](https://github.com/microsoft/nni) toolkit to search the best hyperparameters(learning rate, label smoothing, etc.) in local machine. 
Our code in [here](https://github.com/SmallHedgehog/Awesome-work/blob/master/BL_CA_MU_LB_NNI.py), the **Experiment**
of [NNI](https://github.com/microsoft/nni) (including search space and config) in [here](https://github.com/SmallHedgehog/Awesome-work/tree/master/experiments/cifar10-bl_ca_mu_lb_nni). 

We test three hyperparameters by NNI toolkit, including learning rate, label smoothing and alpha(for [mixup](https://github.com/SmallHedgehog/Awesome-work/blob/master/MIXUP-trian.py) trick).
we use RestNet32 architecture with [CosineAnnealing](https://github.com/SmallHedgehog/Awesome-work/blob/master/COSINEANNEAL-train.py), 
[Mixup](https://github.com/SmallHedgehog/Awesome-work/blob/master/MIXUP-trian.py) and 
[LabelSmoothing](https://github.com/SmallHedgehog/Awesome-work/blob/master/LABELSMOOTH-train.py) tricks for training like Artificial experiment on NNI search experiment, 
the form of below compares the differents between artificial and NNI search hyperparameters, giving the best mAP@1.

|                        | Artificial | NNI Search |
| :--------------------: | :--------: | :--------: |
|   init learning rate   |    0.1     | 0.1339448827100801 |
| alpha(for mixup trick) |    1.0     | 0.24223853906627413 |
|    label smoothing     |    0.1     | 0.03521136514652663 |
|         mAP@1          |   94.38%   | 94.39% |

## Experiments on CIFAR10
image classification tricks on CIFAR10, [refer to](https://arxiv.org/pdf/1812.01187.pdf)[1]

|                 **Heuristic**                 |   mAP@1    |    Lift    |
| :-------------------------------------------: | :--------: | :--------: |
|                   ResNet32                    |   92.40%   |     (+0.0, BASELINE)     |
|              ResNet32+ZeroGamma               |   93.39%   |   +0.99%   |
|                   ResNet32D                   |   93.06%   |   +0.66%   |
|              ResNet32D+ZeroGamma              |   93.05%   |   +0.65%   |
|           ResNet32+CosineAnnealing            |   93.66%   |   +1.26%   |
|            ResNet32+LabelSmoothing            |   92.92%   |   +0.52%   |
|      ResNet32+ZeroGamma+CosineAnnealing       |   93.63%   |   +1.23%   |
|        ResNet32+CosineAnnealing+Mixup         |   94.18%   |   +1.78%   |
| ResNet32+CosineAnnealing+Mixup+LabelSmoothing | **94.38%** | **+1.98%** |
