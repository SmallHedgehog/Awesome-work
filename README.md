Awesome-Work implemented by PyTorch
========================

This repository aims to reimplement some interesting work and provides examples for self-study and customization in your 
own applications.

## Contents
- [Object detection](#Object-detection)
- [Image classification](#Image-classification)

## Object detection
1. [GHM](https://github.com/libuyu/GHM_Detection/blob/master/mmdetection/mmdet/core/loss/ghm_loss.py)

## Image classification
1. [Tricks](https://arxiv.org/pdf/1812.01187.pdf)

## Experiments on CIFAR10
image classification tricks on CIFAR10, [refer to](https://arxiv.org/pdf/1812.01187.pdf)[1]


|                 **Heuristic**                 |   mAP@1    |    Lift    |
| :-------------------------------------------: | :--------: | :--------: |
|                   ResNet32                    |   92.40%   |     BL     |
|              ResNet32+ZeroGamma               |   93.39%   |   +0.99%   |
|                   ResNet32D                   |   93.06%   |   +0.66%   |
|              ResNet32D+ZeroGamma              |   93.05%   |   +0.65%   |
|           ResNet32+CosineAnnealing            |   93.66%   |   +1.26%   |
|            ResNet32+LabelSmoothing            |   92.92%   |   +0.52%   |
|      ResNet32+ZeroGamma+CosineAnnealing       |   93.63%   |   +1.23%   |
|        ResNet32+CosineAnnealing+Mixup         |   94.18%   |   +1.78%   |
| ResNet32+CosineAnnealing+Mixup+LabelSmoothing | **94.38%** | **+1.98%** |
