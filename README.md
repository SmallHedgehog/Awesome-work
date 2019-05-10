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

|                   Heuristic                   |    Top1    |
| :-------------------------------------------: | :--------: |
|                   ResNet32                    |   92.40%   |
|              ResNet32+ZeroGamma               |   93.39%   |
|                   ResNet32D                   |   93.06%   |
|              ResNet32D+ZeroGamma              |   93.05%   |
|           ResNet32+CosineAnnealing            |   93.66%   |
|            ResNet32+LabelSmoothing            |   92.92%   |
|      ResNet32+ZeroGamma+CosineAnnealing       |   93.63%   |
|        ResNet32+CosineAnnealing+Mixup         |   94.18%   |
| ResNet32+CosineAnnealing+Mixup+LabelSmoothing | **94.38%** |
