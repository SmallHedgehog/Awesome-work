import argparse
import os
import yaml
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.datasets as Datasets
import torchvision.transforms as transforms

from easydict import EasyDict
from tensorboardX import SummaryWriter

from tricks.training_refinements.LR import CosineAnnealing
from tricks.training_refinements.Mixup import mixup_data, mixup_criterion
from GHM.GHMC_Loss import GHMC_Loss
from model.resnet import resnet32
from utils.scheduler import get_scheduler
from utils.calc_acc import PerClassAccuracy

GPU_ID = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID


parser = argparse.ArgumentParser(description='Gradient Harmonized in Classify')
parser.add_argument('--config', default='experiments/cifar10-bl_ca_mixup/config.yaml')


def train(model, epoch_idx, criterion, lr_scheduler, optimizer, trainloader):
    model.train()

    for batch_idx, data in enumerate(trainloader):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()

        # Mixup images
        images, targets_a, targets_b, lam = mixup_data(images, labels, config.alpha)

        optimizer.zero_grad()
        out = model(images)
        pred = out.data.max(1)[1]
        PCA.update(labels, pred)

        # Mixup criterion
        loss = mixup_criterion(criterion, out, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        Writer.add_scalar('train loss', loss.item(), epoch_idx * len(trainloader) + batch_idx)

        print('TRAIN:{}/{} EPOCHs, {}/{} BATCHs, LOSS:{}'.format(epoch_idx, config.max_iter, batch_idx,
                len(trainloader), loss.item()))

    _, _, mAP = PCA.calc()
    Writer.add_scalar('train mAP', mAP, epoch_idx)
    PCA.reset()

def val(model, epoch_idx, criterion, testloader):
    model.eval()

    for batch_idx, data in enumerate(testloader):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            out = model(images)
            pred = out.data.max(1)[1]
            PCA.update(labels, pred)
            loss = criterion(out, labels)

        Writer.add_scalar('val loss', loss.item(), epoch_idx * len(testloader) + batch_idx)

        print('VAL:{}/{} EPOCHs, {}/{} BATCHs'.format(epoch_idx, config.max_iter, batch_idx, len(testloader)))

    _, AVG_acc, mAP = PCA.calc()
    Writer.add_scalar('avg acc', AVG_acc, epoch_idx)
    Writer.add_scalar('val mAP', mAP, epoch_idx)
    PCA.reset()


def main():
    global args, config

    args = parser.parse_args()

    with open(args.config) as rPtr:
        config = EasyDict(yaml.load(rPtr))

    config.save_path = os.path.dirname(args.config)

    # Random seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Datasets
    train_transform = transforms.Compose([
        transforms.RandomCrop((32, 32), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
    ])

    trainset = Datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
    trainloader = Data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)

    testset = Datasets.CIFAR10(root='data', train=False, download=True, transform=val_transform)
    testloader = Data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)

    # Model
    model = resnet32()
    model = model.cuda()

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr_scheduler.base_lr, momentum=config.momentum,
            weight_decay=config.weight_decay)

    # LR scheduler
    lr_scheduler = CosineAnnealing(optimizer, len(trainloader) * config.max_iter)

    global PCA, Writer
    PCA = PerClassAccuracy(num_classes=config.num_classes)
    Writer = SummaryWriter(config.save_path + '/events')
    for iter_idx in range(config.max_iter):
        train(model, iter_idx, criterion, lr_scheduler, optimizer, trainloader)
        val(model, iter_idx, criterion, testloader)

    Writer.close()


if __name__ == '__main__':
    main()
