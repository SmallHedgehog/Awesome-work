"""Reference:
    [1] Buyu Li, Yu Liu etc. Gradient Harmonized Single-stage Detector. AAAI 2019.
    [2] https://github.com/libuyu/GHM_Detection/blob/master/mmdetection/mmdet/core/loss/ghm_loss.py
"""

import torch.nn as nn
import torch


class GHMC_Loss(nn.Module):
    def __init__(self, bins=10, momentum=0, eps=1e-8):
        super(GHMC_Loss, self).__init__()

        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if self.momentum > 0:
            self.acc_sum = [0.0] * bins
        self.eps = eps

    def forward(self, pred, target):
        """ pred (batch_size, num_classes)
            target (batch_size)
        """
        batch_size = pred.size(0)
        weights = torch.zeros(batch_size)

        idx = torch.arange(batch_size).long()
        g = 1 - torch.softmax(pred.detach(), dim=1)[idx, target]

        n = 0   # n valid bins
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] \
                        + (1 - self.momentum) * num_in_bin
                    # weights[inds] = batch_size / self.acc_sum[i]
                    weights[inds] = 1. / self.acc_sum[i]
                else:
                    # weights[inds] = batch_size / num_in_bin
                    weights[inds] = 1. / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        weights = weights.to(pred.device)

        # pred_prob = pred.softmax(dim=1)[idx, target] + self.eps
        # # return -(torch.log(pred_prob) * weights).sum() / batch_size
        # return -(torch.log(pred_prob) * weights).sum()

        loss = (-pred[idx, target] + torch.log(torch.exp(pred).sum(dim=1) + self.eps))
        # return (loss * weights).sum() / batch_size
        return (loss * weights).sum()


class GHMC_LossV2(nn.Module):
    def __init__(self, bins=10, momentum=0, eps=1e-8):
        super(GHMC_LossV2, self).__init__()

        self.alpha = 1. / (2 * bins)
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        if self.momentum > 0:
            self.acc_sum = [0.0] * (bins + 1)
        self.eps = eps

    def forward(self, pred, target):
        """ pred (batch_size, num_classes)
            target (batch_size)
        """
        batch_size = pred.size(0)
        weights = torch.zeros(batch_size)

        idx = torch.arange(batch_size)
        g = 1 - torch.softmax(pred.detach(), dim=1)[idx, target]

        n = 0   # n valid bins
        for i in range(self.bins + 1):
            inds = (g >= (self.edges[i] - self.alpha)) & (g < (self.edges[i] + self.alpha))
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] \
                        + (1 - self.momentum) * num_in_bin
                    # weights[inds] = batch_size / self.acc_sum[i]
                    weights[inds] = 1. / self.acc_sum[i]
                else:
                    # weights[inds] = batch_size / num_in_bin
                    weights[inds] = 1. / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        weights = weights.to(pred.device)

        # pred_prob = pred.softmax(dim=1)[idx, target] + self.eps
        # # return -(torch.log(pred_prob) * weights).sum() / batch_size
        # return -(torch.log(pred_prob) * weights).sum()

        # Be careful, Not recommended!!!
        loss = (-pred[idx, target] + torch.log(torch.exp(pred).sum(dim=1) + self.eps))
        # return (loss * weights).sum() / batch_size
        return (loss * weights).sum()


if __name__ == '__main__':
    x = torch.randn(5, 21)
    target = torch.Tensor([20, 1, 15, 3, 9]).long()

    loss = GHMC_LossV2()(x, target)
    print(loss)
