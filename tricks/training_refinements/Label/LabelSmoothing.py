import torch.nn as nn

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()

        self.smoothing = smoothing
        self.kldivloss = nn.KLDivLoss()

    def forward(self, x, target):
        """ x is not the predicted probabilities
        """
        x_prob = nn.LogSoftmax(1)(x)

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / x.size(1) - 1)
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)

        return self.kldivloss(x_prob, true_dist)
