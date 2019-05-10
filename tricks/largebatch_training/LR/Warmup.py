import torch.optim.lr_scheduler

def Warmup(optimizer, lr_warmup_iters):
    lr_lambda = lambda iters: (iters + 1) / lr_warmup_iters if iters <= lr_warmup_iters else lr_warmup_iters
    return lr_scheduler.LambdaLR(optimizer, lr_lambda)
