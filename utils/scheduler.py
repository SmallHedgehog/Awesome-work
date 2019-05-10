import torch.optim as optim


def get_scheduler(optimizer, config):
    if config.type == 'STEP':
        return optim.lr_scheduler.MultiStepLR(optimizer, config.lr_steps, config.lr_mults)
    else:
        raise RuntimeError('unknown lr_scheduler type {}'.format(config.type))

