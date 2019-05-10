import torch.nn as nn

from collections import Iterable


def Nobiasdecay(model, skips=[]):
    assert isinstance(skips, Iterable)
    assert isinstance(model, nn.Module)

    no_decay_params = []
    decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (len(param.shape) == 1 and 'bn' in name) or name.endswith('.bias') or name in skips:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    assert len(list(model.parameters())) == (len(decay_params) + len(no_decay_params))

    return [{'params': no_decay_params, 'weight_decay': 0.0},
            {'params': decay_params}]
