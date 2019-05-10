import torch.nn.init as init

def Zerogamma(model, last_bn_name):
    def init_func(m):
        if hasattr(m, last_bn_name):
            init.constant_(getattr(m, last_bn_name).weight.data, 0.0)
            init.constant_(getattr(m, last_bn_name).bias.data, 0.0)

    model.apply(init_func)
