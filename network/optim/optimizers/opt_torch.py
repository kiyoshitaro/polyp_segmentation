import torch
from adamp import AdamP


def adam(params, lr):
    return torch.optim.Adam(params, lr)


def sgd(params, lr):
    return torch.optim.SGD(params, lr=lr)

def adamp(params, lr):
    return AdamP(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
