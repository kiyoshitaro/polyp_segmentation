import torch


def adam(params, lr):
    return torch.optim.Adam(params, lr)


def sgd(params, lr):
    return torch.optim.SGD(params, lr=lr)
