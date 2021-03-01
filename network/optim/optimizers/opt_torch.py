import torch


def adam(params, lr):
    return torch.optim.Adam(params, lr)
