import torch
import numpy as np
from thop import profile
from thop import clever_format

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import matplotlib.pyplot as plt

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def cosine_warmup(optimizer, init_lr, epoch, total_epoch, num_warmup_epoch):
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch, eta_min=init_lr, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=num_warmup_epoch, after_scheduler=cosine_scheduler)
    scheduler.step(epoch)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))




    # v = torch.zeros(10)
    # optimizer = torch.optim.SGD([v], lr=lr/8)
    # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0, last_epoch=-1)
    # scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)
    # a = []
    # b = []
    # for epoch in range(1, 100):
    #     scheduler.step(epoch)
    #     a.append(epoch)
    #     b.append(optimizer.param_groups[0]['lr'])
    #     print(epoch, optimizer.param_groups[0]['lr'])

    # plt.plot(a,b)