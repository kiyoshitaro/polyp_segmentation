from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

import torch
from bisect import bisect_right


class GradualWarmupScheduler(_LRScheduler):
    # https://github.com/seominseok0429/pytorch-warmup-cosine-lr
    def __init__(self, optimizer, multiplier, total_warmup_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_warmup_epoch = total_warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_warmup_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * (
                (self.multiplier - 1.0) * self.last_epoch / self.total_warmup_epoch
                + 1.0
            )
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_warmup_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def cosine_warmup(optimizer, total_epoch, num_warmup_epoch):
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_epoch, eta_min=0, last_epoch=-1
    )
    return GradualWarmupScheduler(
        optimizer,
        multiplier=8,
        total_warmup_epoch=num_warmup_epoch,
        after_scheduler=cosine_scheduler,
    )
    # print(optimizer.param_groups[0]["lr"])


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] *= decay


import math


"""
https://github.com/CoinCheung/BiSeNet/blob/master/lib/lr_scheduler.py
"""


class WarmupLrScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_iter=500,
        warmup_ratio=5e-4,
        warmup="exp",
        last_epoch=-1,
    ):
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            ratio = self.get_main_ratio()
        return ratio

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ("linear", "exp")
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == "linear":
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == "exp":
            ratio = self.warmup_ratio ** (1.0 - alpha)
        return ratio


class WarmupPolyLrScheduler(WarmupLrScheduler):
    def __init__(
        self,
        optimizer,
        power,
        max_iter,
        warmup_iter=500,
        warmup_ratio=5e-4,
        warmup="exp",
        last_epoch=-1,
    ):
        self.power = power
        self.max_iter = max_iter
        super(WarmupPolyLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch
        )

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter
        ratio = (1 - alpha) ** self.power
        return ratio


class WarmupExpLrScheduler(WarmupLrScheduler):
    def __init__(
        self,
        optimizer,
        gamma,
        interval=1,
        warmup_iter=500,
        warmup_ratio=5e-4,
        warmup="exp",
        last_epoch=-1,
    ):
        self.gamma = gamma
        self.interval = interval
        super(WarmupExpLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch
        )

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        ratio = self.gamma ** (real_iter // self.interval)
        return ratio


class WarmupCosineLrScheduler(WarmupLrScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        eta_ratio=0,
        warmup_iter=5,
        warmup_ratio=5e-4,
        warmup="exp",
        last_epoch=-1,
    ):
        self.eta_ratio = eta_ratio
        self.max_iter = max_iter
        super(WarmupCosineLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch
        )

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        return (
            self.eta_ratio
            + (1 - self.eta_ratio)
            * (1 + math.cos(math.pi * self.last_epoch / real_max_iter))
            / 2
        )


class WarmupStepLrScheduler(WarmupLrScheduler):
    def __init__(
        self,
        optimizer,
        milestones: list,
        gamma=0.1,
        warmup_iter=500,
        warmup_ratio=5e-4,
        warmup="exp",
        last_epoch=-1,
    ):
        self.milestones = milestones
        self.gamma = gamma
        super(WarmupStepLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch
        )

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        ratio = self.gamma ** bisect_right(self.milestones, real_iter)
        return ratio


if __name__ == "__main__":
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.0001)
    scheduler = cosine_warmup(optim,200,8)
    # scheduler = WarmupCosineLrScheduler(optim, max_iter=200,warmup_iter=8)
    # scheduler = WarmupPolyLrScheduler(optim, power=3, max_iter=200,warmup_iter=8)
    # scheduler = WarmupStepLrScheduler(optim, gamma=3, warmup_iter=200)
    # scheduler = WarmupExpLrScheduler(optim, gamma=3, warmup_iter=200)

    a = []
    b = []
    for epoch in range(0, 200):
        scheduler.step(epoch)
        a.append(epoch)
        b.append(optim.param_groups[0]["lr"])
        print(epoch, optim.param_groups[0]["lr"])

    plt.plot(a, b)
    plt.savefig("aaa.png")
