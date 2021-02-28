from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

import torch
class GradualWarmupScheduler(_LRScheduler):
    # https://github.com/seominseok0429/pytorch-warmup-cosine-lr
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


    def cosine_warmup(optimizer, init_lr, total_epoch, num_warmup_epoch):
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch, eta_min=init_lr, last_epoch=-1)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=num_warmup_epoch, after_scheduler=cosine_scheduler)
        return scheduler

    def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
        decay = decay_rate ** (epoch // decay_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay
    
if __name__ == "__main__":
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.01)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 100, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optim, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)
    a = []
    b = []
    for epoch in range(1, 100):
        scheduler.step(epoch)
        a.append(epoch)
        b.append(optim.param_groups[0]['lr'])
        print(epoch, optim.param_groups[0]['lr'])

    plt.plot(a,b)
    plt.show()
