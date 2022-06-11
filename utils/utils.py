import torch
import numpy as np
from thop import profile
from thop import clever_format

import torch
import matplotlib.pyplot as plt


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



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
        return torch.mean(
            torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0) :])
        )


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
    print("[Statistics Information]\nFLOPs: {}\nParams: {}".format(flops, params))

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

def rle_encoding(x):
    dots = np.where(x.T.flatten()==1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
from skimage.morphology import label
def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def check_type_image(path):
    img = imread(path, as_gray=True)
    print("shape", img.shape)
    print("value", set(img.reshape((-1,))))

if __name__ == "__main__":
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.01)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, 100, eta_min=0, last_epoch=-1
    )
    scheduler = GradualWarmupScheduler(
        optim, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler
    )
    a = []
    b = []
    for epoch in range(1, 100):
        scheduler.step(epoch)
        a.append(epoch)
        b.append(optim.param_groups[0]["lr"])
        print(epoch, optim.param_groups[0]["lr"])

    plt.plot(a, b)
    plt.show()
