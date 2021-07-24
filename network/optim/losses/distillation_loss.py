from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F

from .focal_loss import FocalLoss
import numpy as np


class distillation_loss(_Loss):
    def __init__(self):
        super(distillation_loss, self).__init__()

    def forward(
        self, pred: torch.Tensor, mask: torch.Tensor, softlabel: torch.Tensor
    ) -> torch.Tensor:
        # print(np.histogram(mask.detach().cpu().numpy()))
        # print(np.histogram(softlabel.detach().cpu().numpy()))
        # print(np.histogram(pred.detach().cpu().numpy()))
        # import sys

        # sys.exit()
        # print(pred.shape, mask.shape)  # (bs, 1, h, w) (bs, 1, h,w)
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )

        softlabel = torch.sigmoid(softlabel)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
        wdistill = F.binary_cross_entropy_with_logits(pred, softlabel, reduce="none")

        wbce = (weit * (wbce + wdistill) / 2).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        inter2 = ((pred * softlabel) * weit).sum(dim=(2, 3))
        union2 = ((pred + softlabel) * weit).sum(dim=(2, 3))
        wiou2 = 1 - (inter2 + 1) / (union2 - inter2 + 1)

        wiou = (wiou + wiou2) / 2
        return (wbce + wiou).mean()
        # return (wfocal + wiou).mean()
