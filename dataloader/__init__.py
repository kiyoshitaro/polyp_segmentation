from .kvasir import *
from .kvasir_distillation import *
from .isic import *
from .isic_distillation import *

from .usnerve import *
from .brats import *
from .org_kvasir import *
from .chase import *
from .em import *
import torch


def get_loader(
    image_paths,
    gt_paths,
    batchsize,
    img_size,
    transform,
    softlabel_paths=None,
    name="KvasirDataset",
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    type="train",
):
    print(f"Load by {name} dataloader")
    if "Distill" in name:
        dataset = globals()[name](
            image_paths,
            gt_paths,
            softlabel_paths,
            img_size,
            transform=transform,
            type=type,
        )
    else:
        dataset = globals()[name](
            image_paths, gt_paths, img_size, transform=transform, type=type
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader
