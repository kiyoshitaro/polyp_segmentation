from .kvasir import *
from .isic import *
from .usnerve import *
import torch


def get_loader(
    image_paths,
    gt_paths,
    batchsize,
    img_size,
    transform,
    name="KvasirDataset",
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    type="train",
):
    print(f"Load by {name} dataloader")
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
