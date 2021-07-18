import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision
import cv2
import re
import os
import pickle
from .augment3d import Uniform
from .augment3d import Rot90, Flip, Identity, Compose
from .augment3d import GaussianBlur, Noise, Normalize, RandSelect
from .augment3d import (
    RandCrop,
    CenterCrop,
    Pad,
    RandCrop3D,
    RandomRotion,
    RandomFlip,
    RandomIntensityChange,
)
from .augment3d import NumpyType


def pkload(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


class BraTSDataset(Dataset):
    def __init__(self, img_paths, mask_paths, img_size, transform=None, type="train"):
        self.paths = img_paths
        self.img_size = img_size
        self.transform = transform
        self.type = type
        self.transform = eval(transform or "Identity()")

    def __getitem__(self, index):
        path = self.paths[index]
        x_org, y = pkload(path)  # endwith data_f32.pkl
        # print(x_org.shape, y.shape)#(240, 240, 155, 4) (240, 240, 155)
        # transforms work with nhwtc

        x, y = x_org, y

        x, y = x[None, ...], y[None, ...]

        # # if(transforms):
        x, y = self.transform([x, y])
        x, y = x[0, :], y[0, :]

        x = np.ascontiguousarray(x.transpose(3, 0, 1, 2)).astype(
            np.float32
        )  # [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y).astype(np.float32)

        # print(x.shape, y.shape)  # (4, 240, 240, 155) (240, 240, 155), np.float32,  y label is 0,1,2,4,

        if self.type == "train":
            return x, y

        elif self.type == "test":
            return (
                x,
                y,
                os.path.basename(path),
                np.asarray(x_org),
            )
        else:
            return (
                x,
                y,
                y,
            )

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":
    import albumentations as al
    from albumentations.augmentations import transforms
    from albumentations.core.composition import Compose, OneOf
    import matplotlib.pyplot as plt
    from glob import glob

    train_img_paths = []
    train_mask_paths = []
    train_data_path = ["data/kvasir-seg/TrainDataset"]
    for i in train_data_path:
        train_img_paths.extend(glob(os.path.join(i, "images", "*")))
        train_mask_paths.extend(glob(os.path.join(i, "masks", "*")))
    train_img_paths.sort()
    train_mask_paths.sort()

    transforms = al.Compose(
        [
            transforms.RandomRotate90(),
            transforms.Flip(),
            transforms.HueSaturationValue(),
            transforms.RandomBrightnessContrast(),
            transforms.Transpose(),
            OneOf(
                [
                    transforms.RandomCrop(220, 220, p=0.5),
                    transforms.CenterCrop(220, 220, p=0.5),
                ],
                p=0.5,
            ),
        ],
        p=0.7,
    )
    dataset = USNerveDataset(
        train_img_paths, train_mask_paths, 352, transform=transforms, type="train"
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 9))
    image = dataset[0][0].transpose((1, 2, 0))
    mask = dataset[0][1].transpose((1, 2, 0))

    ax[0].imshow(image)
    ax[1].imshow(mask)