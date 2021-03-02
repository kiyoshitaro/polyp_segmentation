import torch
from skimage.io import imread
import numpy as np
import os
import cv2


class PranetDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, mask_paths, img_size, transform=None, is_train=True):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image_ = imread(img_path)

        mask = imread(mask_path, as_gray=True)

        augmented = self.transform(image=image_, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        if os.path.splitext(os.path.basename(img_path))[0].isnumeric():
            mask = mask / 255

        if self.is_train:
            mask = cv2.resize(mask, (self.img_size, self.img_size))

        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype("float32") / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]

        mask = mask.astype("float32")
        mask = mask.transpose((2, 0, 1))
        if self.is_train:
            return np.asarray(image), np.asarray(mask)
        else:
            return (
                np.asarray(image),
                np.asarray(mask),
                os.path.basename(img_path),
                np.asarray(image_),
            )


def get_loader(
    image_paths,
    gt_paths,
    batchsize,
    img_size,
    transform,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    is_train=True,
):

    dataset = PranetDataset(
        image_paths, gt_paths, img_size, transform=transform, is_train=is_train
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader
