import torch
import numpy as np
import os
import cv2
from PIL import Image


class ISICDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, mask_paths, img_size, transform=None, type="train"):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.transform = transform
        self.type = type

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image_ = np.array(Image.open(img_path).convert("RGB"))

        # image_ = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.type == "private":
            image = cv2.resize(image_, (self.img_size, self.img_size))
            image = image.astype("float32") / 255
            image = image.transpose((2, 0, 1))
            return (np.asarray(image), os.path.basename(img_path), np.asarray(image_))

        mask = np.array(Image.open(mask_path).convert("L"))

        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        augmented = self.transform(image=image_, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        mask_resize = mask
        mask = mask / 255

        if self.type == "train":
            # pass
            mask = cv2.resize(mask, (self.img_size, self.img_size))

        elif self.type == "val":
            # image = cv2.resize(image, (self.img_size, self.img_size))
            mask_resize = cv2.resize(mask, (self.img_size, self.img_size))

            mask_resize = mask_resize[:, :, np.newaxis]
            mask_resize = mask_resize.astype("float32")
            mask_resize = mask_resize.transpose((2, 0, 1))

        image = cv2.resize(image, (self.img_size, self.img_size))

        image = image.astype("float32") / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]
        mask = mask.astype("float32")
        mask = mask.transpose((2, 0, 1))

        if self.type == "train":
            return np.asarray(image), np.asarray(mask)

        elif self.type == "test":
            return (
                np.asarray(image),
                np.asarray(mask),
                os.path.basename(img_path),
                np.asarray(image_),
            )
        elif self.type == "val":
            return (
                np.asarray(image),
                np.asarray(mask),
                np.asarray(mask_resize),
            )
        else:
            return (np.asarray(image), os.path.basename(img_path), np.asarray(image_))
