import torch
from skimage.io import imread
from PIL import Image

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


class OrgKvasirDataset(torch.utils.data.Dataset):
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
        image_ = np.array(
            Image.open(img_path).convert("RGB")
        )  # h, w , 3 (0-255), numpy
        mask = np.array(Image.open(mask_path).convert("L"))  # h , w (0-255), numpy

        # image_ = imread(img_path)  # h, w , 3 (0-255), numpy
        # mask = imread(mask_path, as_gray=True)  # h , w (0-255), numpy

        augmented = self.transform(image=image_, mask=mask)
        image = augmented["image"]
        if os.path.splitext(os.path.basename(img_path))[0].isnumeric():
            mask = augmented["mask"] / 255

        image = image.astype("float32")
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]

        mask = mask.astype("float32")
        mask = mask.transpose((2, 0, 1))

        return (
            np.asarray(image),
            np.asarray(mask),
            os.path.basename(img_path),
            np.asarray(image_),
        )


if __name__ == "__main__":
    import albumentations as al
    from albumentations.augmentations import transforms
    from albumentations.core.composition import Compose, OneOf

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
            # transforms.Resize(352,352),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
        p=0.7,
    )
    dataset = KvasirDataset(
        train_img_paths, train_mask_paths, 352, transform=transforms, type="train"
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 9))
    image = dataset[0][0].transpose((1, 2, 0))
    mask = dataset[0][1].transpose((1, 2, 0))

    ax[0].imshow(image)
    ax[1].imshow(mask)

    np.histogram(image)
