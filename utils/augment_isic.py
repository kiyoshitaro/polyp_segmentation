import os
from glob import glob

train_img_paths = []
train_mask_paths = []
train_data_path = [
    "/home/admin_mcn/hung/polyp_segmentation/data/ISIC2018_BCDU/traindataset"
]
for i in train_data_path:
    train_img_paths.extend(glob(os.path.join(i, "images", "*")))
    train_mask_paths.extend(glob(os.path.join(i, "masks", "*")))
train_img_paths.sort()
train_mask_paths.sort()
print(f"There are {len(train_img_paths)} images to train")

import albumentations as al
import matplotlib.pyplot as plt
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout,
)
import torch.nn as nn
import torch
from skimage.io import imread
import numpy as np
import os
import cv2
from PIL import Image

crop_size = (192 - 32, 256 - 32)
size = (256, 256)
for i in range(len(train_data_path)):
    directory_imgaug = os.path.join(train_data_path[i] + "_256", "images")
    if not os.path.exists(directory_imgaug):
        os.makedirs(directory_imgaug)
    directory_maskaug = os.path.join(train_data_path[i] + "_256", "masks")
    if not os.path.exists(directory_maskaug):
        os.makedirs(directory_maskaug)

import tqdm

for id in tqdm.tqdm(range(len(train_img_paths))):
    img_path = train_img_paths[id]
    mask_path = train_mask_paths[id]
    # x = imread(img_path)
    # y = imread(mask_path, as_gray=True)

    if not os.path.exists(
        os.path.join(
            directory_imgaug, f"{os.path.splitext(os.path.basename(img_path))[0]}.jpg"
        )
    ):

        x = np.array(Image.open(img_path).convert("RGB"))  # h, w , 3 (0-255), numpy
        y = np.array(Image.open(mask_path).convert("L"))  # h , w (0-255), numpy

        # aug = CenterCrop(p=1, height=crop_size[0], width=crop_size[1])
        # augmented = aug(image=x, mask=y)
        # x1 = augmented['image']
        # y1 = augmented['mask']

        # ## Crop
        # x_min = 0
        # y_min = 0
        # x_max = x_min + size[0]
        # y_max = y_min + size[1]

        # aug = Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        # augmented = aug(image=x, mask=y)
        # x2 = augmented['image']
        # y2 = augmented['mask']

        # ## Random Rotate 90 degree
        # aug = RandomRotate90(p=1)
        # augmented = aug(image=x, mask=y)
        # x3 = augmented['image']
        # y3 = augmented['mask']

        # ## Transpose
        # aug = Transpose(p=1)
        # augmented = aug(image=x, mask=y)
        # x4 = augmented['image']
        # y4 = augmented['mask']

        # ## ElasticTransform
        # aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
        # augmented = aug(image=x, mask=y)
        # x5 = augmented['image']
        # y5 = augmented['mask']

        # ## Grid Distortion
        # aug = GridDistortion(p=1)
        # augmented = aug(image=x, mask=y)
        # x6 = augmented['image']
        # y6 = augmented['mask']

        # ## Optical Distortion
        # aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        # augmented = aug(image=x, mask=y)
        # x7 = augmented['image']
        # y7 = augmented['mask']

        # ## Vertical Flip
        # aug = VerticalFlip(p=1)
        # augmented = aug(image=x, mask=y)
        # x8 = augmented['image']
        # y8 = augmented['mask']

        # ## Horizontal Flip
        # aug = HorizontalFlip(p=1)
        # augmented = aug(image=x, mask=y)
        # x9 = augmented['image']
        # y9 = augmented['mask']

        # ## Grayscale
        # x10 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        # y10 = y

        # ## Grayscale Vertical Flip
        # aug = VerticalFlip(p=1)
        # augmented = aug(image=x10, mask=y10)
        # x11 = augmented['image']
        # y11 = augmented['mask']

        # ## Grayscale Horizontal Flip
        # aug = HorizontalFlip(p=1)
        # augmented = aug(image=x10, mask=y10)
        # x12 = augmented['image']
        # y12 = augmented['mask']

        # ## Grayscale Center Crop
        # aug = CenterCrop(p=1, height=crop_size[0], width=crop_size[1])
        # augmented = aug(image=x10, mask=y10)
        # x13 = augmented['image']
        # y13 = augmented['mask']

        # ##
        # aug = RandomBrightnessContrast(p=1)
        # augmented = aug(image=x, mask=y)
        # x14 = augmented['image']
        # y14 = augmented['mask']

        # aug = RandomGamma(p=1)
        # augmented = aug(image=x, mask=y)
        # x15 = augmented['image']
        # y15 = augmented['mask']

        # aug = HueSaturationValue(p=1)
        # augmented = aug(image=x, mask=y)
        # x16 = augmented['image']
        # y16 = augmented['mask']

        # aug = RGBShift(p=1)
        # augmented = aug(image=x, mask=y)
        # x17 = augmented['image']
        # y17 = augmented['mask']

        # aug = RandomBrightness(p=1)
        # augmented = aug(image=x, mask=y)
        # x18 = augmented['image']
        # y18 = augmented['mask']

        # aug = RandomContrast(p=1)
        # augmented = aug(image=x, mask=y)
        # x19 = augmented['image']
        # y19 = augmented['mask']

        # aug = MotionBlur(p=1, blur_limit=7)
        # augmented = aug(image=x, mask=y)
        # x20 = augmented['image']
        # y20 = augmented['mask']

        # aug = MedianBlur(p=1, blur_limit=9)
        # augmented = aug(image=x, mask=y)
        # x21 = augmented['image']
        # y21 = augmented['mask']

        # aug = GaussianBlur(p=1, blur_limit=9)
        # augmented = aug(image=x, mask=y)
        # x22 = augmented['image']
        # y22 = augmented['mask']

        # aug = GaussNoise(p=1)
        # augmented = aug(image=x, mask=y)
        # x23 = augmented['image']
        # y23 = augmented['mask']

        # aug = ChannelShuffle(p=1)
        # augmented = aug(image=x, mask=y)
        # x24 = augmented['image']
        # y24 = augmented['mask']

        # aug = CoarseDropout(p=1, max_holes=8, max_height=32, max_width=32)
        # augmented = aug(image=x, mask=y)
        # x25 = augmented['image']
        # y25 = augmented['mask']

        # images = [
        #     x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
        #     x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
        #     x21, x22, x23, x24, x25
        # ]
        # masks  = [
        #     y, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10,
        #     y11, y12, y13, y14, y15, y16, y17, y18, y19, y20,
        #     y21, y22, y23, y24, y25
        # ]
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = os.path.splitext(os.path.basename(mask_path))[0]
        idx = 0
        # for i, m in zip(x, y):
        i = cv2.resize(x, size)
        m = cv2.resize(y, size)

        tmp_image_name = f"{image_name}.jpg"
        tmp_mask_name = f"{mask_name}.jpg"
        image_path = os.path.join(directory_imgaug, tmp_image_name)
        mask_path = os.path.join(directory_maskaug, tmp_mask_name)

        cv2.imwrite(image_path, i)
        cv2.imwrite(mask_path, m)

        idx += 1
