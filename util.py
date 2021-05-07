from skimage.io import imread

import glob

train_mask = glob.glob("/Users/brown/code/PraNet/TrainDataset/mask/*")
test_mask = glob.glob("/Users/brown/code/PraNet/TestDataset/masks/*")
for i in test_mask:
    if 255 not in set(imread(i).reshape((-1,))):
        print(i)
        print(set(imread(i).reshape((-1,))))

import os

for id in range(5):
    fold = [
        os.path.basename(i) for i in glob.glob(f"Kvasir_fold_new/fold_{id}/masks/*")
    ]
    os.system(f"mkdir Kvasir_fold_new/fold_{id}/_masks/")
    for i in fold:
        cmd = f"cp all_masks/{i} Kvasir_fold_new/fold_{id}/_masks/"
        os.system(cmd)


def check_type_image(path):
    img = imread(path, as_gray=True)
    print("shape", img.shape)
    print("value", set(img.reshape((-1,))))


# Weighted Loss
import numpy as np
import torch
mask = imread("data/Kvasir_fold_new/fold_0/masks/cju0qoxqj9q6s0835b43399p4.png")
mask = mask[:, :, np.newaxis]
mask = mask.astype("float32")
mask = mask.transpose((2, 0, 1))
x=torch.from_numpy(mask)
weit = 1 + 5 * torch.abs(
    F.avg_pool2d(x, kernel_size=41, stride=1, padding=20) - x
)
plt.imshow(weit[0])
np.histogram(weit)



# AUGMENT
import albumentations as al
import matplotlib.pyplot as plt
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
image = imread("data/TrainDataset/images/cju2y5zas8m7f0801d34g5owq.png")

fig, axs = plt.subplots(1, 6, constrained_layout=True, figsize=(20, 20))
[axs[i].set_axis_off() for i in range(6)]

axs[0].imshow(image)
axs[0].set_title("Ảnh gốc")

image_ = transforms.RandomRotate90(always_apply=True)(image=image)["image"]
axs[1].imshow(image_)
axs[1].set_title("Xoay ngẫu nhiên")

image_ = transforms.Flip(always_apply=True)(image=image)["image"]
axs[2].imshow(image_)
axs[2].set_title("Lật")

image_ = transforms.HueSaturationValue(always_apply=True)(image=image)["image"]
axs[3].imshow(image_)
axs[3].set_title("Đổi tông màu")

image_ = transforms.RandomBrightnessContrast(always_apply=True)(image=image)["image"]
axs[4].imshow(image_)
axs[4].set_title("Chỉnh độ tương phản")

image_ = OneOf(
            [
                transforms.RandomCrop(204,250, p=1),
                transforms.CenterCrop(204, 250, p=1),
            ],
            p=1,
        )(image=image)["image"]
axs[5].imshow(image_)
axs[5].set_title("Cắt vùng ngẫu nhiên")