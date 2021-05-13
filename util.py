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


from skimage.io import imread
mask_path = "/Users/brown/code/polyp_segmentation/data/kvasir-seg/TrainDataset/masks/cju88q6h6obpd0871ckmiabbo.png"
mask = imread(mask_path, as_gray=True)
mask_ = mask[:, :, np.newaxis]

mask_ = mask_.astype("float32")
mask_ = mask_.transpose((2, 0, 1))

mask_ = torch.from_numpy(mask_)
weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask_, kernel_size=31, stride=1, padding=15) - mask_
        )
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 8))
[axs[i].set_axis_off() for i in range(2)]

axs[0].imshow(mask)
axs[0].set_title("Nhãn")
axs[1].imshow(weit[0])
axs[1].set_title("Trọng số")

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