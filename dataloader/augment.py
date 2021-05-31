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


class Augmenter(nn.Module):
    def __init__(
        self,
        prob=0,
        Flip_prob=0,
        HueSaturationValue_prob=0,
        RandomBrightnessContrast_prob=0,
        crop_prob=0,
        randomrotate90_prob=0,
        elastictransform_prob=0,
        gridistortion_prob=0,
        opticaldistortion_prob=0,
        verticalflip_prob=0,
        horizontalflip_prob=0,
        randomgamma_prob=0,
        CoarseDropout_prob=0,
        RGBShift_prob=0,
        MotionBlur_prob=0,
        MedianBlur_prob=0,
        GaussianBlur_prob=0,
        GaussNoise_prob=0,
        ChannelShuffle_prob=0,
    ):
        super().__init__()

        self.prob = prob
        self.randomrotate90_prob = randomrotate90_prob
        self.elastictransform_prob = elastictransform_prob

        self.transforms = al.Compose(
            [
                transforms.RandomRotate90(p=randomrotate90_prob),
                transforms.Flip(p=Flip_prob),
                transforms.HueSaturationValue(p=HueSaturationValue_prob),
                transforms.RandomBrightnessContrast(p=RandomBrightnessContrast_prob),
                transforms.Transpose(),
                OneOf(
                    [
                        transforms.RandomCrop(220, 220, p=0.5),
                        transforms.CenterCrop(220, 220, p=0.5),
                    ],
                    p=crop_prob,
                ),
                ElasticTransform(
                    p=elastictransform_prob,
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                ),
                GridDistortion(p=gridistortion_prob),
                OpticalDistortion(
                    p=opticaldistortion_prob, distort_limit=2, shift_limit=0.5
                ),
                VerticalFlip(p=verticalflip_prob),
                HorizontalFlip(p=horizontalflip_prob),
                RandomGamma(p=randomgamma_prob),
                RGBShift(p=RGBShift_prob),
                MotionBlur(p=MotionBlur_prob, blur_limit=7),
                MedianBlur(p=MedianBlur_prob, blur_limit=9),
                GaussianBlur(p=GaussianBlur_prob, blur_limit=9),
                GaussNoise(p=GaussNoise_prob),
                ChannelShuffle(p=ChannelShuffle_prob),
                CoarseDropout(
                    p=CoarseDropout_prob, max_holes=8, max_height=32, max_width=32
                ),
                # transforms.Resize(352, 352),
                # transforms.Normalize(
                #     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                # ),
            ],
            p=self.prob,
        )

    def forward(self, image, mask):
        result = self.transforms(image=image, mask=mask)

        return result

if __name__ =="__main__":
    from skimage.io import imread

    # AUGMENT
    import albumentations as al
    import matplotlib.pyplot as plt
    from albumentations.augmentations import transforms
    from albumentations.core.composition import Compose, OneOf
    image = imread("data/TrainDataset/images/cju2y5zas8m7f0801d34g5owq.png")

    fig, axs = plt.subplots(1, 18, constrained_layout=True, figsize=(20, 20))
    [axs[i].set_axis_off() for i in range(18)]

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
    image_ = transforms.ElasticTransform(
                        p=1,
                        alpha=120,
                        sigma=120 * 0.05,
                        alpha_affine=120 * 0.03,
                    )(image=image)["image"]
    axs[6].imshow(image_)
    axs[6].set_title("ElasticTransform")

    image_ = transforms.ElasticTransform(
                        p=1,
                        alpha=120,
                        sigma=120 * 0.05,
                        alpha_affine=120 * 0.03,
                    )(image=image)["image"]
    axs[6].imshow(image_)
    axs[6].set_title("ElasticTransform")

    image_ = transforms.GridDistortion(p=1)(image=image)["image"]
    axs[7].imshow(image_)
    axs[7].set_title("GridDistortion")

    image_ = transforms.OpticalDistortion(
                        p=1, distort_limit=2, shift_limit=0.5
                    )(image=image)["image"]
    axs[8].imshow(image_)
    axs[8].set_title("OpticalDistortion")

    image_ = transforms.RandomGamma(always_apply=True)(image=image)["image"]
    axs[9].imshow(image_)
    axs[9].set_title("RandomGamma")
    image_ = transforms.RGBShift(always_apply=True)(image=image)["image"]
    axs[10].imshow(image_)
    axs[10].set_title("RGBShift")
    image_ = transforms.MotionBlur(always_apply=True, blur_limit=7)(image=image)["image"]
    axs[11].imshow(image_)
    axs[11].set_title("MotionBlur")
    image_ = transforms.MedianBlur(always_apply=True,blur_limit=9)(image=image)["image"]
    axs[12].imshow(image_)
    axs[12].set_title("MedianBlur")
    image_ = transforms.GaussianBlur(always_apply=True,blur_limit=9)(image=image)["image"]
    axs[13].imshow(image_)
    axs[13].set_title("GaussianBlur")
    image_ = transforms.GaussNoise(always_apply=True)(image=image)["image"]
    axs[14].imshow(image_)
    axs[14].set_title("GaussNoise")
    image_ = transforms.ChannelShuffle(always_apply=True)(image=image)["image"]
    axs[15].imshow(image_)
    axs[15].set_title("ChannelShuffle")
    image_ = transforms.GaussNoise(always_apply=True)(image=image)["image"]
    axs[16].imshow(image_)
    axs[16].set_title("GaussNoise")
    image_ = transforms.CoarseDropout(always_apply=True,max_holes=8, max_height=32, max_width=32)(image=image)["image"]
    axs[17].imshow(image_)
    axs[17].set_title("CoarseDropout")
