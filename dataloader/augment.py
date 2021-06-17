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

    def forward(self, image, mask,softlabel = None):
        # image_n = image_t.numpy().transpose(1, 2, 0)
        # mask_n = mask_t.numpy().transpose(1, 2, 0)
        if(softlabel is None ):
            result = self.transforms(image=image, mask=mask)
        else:
            result = self.transforms(image=image, mask=mask,softlabel=softlabel)

        # image_n, mask_n = result["image"], result["mask"]
        # image_t = torch.from_numpy(image_n).permute(2, 0, 1)
        # mask_t = torch.from_numpy(mask_n).permute(2, 0, 1)

        # image = result['image']
        # mask = result['mask']

        return result
