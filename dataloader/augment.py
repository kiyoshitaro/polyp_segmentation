import albumentations as al
import matplotlib.pyplot as plt
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf

import torch.nn as nn

class Augmenter(nn.Module):
    def __init__(self,
        prob=0.7,
        blur_prob=0.7,
        jitter_prob=0.7,
        rotate_prob=0.7,
        flip_prob=0.7,
    ):
        super().__init__()

        self.prob = prob
        self.blur_prob = blur_prob
        self.jitter_prob = jitter_prob
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob

        self.transforms = al.Compose([
            transforms.RandomRotate90(),
            transforms.Flip(),
            transforms.HueSaturationValue(),
            transforms.RandomBrightnessContrast(),
            transforms.Transpose(),
            OneOf([
              transforms.RandomCrop(220,220, p=0.5),
              transforms.CenterCrop(220,220, p=0.5)
            ], p=0.5),
            # transforms.Resize(352,352),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ], p=self.prob)
    
    def forward(self, image, mask):
        # image_n = image_t.numpy().transpose(1, 2, 0)
        # mask_n = mask_t.numpy().transpose(1, 2, 0)

        result = self.transforms(image=image, mask=mask)

        # image_n, mask_n = result["image"], result["mask"]
        # image_t = torch.from_numpy(image_n).permute(2, 0, 1)
        # mask_t = torch.from_numpy(mask_n).permute(2, 0, 1)

        # image = result['image']
        # mask = result['mask']

        return result
