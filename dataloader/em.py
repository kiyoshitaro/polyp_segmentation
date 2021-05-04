import numpy as np
import torch
import albumentations as al
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from PIL import Image, ImageSequence


class ElectronMicroscopyDataset(Dataset):
    def __init__(self,
        image_path,
        mask_path,
        img_size=(1024, 768),
        transform=None,
        type="train"
    ) -> None:
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        if transform is None:
            transform = al.Compose([], p=0)

        self.img_size = img_size
        self.image_path = image_path
        self.mask_path = mask_path
        self.img_size = img_size
        self.transform = transform
        self.type = type

        self.__read_samples()

    def __len__(self):
        return len(self.__samples)

    def __read_samples(self):
        self.__samples = []

        images = Image.open(self.image_path)
        masks = Image.open(self.mask_path)

        for i in range(images.n_frames):
            images.seek(i)
            masks.seek(i)

            i_arr = np.asarray(images).astype(np.uint8)
            m_arr = np.asarray(masks).astype(np.uint8)
            self.__samples.append((i_arr, m_arr))

    def __getitem__(self, index):
        image, mask = self.__samples[index]

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        image_resize = cv2.resize(image, self.img_size)
        image_resize = np.stack([image_resize] * 3, axis=-1)
        image_resize = (image_resize.astype("float32") / 255.).transpose(2, 0, 1)

        image = np.stack([image] * 3, axis=-1)
        image = (image.astype("float32") / 255.).transpose(2, 0, 1)

        mask_resize = cv2.resize(mask, self.img_size)
        mask_resize = mask_resize[:, :, np.newaxis]
        mask_resize = (mask_resize / 255.).transpose(2, 0, 1)
        mask_resize = (mask_resize > 0.5).astype("float32")

        mask = mask[:, :, np.newaxis]
        mask = (mask / 255.).transpose(2, 0, 1)
        mask = (mask > 0.5).astype("float32")

        if self.type == "train":
            return image_resize, mask_resize
        elif self.type == "test":
            return (
                image_resize,
                mask,
                f"{self.image_path}::{index}",
                image
            )
        elif self.type == "val":
            return (
                image_resize,
                mask,
                mask_resize
            )


def test_1():
    dataset = ElectronMicroscopyDataset(
        "/home/lanpn/workspace/research/polypnet/data/EM/training.tif",
        "/home/lanpn/workspace/research/polypnet/data/EM/training_groundtruth.tif",
        img_size=512, type="train"
    )

    plt.ioff()
    fig, ax = plt.subplots(1, 2, figsize=(10, 9))
    image = dataset[-1][0].transpose((1, 2, 0))
    mask = dataset[-1][1].transpose((1, 2, 0))

    print(image.shape, mask.shape)
    print(np.sum(mask))
    ax[0].imshow(image)
    ax[1].imshow(mask)

    plt.show()


if __name__ == "__main__":
    test_1()
