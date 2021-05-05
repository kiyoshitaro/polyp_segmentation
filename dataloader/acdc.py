import os
import albumentations as al
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset

from .utils import resize_3dwh, pad_3d_depth


class AcdcDataset(Dataset):
    def __init__(self, dir: str,
        img_size=(256, 256),
        depth=24,
        transform=None,
        type="train"
    ) -> None:
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        if transform is None:
            transform = al.Compose([], p=0)

        self.dir = dir
        self.img_size = img_size
        self.transform = transform
        self.depth = depth
        self.type = type

        self.__get_paths()

    def __len__(self):
        return len(self.__paths)

    def __get_paths(self):
        self.__paths = []

        files1 = os.listdir(self.dir)

        for name1 in files1:
            path1 = os.path.join(self.dir, name1)
            if not os.path.isdir(path1):
                continue

            files2 = os.listdir(path1)
            for name2 in files2:
                if not name2.endswith(".nii.gz") or name2.endswith("4d.nii.gz") or name2.endswith("gt.nii.gz"):
                    continue
                path2 = os.path.join(path1, name2)
                noext = name2.split(".")[0]
                gt_name = f"{noext}_gt.nii.gz"
                gt_path = os.path.join(path1, gt_name)

                self.__paths.append((path2, gt_path))

    def __getitem__(self, index):
        img_path, mask_path = self.__paths[index]

        img: np.ndarray = nib.load(img_path).get_data()
        mask: np.ndarray = nib.load(mask_path).get_data()

        img = img.astype(np.uint8)
        img = resize_3dwh(img, self.img_size)
        if self.depth is not None:
            img = pad_3d_depth(img, self.depth)
        img = img.astype(np.float32) / 255.
        img = np.stack([img] * 3, axis=0)
        img = np.expand_dims(img, 0)

        mask = mask.astype(np.uint8)
        mask = resize_3dwh(mask, self.img_size)
        if self.depth is not None:
            mask = pad_3d_depth(mask, self.depth)
        mask = mask.astype(np.float32) / 255.
        mask = np.expand_dims(mask, 0)
        mask = np.expand_dims(mask, 0)

        print(img.shape)

        if self.type == "train":
            return img, mask
        elif self.type == "test":
            return (
                img,
                mask,
                os.path.basename(img_path),
                img
            )
        elif self.type == "val":
            return (
                img,
                mask,
                img
            )

def test_1():
    dataset = AcdcDataset(
        "/home/lanpn/workspace/research/polypnet/data/acdc-seg/training",
        img_size=256
    )

    print(len(dataset))
    for i in range(len(dataset)):
        dataset[i]


if __name__ == '__main__':
    test_1()
