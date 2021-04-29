import pickle
import os
import functools
import itertools as IT
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import nibabel as nib
import glob

modalities = ("flair", "t1ce", "t1", "t2")


def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def savepkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


paths = [
    os.path.join(p, p.split("/")[-1])
    for p in glob.glob("data/MICCAI_BraTS_2018/Training/LGG/*")
]
from matplotlib.pyplot import imshow


for path in paths:
    label = np.array(nib_load(path + "_seg.nii.gz"), dtype="uint8", order="C")
    images = np.stack(
        [
            np.array(
                nib_load(path + "_" + modal + ".nii.gz"), dtype="float32", order="C"
            )
            for modal in modalities
        ],
        -1,
    )
    mask = images.sum(-1) > 0
    for k in range(4):
        x = images[..., k]  #
        y = x[mask]  #

        lower = np.percentile(y, 0.2)  # 算分位数
        upper = np.percentile(y, 99.8)

        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper
        y = x[mask]
        x -= y.mean()
        x /= y.std()
        images[..., k] = x

    output = (
        "data/MICCAI_BraTS_2018/Training_pkl/" + os.path.basename(path) + "data_f32.pkl"
    )
    print("saving:", output)
    savepkl(data=(images, label), path=output)
