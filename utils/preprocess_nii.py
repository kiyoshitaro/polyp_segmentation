"""
Load the 'nii' file and save as pkl file.
Carefully check your path please.
"""

import pickle
import os
import numpy as np
import nibabel as nib

modalities = ("flair", "t1ce", "t1", "t2")


train_set = {
    "root": "/data2/liuxiaopeng/Data/BraTS2018/Train",
    "flist": "all.txt",
}

valid_set = {
    "root": "/data/BraTS2018/Valid",
    "flist": "valid.txt",
}

test_set = {
    "root": "/data2/liuxiaopeng/Data/BraTS2018/Test",
    "flist": "test.txt",
}


def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def normalize(image, mask=None):
    assert len(image.shape) == 3  # shape is [H,W,D]
    assert image[0, 0, 0] == 0  # check the background is zero
    if mask is not None:
        mask = image > 0  # The bg is zero

    mean = image[mask].mean()
    std = image[mask].std()
    image = image.astype(dtype=np.float32)
    image[mask] = (image[mask] - mean) / std
    return image


def savepkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def process_f32(path):
    """ Set all Voxels that are outside of the brain mask to 0"""
    # label = np.array(nib_load(path + "seg.nii.gz"), dtype="uint8", order="C")
    images = np.stack(
        [
            np.array(
                nib_load(path + "_" + modal + ".nii.gz"), dtype="float32", order="C"
            )
            for modal in modalities
        ],
        -1,
    )

    # mask = images.sum(-1) > 0

    # for k in range(4):
    #     x = images[..., k]  #
    #     y = x[mask]  #

    #     lower = np.percentile(y, 0.2)  # 算分位数
    #     upper = np.percentile(y, 99.8)

    #     x[mask & (x < lower)] = lower
    #     x[mask & (x > upper)] = upper

    #     y = x[mask]

    #     x -= y.mean()
    #     x /= y.std()

    #     images[..., k] = x

    output = path + "data_f32.pkl"
    print("saving:", output)
    savepkl(data=(images), path=output)


def doit(dset):
    # root, has_label = dset["root"]
    # file_list = os.path.join(root, dset["flist"])
    # subjects = open(file_list).read().splitlines()
    # names = [sub.split("/")[-1] for sub in subjects]
    # paths = [os.path.join(root, sub, name + "_") for sub, name in zip(subjects, names)]
    paths = [
        os.path.join(p, p.split("/")[-1]) for p in glob.glob("data/BraTS_2018/Valid/*")
    ]
    for path in paths:
        process_f32(path)


doit(train_set)
doit(valid_set)

import functools
import itertools as IT
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import nibabel as nib
def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


import glob
import os 
paths = [
    os.path.join(p, p.split("/")[-1]) for p in glob.glob("data/BraTS_2018/Valid/*")
]
path = paths[0]
from matplotlib.pyplot import imshow
modalities = ("flair", "t1ce", "t1", "t2")

images = np.stack(
    [
        np.array(nib_load(path + "_" + modal + ".nii.gz"), dtype="float32", order="C")
        for modal in modalities
    ],
    -1,
)
image = images[:,:,:,0]
def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))

    plt.show()
    
show_histogram(image)
def scale_by(arr, fac):
    mean = np.mean(arr)
    return (arr-mean)*fac + mean

transformed = np.clip(scale_by(np.clip(normalize(image)-0.1, 0, 1)**0.4, 2)-0.1, 0, 1)
IMG_DIM = 50

from skimage.transform import resize
resized = resize(transformed, (IMG_DIM, IMG_DIM, IMG_DIM), mode='constant')

# import requests
# images = requests.get('http://www.fil.ion.ucl.ac.uk/spm/download/data/attention/attention.zip')
# import zipfile
# from io import BytesIO

# zipstream = BytesIO(images.content)
# zf = zipfile.ZipFile(zipstream)
# from nibabel import FileHolder
# from nibabel.analyze import AnalyzeImage

# header = BytesIO(zf.open('attention/structural/nsM00587_0002.hdr').read())
# image = BytesIO(zf.open('attention/structural/nsM00587_0002.img').read())
# img = AnalyzeImage.from_file_map({'header': FileHolder(fileobj=header), 'image': FileHolder(fileobj=image)})
# arr = img.get_fdata()
# arr.shape


from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMG_DIM = 50


def normalize(arr):
    arr_min = np.min(arr)
    return (arr - arr_min) / (np.max(arr) - arr_min)


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z


def plot_cube(cube, angle=320):
    cube = normalize(cube)

    facecolors = cm.viridis(cube)
    facecolors[:, :, :, -1] = cube
    facecolors = explode(facecolors)

    filled = facecolors[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
    ax = fig.gca(projection="3d")
    ax.view_init(30, angle)
    ax.set_xlim(right=IMG_DIM * 2)
    ax.set_ylim(top=IMG_DIM * 2)
    ax.set_zlim(top=IMG_DIM * 2)

    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    plt.show()