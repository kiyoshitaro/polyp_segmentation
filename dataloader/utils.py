import numpy as np
import math
import cv2


def resize_3dwh(img, size):
    out_slices = []
    h, w, d = img.shape

    for i in range(d):
        slice = img[:, :, i]
        out_slices.append(cv2.resize(slice, size))

    return np.stack(out_slices, axis=-1)


def pad_3d_depth(img, depth):
    h, w, orig_d = img.shape[:3]
    if orig_d > depth:
        raise ValueError(f"Cannot pad depth {orig_d} to {depth}")

    pad_count = depth - orig_d
    upper_pad_count = math.floor(pad_count / 2)
    lower_pad_count = math.ceil(pad_count / 2)

    upper_pad = np.zeros((h, w, upper_pad_count) + img.shape[3:])
    lower_pad = np.zeros((h, w, lower_pad_count) + img.shape[3:])

    out = np.concatenate([upper_pad, img, lower_pad], axis=2)
    return out
