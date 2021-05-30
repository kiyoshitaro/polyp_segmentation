import os
import skimage
import cv2
import nibabel as nib
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def save_img(path, img, lib="cv2", overwrite=True):
    if not overwrite and os.path.exists(path):
        pass
    else:
        print(path)
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if lib == "skimage":
            skimage.io.imsave(path, img)
        elif lib == "cv2":
            cv2.imwrite(path, img)
        elif lib == "nib":
            nib.save(nib.Nifti1Image(img, None), path)


def load_data(path):
    my_dir = sorted(os.listdir(path))
    data = []
    gt = []
    pr = []
    for p in tqdm(my_dir):
        data_list = sorted(os.listdir(path + p))
        print(len(data_list))
        if len(data_list) == 6:
            img_itk = sitk.ReadImage(path + p + "/" + data_list[1])
            flair = sitk.GetArrayFromImage(img_itk)
            img_itk = sitk.ReadImage(path + p + "/" + data_list[0])
            pred = sitk.GetArrayFromImage(img_itk)
            img_itk = sitk.ReadImage(path + p + "/" + data_list[2])
            seg = sitk.GetArrayFromImage(img_itk)
            img_itk = sitk.ReadImage(path + p + "/" + data_list[3])
            t1 = sitk.GetArrayFromImage(img_itk)
            img_itk = sitk.ReadImage(path + p + "/" + data_list[4])
            t1ce = sitk.GetArrayFromImage(img_itk)
            img_itk = sitk.ReadImage(path + p + "/" + data_list[5])
            t2 = sitk.GetArrayFromImage(img_itk)
        else:
            img_itk = sitk.ReadImage(path + p + "/" + data_list[1])
            flair = sitk.GetArrayFromImage(img_itk)
            img_itk = sitk.ReadImage(path + p + "/" + data_list[0])
            pred = sitk.GetArrayFromImage(img_itk)
            img_itk = sitk.ReadImage(path + p + "/" + data_list[0])
            seg = sitk.GetArrayFromImage(img_itk)
            img_itk = sitk.ReadImage(path + p + "/" + data_list[2])
            t1 = sitk.GetArrayFromImage(img_itk)
            img_itk = sitk.ReadImage(path + p + "/" + data_list[3])
            t1ce = sitk.GetArrayFromImage(img_itk)
            img_itk = sitk.ReadImage(path + p + "/" + data_list[4])
            t2 = sitk.GetArrayFromImage(img_itk)

        data.append([flair, t1, t1ce, t2])
        gt.append(seg)
        pr.append(pred)

    data = np.asarray(data, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.uint8)
    pr = np.asarray(pr, dtype=np.uint8)
    return data, gt, pr


def visualize_3d_slice(data, gt, pr, j):
    fig, axs = plt.subplots(7, 6, constrained_layout=True, figsize=(15, 15))
    cols = ["flair", "t1", "t1ce", "t2", "gt", "pr"]
    rows = list(range(50, 120, 10))
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation=0, size="large")

    for i in range(50, 120, 10):
        # axs[0].legend(bbox_to_anchor =(1, 1.6), ncol = 3)
        axs[(i - 50) // 10][0].imshow(data[j, 0, i, :, :])
        axs[(i - 50) // 10][0].set_axis_off()
        axs[(i - 50) // 10][1].imshow(data[j, 1, i, :, :])
        axs[(i - 50) // 10][1].set_axis_off()

        axs[(i - 50) // 10][2].imshow(data[j, 2, i, :, :])
        axs[(i - 50) // 10][2].set_axis_off()

        axs[(i - 50) // 10][3].imshow(data[j, 3, i, :, :])
        axs[(i - 50) // 10][3].set_axis_off()

        axs[(i - 50) // 10][4].imshow(gt[j, i, :, :])
        axs[(i - 50) // 10][4].set_axis_off()

        axs[(i - 50) // 10][5].imshow(pr[j, i, :, :])
        axs[(i - 50) // 10][5].set_axis_off()


# CONTROL
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith("keymap."):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(volume, gt, pr):
    remove_keymap_conflicts({"j", "k"})

    # fig, axs = plt.subplots(1, 6, constrained_layout=True)
    fig, axs = plt.subplots(1, 6, constrained_layout=True, figsize=(15, 15))
    axs[0].volume = volume[0]
    axs[0].index = volume[0].shape[0] // 2
    axs[0].imshow(volume[0][axs[0].index])
    # fig.canvas.mpl_connect("key_press_event", process_key)
    axs[0].set_title("flair")

    axs[1].volume = volume[1]
    axs[1].index = volume[1].shape[0] // 2
    axs[1].imshow(volume[1][axs[1].index])
    # fig.canvas.mpl_connect("key_press_event", process_key)
    axs[1].set_title("t1")

    axs[2].volume = volume[2]
    axs[2].index = volume[2].shape[0] // 2
    axs[2].imshow(volume[2][axs[2].index])
    # fig.canvas.mpl_connect("key_press_event", process_key)
    axs[2].set_title("t1ce")

    axs[3].volume = volume[3]
    axs[3].index = volume[3].shape[0] // 2
    axs[3].imshow(volume[3][axs[3].index])
    # fig.canvas.mpl_connect("key_press_event", process_key)
    axs[3].set_title("t2")

    axs[4].volume = gt
    axs[4].index = gt.shape[0] // 2

    axs[4].imshow(gt[axs[4].index])
    # fig.canvas.mpl_connect("key_press_event", process_key)
    axs[4].set_title("gt")

    axs[5].volume = pr
    axs[5].index = pr.shape[0] // 2
    axs[5].imshow(pr[axs[5].index])
    # fig.canvas.mpl_connect("key_press_event", process_key)
    axs[5].set_title("pr")

    fig.canvas.mpl_connect("key_press_event", process_key)

    # fig, ax = plt.subplots()
    # ax.volume = volume
    # ax.index = volume.shape[0] // 2
    # ax.imshow(volume[ax.index])
    # fig.canvas.mpl_connect("key_press_event", process_key)
    show()


def process_key(event):
    fig = event.canvas.figure
    for i in range(6):
        ax = fig.axes[i]
        if event.key == "j":
            previous_slice(ax)
        elif event.key == "k":
            next_slice(ax)
        fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 10) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 10) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


if __name__ == "__main__":
    path = "/Users/brown/code/polyp_segmentation/data/brats2018/"
    data1, gt1, pr1 = load_data(path)
    visualize_3d_slice(data1, gt1, pr1, 0)
    # multi_slice_viewer(data1[0, :, :, :, :], gt1[0, :, :, :], pr1[0, :, :, :])

    img_path = "data/kvasir-seg/TrainDataset/images/1.png"
    mask_path = "data/kvasir-seg/TrainDataset/masks/1.png"
    # mask_path = ''

    mask = np.array(imread(mask_path, as_gray=True))  # h , w (0-255), numpy
    if os.path.splitext(os.path.basename(img_path))[0].isnumeric():
        mask = mask / 255
    mask = np.asarray(mask.astype("float32"))

    from skimage.io import imread

    vis_x = 180
    img = np.asarray(imread(img_path))  # h, w , 3 (0-255), numpy
    # mask_img = (
    #     np.asarray(img)
    #     + 180
    #     * np.array((mask.round(), np.zeros_like(mask), np.zeros_like(mask))).transpose(
    #         (1, 2, 0)
    #     )
    #     + vis_x
    #     * np.array((np.zeros_like(mask), mask.round(), np.zeros_like(mask))).transpose(
    #         (1, 2, 0)
    #     )
    # )
    mask_img = np.asarray(img) + vis_x * np.array(
        (mask, np.zeros_like(mask), np.zeros_like(mask))
    ).transpose((1, 2, 0))

    mask_img = mask_img[:, :, ::-1]
    # / (mask_img.max() + 1e-8)
    # mask_img = mask_img.astype(int)
    plt.imshow(mask_img)
    # plt.imsave("test.png",mask_img)
    cv2.imwrite("test.png", img)

    img_path = "data/kvasir-seg/TrainDataset/images/1.png"
    mask_path = "data/kvasir-seg/TrainDataset/masks/1.png"
    image_ = imread(img_path)  # h, w , 3 (0-255), numpy
    if os.path.exists(mask_path):
        mask = imread(mask_path, as_gray=True)  # h , w (0-255), numpy
    else:
        mask = np.zeros(image_.shape[:2], dtype=np.float64)

    if os.path.splitext(os.path.basename(img_path))[0].isnumeric():
        mask = mask / 255

    image = cv2.resize(image_, (352, 352))
    image = image.astype("float32") / 255
    image = image.transpose((2, 0, 1))
    image = image[:, :, :, np.newaxis]
    image = image.transpose((3, 0, 1, 2))

    mask = mask.astype("float32")

    image, gt, filename, img = (
        np.asarray(image),
        np.asarray(mask),
        os.path.basename(img_path),
        np.asarray(image_),
    )

    mask_img = np.asarray(img) + vis_x * np.array(
        (gt, np.zeros_like(gt), np.zeros_like(gt))
    ).transpose((1, 2, 0))
    mask_img = mask_img[:, :, ::-1]

    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(15, 15))
    for i in range(4):
        axs[i].imshow(mask_img)
        axs[i].set_title("flair")
    fig.savefig("ax2_figure.png")
