from dataloader import get_loader
from dataloader.augment import Augmenter
import tqdm
import torch
import os
from glob import glob
from utils.visualize import save_img
from utils.metrics import *
import numpy as np
import torch.nn.functional as F
from loguru import logger
from skimage.io import imread
import cv2

def main():
    
    img_path = "data/kvasir-seg/TrainDataset/images/1.png"
    # mask_path = "data/kvasir-seg/TrainDataset/masks/1.png"
    mask_path = ''

    import network.models as models
    # arch = model_prams["arch"]
    arch = "PraNet"
    model = models.__dict__[arch]()
    model_path = "pretrained/PraNet-19.pth"
    device = torch.device("cpu")
    model.cuda()
    model.eval()
    logger.info(f"Loading from {model_path}")
    try:
        model.load_state_dict(torch.load(model_path)["model_state_dict"])
        # model.load_state_dict(torch.load(model_path))
    except:
        model.load_state_dict(torch.load(model_path))




    image_ = imread(img_path)  # h, w , 3 (0-255), numpy
    if(os.path.exists(mask_path)):
        mask = imread(mask_path, as_gray=True)  # h , w (0-255), numpy
    else:
        mask = np.zeros(image_.shape[:2],dtype = np.float64)

    if os.path.splitext(os.path.basename(img_path))[0].isnumeric():
        mask = mask / 255

    image = cv2.resize(image_, (352,352))
    image = image.astype("float32") / 255
    image = image.transpose((2, 0, 1))
    image = image[:, :, :, np.newaxis]
    image = image.transpose((3, 0, 1, 2))

    mask = mask.astype("float32")

    image, gt, filename, img = np.asarray(image), np.asarray(mask), os.path.basename(img_path), np.asarray(image_)

    name = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    gt = np.asarray(gt, np.float32)
    # gt /= gt.max() + 1e-8
    res2 = 0
    image = torch.tensor(image).float().cuda()
    # image = image.cpu()

    res5, res4, res3, res2 = model(image)
    res = res2
    res = F.upsample(
        res, size=gt.shape, mode="bilinear", align_corners=False
    )
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)

    overwrite = True
    vis_x = 180
    visualize_dir = "output"
    save_img(
        os.path.join(
            visualize_dir,
            str(arch),
            name + "_pr" + str(arch) + ext,
        ),
        res.round() * 255,
        "cv2",
        overwrite,
    )
    save_img(
        os.path.join(
            visualize_dir,
            "soft_" + str(arch),
            name + "_soft_pr" + str(arch) + ext,
        ),
        res * 255,
        "cv2",
        overwrite,
    )
    mask_img = (
        np.asarray(img)
        + vis_x
        * np.array(
            (
                np.zeros_like(res.round()),
                res.round(),
                np.zeros_like(res.round()),
            )
        ).transpose((1, 2, 0))
        + vis_x
        * np.array(
            (gt, np.zeros_like(gt), np.zeros_like(gt))
        ).transpose((1, 2, 0))
    )
    mask_img = mask_img[:, :, ::-1]
    save_img(
        os.path.join(
            visualize_dir,
            "mask_" + str(arch),
            name + "mask_pr" + str(arch) + ext,
        ),
        mask_img,
        "cv2",
        overwrite,
    )

    pr = res.round()
    tp = np.sum(gt * pr)
    fp = np.sum(pr) - tp
    fn = np.sum(gt) - tp



    mean_precision = precision_m(gt, pr)
    mean_recall = recall_m(gt, pr)
    mean_iou = jaccard_m(gt, pr)
    mean_dice = dice_m(gt, pr)
    mean_F2 = (5 * precision_m(gt, pr) * recall_m(gt, pr)) / (
        4 * precision_m(gt, pr) + recall_m(gt, pr)
    )
    # mean_acc += (tp+tn)/(tp+tn+fp+fn)
    logger.info(
        "scores ver1: {:.3f} {:.3f} {:.3f} {:.3f}".format(
            mean_iou,
            mean_precision,
            mean_recall,
            mean_dice
            # , mean_F2
        )
    )



    precision_all = tp / (tp + fp + 1e-07)
    recall_all = tp / (tp + fn + 1e-07)
    dice_all = 2 * precision_all * recall_all / (precision_all + recall_all)
    iou_all = (
        recall_all
        * precision_all
        / (recall_all + precision_all - recall_all * precision_all)
    )
    logger.info(
        "scores ver2: {:.3f} {:.3f} {:.3f} {:.3f}".format(
            iou_all, precision_all, recall_all, dice_all
        )
    )

if __name__ == "__main__":
    main()