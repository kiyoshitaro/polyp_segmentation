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
import glob
from PIL import Image


def main():

    logger.add(
        f"test_HardMseg_Clinic.log",
        rotation="10 MB",
    )
    dev = "cpu"
    # dev = "cuda"

    img_paths = glob.glob("clinic_test/images/*")
    mask_paths = glob.glob("clinic_test/masks/*")
    img_paths.sort()
    mask_paths.sort()

    import network.models as models
    img_size = (352,352)

    arch_path = [
        ("UNet", "weights/unet_99.pth"),
        ("PraNet", "weights/pranet-19.pth"),
        ("SCWSRCCANet", "weights/scws_rcca_178.pth"),
    ]
    numpy_vertical = []
    import matplotlib.pyplot as plt
    c = -1
    for (arch, model_path) in arch_path:
        c += 1
        model = models.__dict__[arch]()

        if dev == "cpu":
            model.cpu()
        else:
            model.cuda()
        model.eval()
        logger.info(f"Loading from {model_path}")
        device = torch.device(dev)

        try:
            # model.load_state_dict(torch.load(model_path)["model_state_dict"])
            model.load_state_dict(
                torch.load(model_path, map_location=device)["model_state_dict"]
            )
        except:
            # model.load_state_dict(torch.load(model_path))
            model.load_state_dict(torch.load(model_path, map_location=device))

        mask_img_gt = []
        soft_ress = []
        ress = []
        mask_img_gt_pr = []
        imgs = []
        mean_dices = []
        mean_precisions = []
        mean_recalls = []
        mean_ious = []
        cc = -1
        for img_path, mask_path in zip(img_paths, mask_paths):
            cc += 1
            image_ = imread(img_path)  # h, w , 3 (0-255), numpy
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"))
            else:
                print("not exist")
                mask = np.zeros(image_.shape[:2], dtype=np.float64)

            image = cv2.resize(image_, img_size)
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

            name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            gt = np.asarray(gt, np.float32)
            gt /= gt.max() + 1e-8
            res2 = 0
            # image = torch.tensor(image).float().cuda()
            if dev == "cpu":
                image = torch.tensor(image).float()
            else:
                image = torch.tensor(image).float().cuda()

            # image = image.cpu()
            if arch == "UNet":
                res2 = model(image)
            else:
                res5, res4, res3, res2 = model(image)
            res = res2
            res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

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
            mean_ious.append(mean_iou)
            mean_precisions.append(mean_precision)
            mean_recalls.append(mean_recall)
            mean_dices.append(mean_dice)

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

            overwrite = True
            vis_x = 200
            visualize_dir = "outputs"

            ##### HARD PR
            ress.append(res.round() * 255)
            save_img(
                os.path.join(
                    visualize_dir,
                    "PR_" + str(arch),
                    name + "_hard_pr" + str(arch) + ext,
                ),
                res.round() * 255,
                "cv2",
                overwrite,
            )

            mask_img = np.asarray(img) + vis_x * np.array(
                (gt, np.zeros_like(gt), np.zeros_like(gt))
            ).transpose((1, 2, 0))

            mask_img = mask_img[:, :, ::-1]

            ##### HARD GT
            mask_img_gt.append(mask_img)
            save_img(
                os.path.join(
                    visualize_dir,
                    "GT_" + str(arch),
                    name + "_hard_gt" + str(arch) + ext,
                ),
                mask_img.round(),
                "cv2",
                overwrite,
            )

            ##### SOFT PR
            soft_ress.append(res * 255)

            save_img(
                os.path.join(
                    visualize_dir,
                    "PR_" + str(arch),
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
                    # (gt, gt, np.zeros_like(gt))
                ).transpose((1, 2, 0))
            )
            mask_img = mask_img[:, :, ::-1]

            ##### MASK GT_PR

            mask_img_gt_pr.append(mask_img)
            save_img(
                os.path.join(
                    visualize_dir,
                    "GT_PR_" + str(arch),
                    name + str(arch) + ext,
                ),
                mask_img,
                "cv2",
                overwrite,
            )

if __name__ == "__main__":
    main()