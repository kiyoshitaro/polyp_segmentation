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
from argparse import ArgumentParser
from utils.config import load_cfg
from datetime import datetime
import os


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=False, default="configs/gcpa_gald_net_config.yaml"
    )
    args = parser.parse_args()

    logger.info("Loading config")
    config_path = args.config
    config = load_cfg(config_path)

    folds = config["test"]["folds"]
    print(folds)
    # dataset = config["dataset"]["test_data_path"][0].split("/")[-1]
    dev = config["test"]["dev"]

    # img_paths = glob.glob("/mnt/data/hungnt/data/kvasir-seg/TestDataset/Kvasir/images/*")
    # img_paths = glob.glob("data/kvasir-instrument/testdataset/images/*")
    # img_paths = glob.glob("data/Kvasir_SEG/Kvasir_SEG_Validation_120/images/*")
    # img_paths = glob.glob("/mnt/data/hungnt/data/CHASE_OFF/test/images/*")

    # mask_paths = glob.glob("/mnt/data/hungnt/data/kvasir-seg/TestDataset/Kvasir/masks/*")
    # mask_paths = glob.glob("data/kvasir-instrument/testdataset/masks/*")
    # mask_paths = glob.glob("data/Kvasir_SEG/Kvasir_SEG_Validation_120/masks/*")
    # mask_paths = glob.glob("/mnt/data/hungnt/data/CHASE_OFF/test/masks/*")

    if type(config["infer"]["mask_paths"]) != list:
        mask_paths = glob.glob(os.path.join(config["infer"]["mask_paths"], "masks", "*"))
    else:
        mask_paths = config["infer"]["mask_paths"]
    if type(config["infer"]["img_paths"]) != list:
            img_paths = glob.glob(os.path.join(config["infer"]["img_paths"], "images", "*"))
    else:
        img_paths = config["infer"]["img_paths"]    

    # mask_path = ''
    img_paths.sort()
    mask_paths.sort()

    import network.models as models
    img_size = (config["test"]["dataloader"]["img_size"],config["test"]["dataloader"]["img_size"])
    arch_path = config["infer"]["models"]
    numpy_vertical = []
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(
        len(arch_path) + 2, len(mask_paths), constrained_layout=True, figsize=(15, 15)
    )
    cols = [os.path.basename(i) for i in mask_paths]
    rows = ["Images", "GT"]
    # rows = []
    [rows.append(i[0]) for i in arch_path]
    print(rows)
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, fontsize=20)
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation="horizontal", fontsize=20)
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
        mean_F2 = []
        mean_precisions = []
        mean_recalls = []
        mean_ious = []
        mean_accs = []
        mean_ses = []
        mean_spes = []
        tps = []
        fps = []
        fns = []
        tns = []


        cc = -1
        for img_path, mask_path in zip(img_paths, mask_paths):
            cc += 1
            image_ = imread(img_path)  # h, w , 3 (0-255), numpy
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"))
                # mask = mask / 255

            else:
                print("not exist")
                mask = np.zeros(image_.shape[:2], dtype=np.float64)

            # if os.path.splitext(os.path.basename(img_path))[0].isnumeric():

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
            tn = np.sum((1 - pr) * (1 - gt))
            tps.append(tp/(tp+fn))
            fps.append(fp/(fp+tn))
            fns.append(fn/(fn+tp))
            tns.append(tn/(tn+fp))

            mean_acc = (tp + tn) / (tp + tn + fp + fn)

            mean_se = tp / (tp + fn)
            mean_spe = tn / (tn + fp)


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
            mean_accs.append(mean_dice)
            mean_ses.append(mean_dice)
            mean_spes.append(mean_dice)

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

            overwrite = config["infer"]["vis_overwrite"]
            vis_x = config["infer"]["vis_x"]
            if "visualize_dir" not in config["infer"]:
                visualize_dir = "outputs/infer"
            else:
                visualize_dir = config["infer"]["visualize_dir"]
            if not os.path.exists(visualize_dir):
                os.makedirs(visualize_dir)

            ##### IMAGE
            imgs.append(np.asarray(img[:, :, ::-1]))
            axs[0][cc].imshow(img / (img.max() + 1e-8), cmap="gray")
            axs[0][cc].set_axis_off()

            ##### HARD PR
            ress.append(res.round() * 255)
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

            mask_img = np.asarray(img) + vis_x * np.array(
                (gt, np.zeros_like(gt), np.zeros_like(gt))
                # (np.zeros_like(gt), gt, np.zeros_like(gt))
                # (gt, gt, np.zeros_like(gt))
            ).transpose((1, 2, 0))

            axs[1][cc].imshow(gt, cmap="gray")
            axs[1][cc].set_axis_off()

            mask_img = mask_img[:, :, ::-1]

            ##### MASK GT
            mask_img_gt.append(mask_img)
            save_img(
                os.path.join(
                    visualize_dir,
                    str(arch),
                    name + "_gt" + str(arch) + ext,
                ),
                mask_img,
                "cv2",
                overwrite,
            )

            ##### SOFT PR
            soft_ress.append(res * 255)
            # axs[c + 2][cc].imshow(res * 255, cmap="gray")
            # axs[c + 2][cc].set_axis_off()

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
                    # (gt, gt, np.zeros_like(gt))
                ).transpose((1, 2, 0))
            )
            axs[c + 2][cc].imshow(res.round(),cmap = 'gray')
            axs[c + 2][cc].set_axis_off()
            mask_img = mask_img[:, :, ::-1]

            ##### MASK GT_PR

            mask_img_gt_pr.append(mask_img)
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
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    fig.savefig(os.path.join(visualize_dir,config["infer"]["compare_fig"]))

    s = [
        list(a)
        for a in zip(
            [os.path.basename(o) for o in mask_paths],
            mean_dices,
            mean_ious,
            mean_precisions,
            mean_recalls,
            mean_accs,
            mean_ses,
            mean_spes,
            tps,
            fps,
            fns,
            tns,

        )
    ]
    s.sort(key=lambda x: x[1])
    for i in s:
        logger.info(i)
    
    import pandas as pd
    pd.DataFrame(s,columns = ["name","mean_dices", "mean_ious", "mean_precisions", "mean_recalls","mean_accs","mean_ses","mean_spes","tps","fps","fns","tns"]).to_csv(os.path.join(visualize_dir,config["infer"]["compare_csv"]))

    # imgs = np.hstack(imgs)
    # mask_img_gt = np.hstack(mask_img_gt)
    # soft_ress = np.hstack(soft_ress)
    # ress = np.hstack(ress)
    # mask_img_gt_pr = np.hstack(mask_img_gt_pr)
    #     numpy_vertical.append(mask_img_gt_pr)
    # numpy_vertical.append(imgs)
    # cv2.imwrite("ppp.png", np.vstack(numpy_vertical))


if __name__ == "__main__":
    main()