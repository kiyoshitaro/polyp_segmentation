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


def main():

    logger.add(
        f"test_HardMseg_Clinic.log",
        rotation="10 MB",
    )

    # img_paths = glob.glob("data/kvasir-seg/TestDataset/Kvasir/images/*")
    # mask_paths = glob.glob("data/kvasir-seg/TestDataset/Kvasir/masks/*")

    # cju3ykamdj9u208503pygyuc8
    # cju3uhb79gcgr0871orbrbi3x
    # 425
    # 205
    # 251
    # 73
    # 106
    # 21
    # 119

    # cju30ajhw09sx0988qyahx9s8
    # cju3xga12iixg0817dijbvjxw
    # cju2mh8t6p07008350e01tx2a
    # 25
    # 21
    # 14
    img_paths = [
        # "data/kvasir-seg/TrainDataset/images/1.png",
        # "data/kvasir-seg/TrainDataset/images/27.png",
        # "data/kvasir-seg/TrainDataset/images/562.png",
        # "data/kvasir-seg/TrainDataset/images/75.png",
        # "data/kvasir-seg/TestDataset/CVC-ClinicDB/images/425.png",
        # "data/kvasir-seg/TestDataset/CVC-ClinicDB/images/205.png",
        # "data/kvasir-seg/TestDataset/CVC-ClinicDB/images/119.png",
        # "data/kvasir-seg/TestDataset/CVC-ClinicDB/images/251.png",
        # "data/kvasir-seg/TestDataset/CVC-ClinicDB/images/73.png",

        "data/kvasir-seg/TestDataset/CVC-ClinicDB/images/21.png",
        "data/kvasir-seg/TestDataset/CVC-ClinicDB/images/14.png",
        "data/kvasir-seg/TestDataset/CVC-ClinicDB/images/25.png",
        "data/kvasir-seg/TestDataset/CVC-ClinicDB/images/154.png",
        "data/kvasir-seg/TestDataset/CVC-ClinicDB/images/205.png",
        "data/kvasir-seg/TestDataset/CVC-ClinicDB/images/73.png",

        # "data/kvasir-seg/TestDataset/Kvasir/images/cju2mh8t6p07008350e01tx2a.png",
        # "data/kvasir-seg/TestDataset/Kvasir/images/cju3uhb79gcgr0871orbrbi3x.png",
        # "data/kvasir-seg/TestDataset/Kvasir/images/cju5yeqiwmkgl0801fzv2douc.png",
        # "data/kvasir-seg/TestDataset/Kvasir/images/cju16whaj0e7n0855q7b6cjkm.png",
        # "data/kvasir-seg/TestDataset/Kvasir/images/cju2y40d8ulqo0993q0adtgtb.png",
        # "data/kvasir-seg/TestDataset/Kvasir/images/cju1ddr6p4k5z08780uuuzit2.png",

        # "data/kvasir-seg/TestDataset/Kvasir/images/cju30ajhw09sx0988qyahx9s8.png",
        # "data/kvasir-seg/TestDataset/Kvasir/images/cju30ajhw09sx0988qyahx9s8.png",
    ]

    mask_paths = [
        # "data/kvasir-seg/TrainDataset/masks/1.png",
        # "data/kvasir-seg/TrainDataset/masks/27.png",
        # "data/kvasir-seg/TrainDataset/masks/562.png",
        # "data/kvasir-seg/TrainDataset/masks/75.png",
        # "data/kvasir-seg/TestDataset/CVC-ClinicDB/masks/425.png",
        # "data/kvasir-seg/TestDataset/CVC-ClinicDB/masks/205.png",
        # "data/kvasir-seg/TestDataset/CVC-ClinicDB/masks/119.png",
        # "data/kvasir-seg/TestDataset/CVC-ClinicDB/masks/251.png",
        # "data/kvasir-seg/TestDataset/CVC-ClinicDB/masks/73.png",
        
        "data/kvasir-seg/TestDataset/CVC-ClinicDB/masks/21.png",
        "data/kvasir-seg/TestDataset/CVC-ClinicDB/masks/14.png",
        "data/kvasir-seg/TestDataset/CVC-ClinicDB/masks/25.png",
        "data/kvasir-seg/TestDataset/CVC-ClinicDB/masks/154.png",
        "data/kvasir-seg/TestDataset/CVC-ClinicDB/masks/205.png",
        "data/kvasir-seg/TestDataset/CVC-ClinicDB/masks/73.png",


        # "data/kvasir-seg/TestDataset/Kvasir/masks/cju2mh8t6p07008350e01tx2a.png",
        # "data/kvasir-seg/TestDataset/Kvasir/masks/cju3uhb79gcgr0871orbrbi3x.png",
        # "data/kvasir-seg/TestDataset/Kvasir/masks/cju5yeqiwmkgl0801fzv2douc.png",
        # "data/kvasir-seg/TestDataset/Kvasir/masks/cju16whaj0e7n0855q7b6cjkm.png",
        # "data/kvasir-seg/TestDataset/Kvasir/masks/cju2y40d8ulqo0993q0adtgtb.png",
        # "data/kvasir-seg/TestDataset/Kvasir/masks/cju1ddr6p4k5z08780uuuzit2.png",

        # "data/kvasir-seg/TestDataset/Kvasir/masks/cju30ajhw09sx0988qyahx9s8.png",
        # "data/kvasir-seg/TestDataset/Kvasir/masks/cju30ajhw09sx0988qyahx9s8.png",

    ]

    # mask_path = ''
    img_paths.sort()
    mask_paths.sort()

    import network.models as models

    arch_path = [
        ("UNet", "snapshots/UNet_kfold/PraNetDG-fold5-99.pth"),
        ("PraNet", "pretrained/PraNet-19.pth"),
        ("SCWSRCCANet", "snapshots/SCWSRCCANet_kfold/PraNetDG-fold5-178.pth"),
    ]
    numpy_vertical = []
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(
        len(arch_path) + 2, len(mask_paths), constrained_layout=True, figsize=(15, 15)
    )
    cols = [os.path.basename(i) for i in mask_paths]
    rows = ["Images","GT","UNet","PraNet","Our"]
    [rows.append(i[0]) for i in arch_path]
    for ax, col in zip(axs[0], cols):
        ax.set_title(col,fontsize=20)
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation="horizontal", fontsize=20)
    c = -1
    for (arch, model_path) in arch_path:
        c += 1
        # arch = "SCWSRCCANet"
        # arch = "PraNet"
        # arch = "HardnetMSEG"
        model = models.__dict__[arch]()

        # model_path = "pretrained/HarDNet-MSEG-best.pth"
        # model_path = "pretrained/PraNet-19.pth"
        # model_path = "snapshots/SCWSRCCANet_kfold/PraNetDG-fold5-178.pth"
        model.cuda()
        model.eval()
        logger.info(f"Loading from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path)["model_state_dict"])
            # model.load_state_dict(torch.load(model_path))
        except:
            model.load_state_dict(torch.load(model_path))

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

            name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            gt = np.asarray(gt, np.float32)
            # gt /= gt.max() + 1e-8
            res2 = 0
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
            visualize_dir = "output"

            ##### IMAGE
            imgs.append(np.asarray(img[:, :, ::-1]))
            axs[0][cc].imshow(img / (img.max() + 1e-8),cmap = 'gray')
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

            
            axs[1][cc].imshow(gt,cmap = 'gray')
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

        # s = [
        #     list(a)
        #     for a in zip(
        #         [os.path.basename(o) for o in mask_paths],
        #         mean_dices,
        #         mean_ious,
        #         mean_precisions,
        #         mean_recalls,
        #     )
        # ]
        # s.sort(key=lambda x: x[1])
        # for i in s:
        #     logger.info(i)
        # import pandas as pd
        # pd.DataFrame(s).to_csv("Kvasir.csv")

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