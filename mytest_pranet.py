from argparse import ArgumentParser
from datetime import datetime
from utils.config import load_cfg

from dataloader import get_loader
from dataloader.augment import Augmenter
import tqdm
import torch
from loguru import logger
import os
from glob import glob
from utils.visualize import save_img
from utils.metrics import *
import numpy as np
import torch.nn.functional as F


def main():

    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, default="configs/default_config.yaml"
    )
    args = parser.parse_args()

    logger.info("Loading config")
    config_path = args.config
    config = load_cfg(config_path)

    gts = []
    prs = []

    folds = config["test"]["folds"]
    print(folds)
    dataset = config["dataset"]["test_data_path"][0].split("/")[-1]
    if len(folds.keys()) == 1:
        logger.add(
            f'logs/test_{config["model"]["arch"]}_{str(datetime.now())}_{list(folds.keys())[0]}_{dataset}.log',
            rotation="10 MB",
        )
    else:
        logger.add(
            f'logs/test_{config["model"]["arch"]}_{str(datetime.now())}_kfold.log',
            rotation="10 MB",
        )

    for id in list(folds.keys()):

        # FOR ORIDATASET
        test_img_paths = []
        test_mask_paths = []
        test_data_path = config["dataset"]["test_data_path"]
        for i in test_data_path:
            test_img_paths.extend(glob(os.path.join(i, "images", "*")))
            test_mask_paths.extend(glob(os.path.join(i, "masks", "*")))
        test_img_paths.sort()
        test_mask_paths.sort()

        test_augprams = config["test"]["augment"]
        test_transform = Augmenter(**test_augprams)
        test_loader = get_loader(
            test_img_paths,
            test_mask_paths,
            transform=test_transform,
            **config["test"]["dataloader"],
            type="test",
        )
        test_size = len(test_loader)

        logger.info("Loading model")
        model_prams = config["model"]
        import network.models as models

        arch = model_prams["arch"]


        model = models.__dict__[arch]()  # Pranet

        model_path = "pretrained/PraNet-19.pth"

        device = torch.device("cpu")
        # model.cpu()

        model.cuda()
        model.eval()

        logger.info(f"Loading from {model_path}")
        try:

            model.load_state_dict(torch.load(model_path))
        except RuntimeError:
            model.load_state_dict(torch.load(model_path))

        tp_all = 0
        fp_all = 0
        fn_all = 0

        mean_precision = 0
        mean_recall = 0
        mean_iou = 0
        mean_dice = 0
        mean_F2 = 0
        mean_acc = 0

        mean_precision_np = 0
        mean_recall_np = 0
        mean_iou_np = 0
        mean_dice_np = 0

        test_fold = "fold" + str(config["dataset"]["fold"])
        logger.info(f"Start testing fold{id} epoch {e}")
        if "visualize_dir" not in config["test"]:
            visualize_dir = "results"
        else:
            visualize_dir = config["test"]["visualize_dir"]

        test_fold = "fold" + str(id)
        logger.info(f"Start testing {len(test_loader)} images in {dataset} dataset")

        for i, pack in tqdm.tqdm(enumerate(test_loader, start=1)):
            image, gt, filename, img = pack
            name = os.path.splitext(filename[0])[0]
            ext = os.path.splitext(filename[0])[1]
            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)
            gt /= gt.max() + 1e-8
            res2 = 0
            image = image.cuda()
            # image = image.cpu()

            res5, res4, res3, res2 = model(image)
            # _, _, res5, res4, res3, res2 = model(image)
            # res5_head, res5, res4, res3, res2 = model(image)
            # res2 = model(image)

            res = res2
            res = F.upsample(
                res, size=gt.shape, mode="bilinear", align_corners=False
            )
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            overwrite = config["test"]["vis_overwrite"]
            vis_x = config["test"]["vis_x"]
            if config["test"]["visualize"]:
                save_img(
                    os.path.join(
                        visualize_dir,
                        test_fold,
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
                        test_fold,
                        "soft_" + str(arch),
                        name + "_soft_pr" + str(arch) + ext,
                    ),
                    res * 255,
                    "cv2",
                    overwrite,
                )
                # mask_img = np.asarray(img[0]) + cv2.cvtColor(res.round()*60, cv2.COLOR_GRAY2BGR)
                mask_img = (
                    np.asarray(img[0])
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
                        test_fold,
                        "mask_" + str(arch),
                        name + "mask_pr" + str(arch) + ext,
                    ),
                    mask_img,
                    "cv2",
                    overwrite,
                )

            pr = res.round()
            prs.append(pr)
            gts.append(gt)

            tp = np.sum(gt * pr)
            fp = np.sum(pr) - tp
            fn = np.sum(gt) - tp
            tp_all += tp
            fp_all += fp
            fn_all += fn

            mean_precision += precision_m(gt, pr)
            mean_recall += recall_m(gt, pr)
            mean_iou += jaccard_m(gt, pr)
            mean_dice += dice_m(gt, pr)
            mean_F2 += (5 * precision_m(gt, pr) * recall_m(gt, pr)) / (
                4 * precision_m(gt, pr) + recall_m(gt, pr)
            )
            # mean_acc += (tp+tn)/(tp+tn+fp+fn)

            pr = res
            thresh_precision = 0
            thresh_recall = 0
            thresh_iou = 0
            thresh_dice = 0
            for thresh in np.arange(0, 1, 1 / 256):
                out = pr.copy()
                out[out < thresh] = 0
                out[out >= thresh] = 1
                thresh_precision += precision_m(gt, out)
                thresh_recall += recall_m(gt, out)
                thresh_iou += jaccard_m(gt, out)
                thresh_dice += dice_m(gt, out)

            mean_precision_np += thresh_precision / 256
            mean_recall_np += thresh_recall / 256
            mean_iou_np += thresh_iou / 256
            mean_dice_np += thresh_dice / 256

        mean_precision_np /= len(test_loader)
        mean_recall_np /= len(test_loader)
        mean_iou_np /= len(test_loader)
        mean_dice_np /= len(test_loader)

        logger.info(
            "scores ver0: {:.3f} {:.3f} {:.3f} {:.3f}".format(
                mean_iou_np,
                mean_precision_np,
                mean_recall_np,
                mean_dice_np
                # , mean_F2
            )
        )

        mean_precision /= len(test_loader)
        mean_recall /= len(test_loader)
        mean_iou /= len(test_loader)
        mean_dice /= len(test_loader)
        mean_F2 /= len(test_loader)

        logger.info(
            "scores ver1: {:.3f} {:.3f} {:.3f} {:.3f}".format(
                mean_iou,
                mean_precision,
                mean_recall,
                mean_dice
                # , mean_F2
            )
        )

        # logger.info(
        #     "scores ver1: {:.3f} {:.3f} {:.3f} {:.3f}".format(
        #         mean_iou,
        #         mean_precision,
        #         mean_recall,
        #         mean_dice
        #         # , mean_F2
        #     )
        # )

        precision_all = tp_all / (tp_all + fp_all + 1e-07)
        recall_all = tp_all / (tp_all + fn_all + 1e-07)
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

    from utils.metrics import get_scores_v1, get_scores_v2

    if len(folds.keys()) > 1:
        get_scores_v1(gts, prs, logger)
        get_scores_v2(gts, prs, logger)

    return gts, prs


if __name__ == "__main__":
    main()
