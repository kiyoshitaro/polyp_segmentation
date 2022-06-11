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
    dev = config["test"]["dev"]

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
            x = glob(os.path.join(i,  "*"))
            # x = [os.path.join(i,"images",os.path.basename(i)+".png") for i in x]
            test_img_paths.extend([os.path.join(i,"images",os.path.basename(i)+".png") for i in x])
            test_mask_paths.extend([os.path.join(i,"images",os.path.basename(i)+".png") for i in x])

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

        epochs = folds[id]
        if type(epochs) != list:
            epochs = [3 * (epochs // 3) + 2]
        elif len(epochs) == 2:
            epochs = [i for i in range(epochs[0], epochs[1])]
            # epochs = [3 * i + 2 for i in range(epochs[0] // 3, (epochs[1] + 1) // 3)]
        elif len(epochs) == 1:
            epochs = [3 * (epochs[0] // 3) + 2]
        else:
            logger.debug("Model path must have 0 or 1 num")
            break
        for e in epochs:
            # MODEL

            logger.info("Loading model")
            model_prams = config["model"]
            import network.models as models

            arch = model_prams["arch"]

            model = models.__dict__[arch]()  # Pranet
            if "save_dir" not in model_prams:
                save_dir = os.path.join("snapshots", model_prams["arch"] + "_kfold")
            else:
                save_dir = config["model"]["save_dir"]

            model_path = os.path.join(
                save_dir,
                f"PraNetDG-fold{id}-{e}.pth",
            )

            device = torch.device(dev)
            if dev == "cpu":
                model.cpu()
            else:
                model.cuda()
            model.eval()

            logger.info(f"Loading from {model_path}")
            try:

                # model.load_state_dict(torch.load(model_path))
                # model.load_state_dict(torch.load(model_path)["model_state_dict"])
                model.load_state_dict(
                    torch.load(model_path, map_location=device)["model_state_dict"]
                )
            except RuntimeError:
                # model.load_state_dict(torch.load(model_path))
                model.load_state_dict(torch.load(model_path, map_location=device))


            test_fold = "fold" + str(config["dataset"]["fold"])
            logger.info(f"Start testing fold{id} epoch {e}")
            if "visualize_dir" not in config["test"]:
                visualize_dir = "results"
            else:
                visualize_dir = config["test"]["visualize_dir"]

            test_fold = "fold" + str(id)
            logger.info(f"Start testing {len(test_loader)} images in {dataset} dataset")

            import pandas as pd
            from utils.utils import rle_encoding, prob_to_rles

            submission_df = pd.DataFrame(columns=["ImageId", "EncodedPixels"])

            preds_upsampled = []
            new_test_ids = []
            rles = []
            test_files = []

            for i, pack in tqdm.tqdm(enumerate(test_loader, start=0)):
                image, gt, filename, img = pack
                name = os.path.splitext(filename[0])[0]
                ext = os.path.splitext(filename[0])[1]
                gt = gt[0][0]
                gt = np.asarray(gt, np.float32)
                res2 = 0
                if dev == "cpu":
                    image = image.cpu()
                else:
                    image = image.cuda()
                # image = image.cpu()

                res5, res4, res3, res2 = model(image)
                res = res2
                res = F.upsample(
                    res, size=gt.shape, mode="bilinear", align_corners=False
                )
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                pr = res.round()

                visualize_dir = "results/SCWSRCCANet_Chase_1"
                save_img(
                    os.path.join(
                        visualize_dir,
                        "PR_" + str(arch),
                        "Soft",
                        name + ext,
                    ),
                    1*(res > 0.5)*res*255,
                    # res * 255,
                    "cv2",
                    True,
                )

                preds_upsampled.append(res)
                test_files.append(name)

                # encoding = rle_encoding(res)
                # print(encoding)
                # pixels = " ".join(map(str, encoding))
                # submission_df.loc[i] = [name, pixels]
            rles = []
            for n, id_ in enumerate(test_files):
                rle = list(prob_to_rles(preds_upsampled[n]))
                rles.extend(rle)
                new_test_ids.extend([id_] * len(rle))
            sub = pd.DataFrame()
            sub['ImageId'] = new_test_ids
            sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
            sub.to_csv('sub-dsbowl2018.csv', index=False)

    return gts, prs


if __name__ == "__main__":
    main()
