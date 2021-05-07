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
from utils.utils import AvgMeter
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

        test_img_paths = []
        test_mask_paths = []
        test_data_path = config["dataset"]["test_data_path"]
        for i in test_data_path:
            test_img_paths.extend(glob(os.path.join(i, "*")))
            test_mask_paths.extend(glob(os.path.join(i, "*")))
        test_img_paths.sort()
        test_mask_paths.sort()


        test_transform = None

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

            device = torch.device("cpu")
            # model.cpu()

            model.cuda()
            model.eval()

            logger.info(f"Loading from {model_path}")
            try:
                model.load_state_dict(torch.load(model_path)["model_state_dict"])
            except RuntimeError:
                model.load_state_dict(torch.load(model_path))

            test_fold = "fold" + str(config["dataset"]["fold"])
            logger.info(f"Start testing fold{id} epoch {e}")
            if "visualize_dir" not in config["test"]:
                visualize_dir = "results"
            else:
                visualize_dir = config["test"]["visualize_dir"]

            test_fold = "fold" + str(id)
            logger.info(f"Start testing {len(test_loader)} images in {dataset} dataset")
            vals = AvgMeter()
            H, W, T = 240, 240, 155


            for i, pack in tqdm.tqdm(enumerate(test_loader, start=1)):
                image, gt, filename, img = pack
                name = os.path.splitext(filename[0])[0]
                ext = os.path.splitext(filename[0])[1]
                # print(gt.shape,image.shape,"ppp")
                # import sys
                # sys.exit()
                gt = gt[0]
                gt = np.asarray(gt, np.float32)
                res2 = 0
                image = image.cuda()

                res5, res4, res3, res2 = model(image)

                # res = res2
                # res = F.upsample(
                #     res, size=gt.shape, mode="bilinear", align_corners=False
                # )
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                output = res2[0, :, :H, :W, :T].cpu().detach().numpy()
                output = output.argmax(0) # (num_classes,height,width,depth) num_classes is now one-hot 

                target_cpu = gt[:H, :W, :T].numpy() 
                scores = softmax_output_dice(output, target_cpu)
                vals.update(np.array(scores))
                # msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, scores)])

                seg_img = np.zeros(shape=(H,W,T),dtype=np.uint8)

                # same as res.round()
                seg_img[np.where(output==1)] = 1
                seg_img[np.where(output==2)] = 2
                seg_img[np.where(output==3)] = 4
                # if verbose:
                logger.info(f'1:{np.sum(seg_img==1)} | 2: {np.sum(seg_img==2)} | 4: {np.sum(seg_img==4)}')
                logger.info(f'WT: {np.sum((seg_img==1)|(seg_img==2)|(seg_img==4))} | TC: {np.sum((seg_img==1)|(seg_img==4))} | ET: {np.sum(seg_img==4)}')


                overwrite = config["test"]["vis_overwrite"]
                vis_x = config["test"]["vis_x"]
                if config["test"]["visualize"]:
                    oname = os.path.join(
                                visualize_dir,
                                'submission', name[:-8] + '_pred.nii.gz')
                    save_img(
                        oname,
                        seg_img,
                        "nib",
                        overwrite,
                    )
            logger.info(vals.avg)




if __name__ == "__main__":
    main()
