from argparse import ArgumentParser
from dataloader import get_loader
from dataloader.augment import Augmenter
from loguru import logger
from glob import glob
import torch
import torch.nn.functional as F
from utils.config import load_cfg
from datetime import datetime
import os


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, default="configs/default_config.yaml"
    )
    args = parser.parse_args()

    logger.info("Loading config")
    config_path = args.config
    config = load_cfg(config_path)
    logger.add(
        f'logs/train_{str(datetime.now())}_{config["model"]["arch"]}_{config["dataset"]["fold"]}.log',
        rotation="10 MB",
    )

    logger.info(f"Load config from {config_path}")
    logger.info(f"{config}")

    # GET_DATA_PATH
    logger.info("Getting datapath")

    train_img_paths = []
    train_mask_paths = []
    train_data_path = config["dataset"]["train_data_path"]
    for i in train_data_path:
        train_img_paths.extend(glob(os.path.join(i, "images", "*")))
        train_mask_paths.extend(glob(os.path.join(i, "masks", "*")))
    train_img_paths.sort()
    train_mask_paths.sort()

    test_img_paths = []
    test_mask_paths = []
    test_data_path = config["dataset"]["test_data_path"]
    for i in test_data_path:
        test_img_paths.extend(glob(os.path.join(i, "images", "*")))
        test_mask_paths.extend(glob(os.path.join(i, "masks", "*")))
    test_img_paths.sort()
    test_mask_paths.sort()

    # DATALOADER
    logger.info("Loading data")
    train_augprams = config["train"]["augment"]
    train_transform = Augmenter(**train_augprams)
    train_loader = get_loader(
        train_img_paths,
        train_mask_paths,
        transform=train_transform,
        **config["train"]["dataloader"],
        is_train=True,
    )
    total_step = len(train_loader)

    test_augprams = config["test"]["augment"]
    test_transform = Augmenter(**test_augprams)
    test_loader = get_loader(
        test_img_paths,
        test_mask_paths,
        transform=test_transform,
        **config["test"]["dataloader"],
        is_train=True,
    )
    test_size = len(test_loader)

    # USE MODEL
    logger.info("Loading model")
    model_prams = config["model"]
    if "save_dir" not in model_prams:
        save_dir = os.path.join("snapshots", model_prams["arch"] + "_kfold")
    else:
        save_dir = config["model"]["save_dir"]

    # n_skip = 3
    # vit_name = "R50-ViT-B_16"
    # vit_patches_size = 16
    # img_size = config["dataset"]["img_size"]
    # import torch.backends.cudnn as cudnn
    # from network.models.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
    # import numpy as np

    # config_vit = CONFIGS_ViT_seg[vit_name]
    # config_vit.n_classes = 1
    # config_vit.n_skip = n_skip
    # if vit_name.find("R50") != -1:
    #     config_vit.patches.grid = (
    #         int(img_size / vit_patches_size),
    #         int(img_size / vit_patches_size),
    #     )

    import network.models as models

    model = models.__dict__[model_prams["arch"]]()  # Pranet
    # model = models.__dict__[model_prams["arch"]](
    #     config_vit, img_size=img_size, num_classes=config_vit.n_classes
    # )  # TransUnet
    model = model.cuda()

    # LOAD PRETRAIN

    # TransUnet
    # model.load_from(weights=np.load(config_vit.pretrained_path))

    # Pranet
    # if model_prams["start_from"] != 0:
    #     restore_from = os.path.join(save_dir,f'PraNetDG-fold{config["dataset"]["fold"]}-{model_prams["start_from"]}.pth')
    #     lr = model.initialize_weights(restore_from)

    params = model.parameters()

    # USE OPTIMIZER
    opt_params = config["optimizer"]
    import network.optim.optimizers as optims

    lr = opt_params["lr"]
    optimizer = optims.__dict__[opt_params["name"].lower()](params, lr / 8)

    # USE SCHEDULE
    import network.optim.schedulers as schedulers

    scheduler = schedulers.__dict__[opt_params["scheduler"]](
        optimizer, model_prams["num_epochs"], opt_params["num_warmup_epoch"]
    )
    # scheduler = None
    # USE LOSS
    import network.optim.losses as losses

    loss = losses.__dict__[opt_params["loss"]]()

    # TRAINER
    fold = config["dataset"]["fold"]
    logger.info("#" * 20 + f"Start Training Fold {fold}" + "#" * 20)
    from network.models import Trainer, TransUnetTrainer, TrainerGCPAGALD

    trainer = Trainer(
        model, optimizer, loss, scheduler, save_dir, model_prams["save_from"], logger
    )

    trainer.fit(
        train_loader=train_loader,
        is_val=config["train"]["is_val"],
        test_loader=test_loader,
        img_size=config["train"]["dataloader"]["img_size"],
        start_from=model_prams["start_from"],
        num_epochs=model_prams["num_epochs"],
        batchsize=config["train"]["dataloader"]["batchsize"],
        fold=fold,
    )


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 python my_train.py -c configs/gcpa_gald_net_config.yaml
# CUDA_VISIBLE_DEVICES=1 python my_test.py -c configs/gcpa_gald_net_config.yaml
