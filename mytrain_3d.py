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
        f'logs/train_{config["model"]["arch"]}_{str(datetime.now())}_{config["dataset"]["fold"]}.log',
        rotation="10 MB",
    )

    logger.info(f"Load config from {config_path}")
    logger.info(f"{config}")

    # GET_DATA_PATH
    logger.info("Getting datapath")

    # For 3d .nii
    train_img_paths = []
    train_mask_paths = []
    train_data_path = config["dataset"]["train_data_path"]
    for i in train_data_path:
        train_img_paths.extend(glob(os.path.join(i, "*")))
        train_mask_paths.extend(glob(os.path.join(i, "*")))
    train_img_paths.sort()
    train_mask_paths.sort()
    logger.info(f"There are {len(train_img_paths)} images to train")

    # For 3d .nii
    val_img_paths = []
    val_mask_paths = []
    val_data_path = config["dataset"]["val_data_path"]
    for i in val_data_path:
        val_img_paths.extend(glob(os.path.join(i, "*")))
        val_mask_paths.extend(glob(os.path.join(i, "*")))
    val_img_paths.sort()
    val_mask_paths.sort()
    logger.info(f'There are {len(val_mask_paths)} images to val')



    # DATALOADER
    logger.info("Loading data")
    train_augprams = config["train"]["augment"]
    # 3d .nii no need augment
    train_transform = train_augprams["train_transforms"]

    train_loader = get_loader(
        train_img_paths,
        train_mask_paths,
        transform=train_transform,
        **config["train"]["dataloader"],
        type="train",
    )
    total_step = len(train_loader)

    logger.info(f"{total_step} batches to train")

    val_augprams = config["test"]["augment"]

    # 3d .nii no need augment
    # val_transform = Augmenter(**val_augprams)
    val_transform = None

    val_loader = get_loader(
        val_img_paths,
        val_mask_paths,
        transform=val_transform,
        **config["test"]["dataloader"],
        type="val",
    )
    val_size = len(val_loader)

    # USE MODEL
    logger.info("Loading model")
    model_prams = config["model"]
    if "save_dir" not in model_prams:
        save_dir = os.path.join("snapshots", model_prams["arch"] + "_kfold")
    else:
        save_dir = config["model"]["save_dir"]

    import network.models as models

    model = models.__dict__[model_prams["arch"]]()  # Pranet
    model = model.cuda()
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
    from network.models import Trainer3D

    trainer = Trainer3D(
        model, optimizer, loss, scheduler, save_dir, model_prams["save_from"], logger
    )

    trainer.fit(
        train_loader=train_loader,
        is_val=config["train"]["is_val"],
        val_loader=val_loader,
        img_size=config["train"]["dataloader"]["img_size"],
        start_from=model_prams["start_from"],
        num_epochs=model_prams["num_epochs"],
        batchsize=config["train"]["dataloader"]["batchsize"],
        fold=fold,
        size_rates=config["train"]["size_rates"],
    )


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0 python my_train.py -c configs/gcpa_cc_instrument.yaml
# CUDA_VISIBLE_DEVICES=0 python my_train.py -c configs/gcpa_cc_isic.yaml
# CUDA_VISIBLE_DEVICES=1 python my_test.py -c configs/scws_cc_config.yaml
# CUDA_VISIBLE_DEVICES=0 python mytest_3d.py -c configs/gcpa_gald_brats.yaml
# CUDA_VISIBLE_DEVICES=1 python my_train.py -c configs/gcpa_cc_usnerve.yaml
# CUDA_VISIBLE_DEVICES=0 python mytrain_3d.py -c configs/gcpa_gald_brats.yaml
