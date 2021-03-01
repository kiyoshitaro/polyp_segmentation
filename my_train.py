from argparse import ArgumentParser
from dataloader import  get_loader
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
    parser.add_argument("-c", "--config", required=True, default = "configs/default_config.yaml")
    args = parser.parse_args()


    logger.info("Loading config")
    config_path = args.config
    config = load_cfg(config_path)

    logger.add(f'logs/{str(datetime.now())}_train_log_file.log', rotation="10 MB")
    logger.info(f'Load config from {config_path}')
    logger.info(f'{config}')




    # GET_DATA_PATH
    logger.info("Getting datapath")
    train_img_paths =[]
    train_mask_paths = []
    train_data_path = config["dataset"]["train_data_path"]
    for i in train_data_path:
        train_img_paths.extend(glob(os.path.join(i,"images","*")))
        train_mask_paths.extend(glob(os.path.join(i,"masks","*")))
    train_img_paths.sort()
    train_mask_paths.sort()


    test_img_paths =[]
    test_mask_paths = []
    test_data_path = config["dataset"]["test_data_path"]
    for i in test_data_path:
        test_img_paths.extend(glob(os.path.join(i,"images","*")))
        test_mask_paths.extend(glob(os.path.join(i,"masks","*")))
    test_img_paths.sort()
    test_mask_paths.sort()

    # DATALOADER 
    logger.info("Loading data")
    train_augprams = config["train"]["augment"]
    train_transform = Augmenter(**train_augprams)
    train_loader = get_loader(train_img_paths, train_mask_paths, transform = train_transform, **config["train"]["dataloader"])
    total_step = len(train_loader)

    test_augprams = config["test"]["augment"]
    test_transform = Augmenter(**test_augprams)
    test_loader = get_loader(test_img_paths, test_mask_paths, transform = test_transform, **config["test"]["dataloader"])
    test_size = len(test_loader)


    # USE MODEL
    logger.info("Loading model")
    model_prams = config["model"]
    import network.models as models
    model = models.__dict__[model_prams["arch"]]()
    model = model.cuda()
    params = model.parameters()
    save_dir = os.path.join(model_prams["save_dir"],model_prams["arch"])

    if model_prams["start_from"] != 0: 
        restore_from = os.path.join(save_dir,f'PraNetDG-fold{config["dataset"]["fold"]}-{model_prams["start_from"]}.pth')
        lr = model.initialize_weights(restore_from)




    # USE OPTIMIZER
    opt_params = config["optimizer"]
    import network.optim.optimizers as optims
    lr = opt_params["lr"]
    optimizer = optims.__dict__[opt_params["name"].lower()](params, lr/8)

    # USE SCHEDULE
    import network.optim.schedulers as schedulers
    scheduler = schedulers.__dict__[opt_params["scheduler"]](optimizer, lr, model_prams["num_epochs"], opt_params["num_warmup_epoch"])

    # USE LOSS
    import network.optim.losses as losses
    loss = losses.__dict__[opt_params["loss"]]()

    # TRAINER
    fold = config["dataset"]["fold"]
    logger.info("#"*20, f"Start Training Fold {fold}", "#"*20)
    from network.models import Trainer
    trainer = Trainer(model, optimizer, loss, scheduler, save_dir, model_prams["save_from"], logger)
    trainer.fit(train_loader =train_loader, is_val = config["train"]["is_val"],test_loader = test_loader, img_size = config["train"]["dataloader"]["img_size"], start_from = model_prams["start_from"], num_epochs = model_prams["num_epochs"], batchsize = config["train"]["dataloader"]["batchsize"], fold =fold )
    

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 python my_train.py -c configs/default_config.yaml