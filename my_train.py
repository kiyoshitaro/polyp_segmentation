from argparse import ArgumentParser

from dataloader import  get_loader


from dataloader.augment import Augmenter


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", action="append", default=[])
    parser.add_argument("-n", "--name", required=True)
    args = parser.parse_args()
    
    logger.info("Loading config")
    # config_paths = ["config/defaults.yml"]
    # print_config(config)

    train_save,lr, arch, start_from, img_size, batchsize, image_train_paths, gt_train_paths , image_test_paths, gt_test_paths= config(...)


    # DATALOADER 
    train_transform = Augmenter(prob=0.7,
                    blur_prob=0.7,
                    jitter_prob=0.7,
                    rotate_prob=0.7,
                    flip_prob=0.7,
                    )
    train_loader = get_loader(image_train_paths, gt_train_paths, batchsize,img_size, train_transform, shuffle=True, pin_memory=True, drop_last=True)
    total_step = len(train_loader)
    test_transform = Augmenter(prob=0,
                    blur_prob=0,
                    jitter_prob=0,
                    rotate_prob=0,
                    flip_prob=0,
                    )
    test_loader = get_loader(image_test_paths, gt_test_paths, 1, img_size, test_transform, shuffle=False, pin_memory=True, drop_last=True)
    test_size = len(test_loader)


    # USE MODEL
    import network.models as models
    model = models.__dict__[arch]
    model = model().cuda()
    if start_from != 0: 
        restore_from = "./snapshots/PraNetv{}_Res2Net_kfold/PraNetDG-fold{}-{}.pth".format(v,i,start_from)
        lr = model.initialize_weights(restore_from)



    logger.info("#"*20, f"Start Training Fold", "#"*20)

    # USE OPTIMIZER
    optimizer = torch.optim.Adam(params, lr/8)


    # USE SCHEDULE
    from optim.schedulers import CosineAnnealingLR
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)


    # USE LOSS
    from optim.losses import StructureLoss
    loss = StructureLoss()



    # TRAINER
    from network.models import Trainer
    save_path = 'snapshots/{}/'.format(train_save)
    trainer = Trainer(model, optimizer, loss, scheduler,train_save, save_from)
    trainer.fit(train_loader, is_val = True, img_size = img_size, start_from = start_from, num_epochs = num_epochs, batchsize = batchsize)
    

if __name__ == "__main__":
    main()