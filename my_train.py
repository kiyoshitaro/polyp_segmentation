from argparse import ArgumentParser
from dataloader import  get_loader
from dataloader.augment import Augmenter
from loguru import logger
from glob import glob
import torch
import torch.nn.functional as F


def main():
    # parser = ArgumentParser()
    # parser.add_argument("-c", "--config", action="append", default=[])
    # parser.add_argument("-n", "--name", required=True)
    # args = parser.parse_args()
    
    logger.info("Loading config")
    # config_paths = ["config/defaults.yml"]
    # print_config(config)

    # train_save,lr, arch, start_from, img_size, batchsize, clip,save_from, num_epochs, image_train_paths, gt_train_paths , image_test_paths, gt_test_paths= config(...)

    train_save = ""
    lr = 1e-4
    arch = "PraNet"
    start_from = 0
    img_size = 352
    batchsize = 16
    clip = 0.5
    save_from = 160
    num_epochs = 200


    name = [[1,2,3,4], [0,2,3,4], [0,1,3,4], [0,1,2,4], [0,1,2,3]]

    i = 4
    train1 = 'fold_' + str(name[i][0])
    train2 = 'fold_' + str(name[i][1])
    train3 = 'fold_' + str(name[i][2])
    train4 = 'fold_' + str(name[i][3])
    test_fold = 'fold' + str(i)
    train_img_paths =[]
    train_mask_paths = []
    train_img_path_1 = glob('data/Kvasir_fold_new/' + train1 + "/images/*")
    train_img_paths.extend(train_img_path_1)
    train_img_path_2 = glob('data/Kvasir_fold_new/' + train2 + "/images/*")
    train_img_paths.extend(train_img_path_2)
    train_img_path_3 = glob('data/Kvasir_fold_new/' + train3 + "/images/*")
    train_img_paths.extend(train_img_path_3)
    train_img_path_4 = glob('data/Kvasir_fold_new/' + train4 + "/images/*")
    train_img_paths.extend(train_img_path_4)
    train_mask_path_1 = glob('data/Kvasir_fold_new/' + train1 + "/masks/*")
    train_mask_paths.extend(train_mask_path_1)
    train_mask_path_2 = glob('data/Kvasir_fold_new/' + train2 + "/masks/*")
    train_mask_paths.extend(train_mask_path_2)
    train_mask_path_3 = glob('data/Kvasir_fold_new/' + train3 + "/masks/*")
    train_mask_paths.extend(train_mask_path_3)
    train_mask_path_4 = glob('data/Kvasir_fold_new/' + train4 + "/masks/*")
    train_mask_paths.extend(train_mask_path_4)
    train_img_paths.sort()
    train_mask_paths.sort()


    data_path = 'data/Kvasir_fold_new/' + 'fold_' + str(i)
    X_test = glob('{}/images/*'.format(data_path))
    X_test.sort()
    y_test = glob('{}/masks/*'.format(data_path))
    y_test.sort()


    # DATALOADER 
    train_transform = Augmenter(prob=0.7,
                    blur_prob=0.7,
                    jitter_prob=0.7,
                    rotate_prob=0.7,
                    flip_prob=0.7,
                    )
    train_loader = get_loader(train_img_paths, train_mask_paths, batchsize,img_size, train_transform, shuffle=True, pin_memory=True, drop_last=True)
    total_step = len(train_loader)
    test_transform = Augmenter(prob=0,
                    blur_prob=0,
                    jitter_prob=0,
                    rotate_prob=0,
                    flip_prob=0,
                    )
    test_loader = get_loader(X_test, y_test, 1, img_size, test_transform, shuffle=False, pin_memory=True, drop_last=True)
    test_size = len(test_loader)


    # USE MODEL
    import network.models as models
    model = models.__dict__[arch]
    model = model().cuda()
    params = model.parameters()

    if start_from != 0: 
        restore_from = "./snapshots/PraNetv{}_Res2Net_kfold/PraNetDG-fold{}-{}.pth".format(v,i,start_from)
        lr = model.initialize_weights(restore_from)



    logger.info("#"*20, f"Start Training Fold", "#"*20)

    # USE OPTIMIZER
    optimizer = torch.optim.Adam(params, lr/8)


    # USE SCHEDULE
    from network.optim.schedulers import GradualWarmupScheduler
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)


    # USE LOSS
    from network.optim.losses import StructureLoss
    loss = StructureLoss()



    # TRAINER
    from network.models import Trainer
    save_path = 'snapshots/{}/'.format(train_save)
    trainer = Trainer(model, optimizer, loss, scheduler,save_path, save_from)
    trainer.fit(train_loader =train_loader, is_val = True,test_loader = test_loader, img_size = img_size, start_from = start_from, num_epochs = num_epochs, batchsize = batchsize)
    

if __name__ == "__main__":
    main()