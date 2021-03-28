from torch.autograd import Variable
import torch
from utils.utils import clip_gradient, AvgMeter
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import timeit
import torch.nn.functional as F
from datetime import datetime

from utils.metrics import *

class Trainer:
    def __init__(self, net, optimizer, loss, scheduler, save_dir, save_from, logger):
        self.net = net
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_from = save_from
        self.writer = SummaryWriter()
        self.logger = logger

    def val(self, val_loader, epoch):
        len_val = len(val_loader)

        tp_all = 0
        fp_all = 0
        fn_all = 0

        mean_precision = 0
        mean_recall = 0
        mean_iou = 0
        mean_dice = 0

        (
            loss_recordx2,
            loss_recordx3,
            loss_recordx4,
            loss_record2,
            loss_record3,
            loss_record4,
            loss_record5,
        ) = (
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
        )

        for i, pack in enumerate(val_loader, start=1):
            image, gt, gt_resize = pack
            self.net.eval()

            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)

            res2 = 0
            image = image.cuda()
            gt_resize = gt_resize.cuda()


            # print(gt.shape,"sssss")
            # import sys
            # sys.exit()


            res5, res4, res3, res2 = self.net(image)

            # loss5 = self.loss(res5, gt_resize)
            # loss4 = self.loss(res4, gt_resize)
            # loss3 = self.loss(res3, gt_resize)
            loss2 = self.loss(res2, gt_resize)
            # loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4
            # loss = loss2 + loss3 + loss4 + loss5

            loss_record2.update(loss2.data, 1)
            # loss_record3.update(loss3.data, 1)
            # loss_record4.update(loss4.data, 1)
            # loss_record5.update(loss5.data, 1)

            self.writer.add_scalar(
                "Loss2_val", loss_record2.show(), (epoch - 1) * len(val_loader) + i
            )
            # # writer.add_scalar("Loss2", loss_record3.show(), (epoch-1)*len(train_loader) + i)
            # # writer.add_scalar("Loss3", loss_record4.show(), (epoch-1)*len(train_loader) + i)
            # # writer.add_scalar("Loss4", loss_record5.show(), (epoch-1)*len(train_loader) + i)

            if i == len_val - 1:
                self.logger.info(
                    "Val :{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}],\
                    [loss_record2: {:.4f}]".format(
                        datetime.now(),
                        epoch,
                        epoch,
                        self.optimizer.param_groups[0]["lr"],
                        i,
                        loss_record2.show(),
                        # loss_record3.show(),
                        # loss_record4.show(),
                        # loss_record5.show(),
                    )
                )


            res = res2
            res = F.upsample(
                res, size=gt.shape, mode="bilinear", align_corners=False
            )
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            pr = res.round()
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



        mean_precision /= len_val
        mean_recall /= len_val
        mean_iou /= len_val
        mean_dice /= len_val
        self.logger.info(
            "scores ver1: {:.3f} {:.3f} {:.3f} {:.3f}".format(
                mean_iou, mean_precision, mean_recall, mean_dice
            )
        )

        precision_all = tp_all / (tp_all + fp_all + K.epsilon())
        recall_all = tp_all / (tp_all + fn_all + K.epsilon())
        dice_all = 2 * precision_all * recall_all / (precision_all + recall_all)
        iou_all = (
            recall_all
            * precision_all
            / (recall_all + precision_all - recall_all * precision_all)
        )
        self.logger.info(
            "scores ver2: {:.3f} {:.3f} {:.3f} {:.3f}".format(
                iou_all, precision_all, recall_all, dice_all
            )
        )
    def fit(
        self,
        train_loader,
        is_val=False,
        val_loader=None,
        img_size=352,
        start_from=0,
        num_epochs=200,
        batchsize=16,
        clip=0.5,
        fold=4,
        size_rates=[1],
    ):

        val_fold = f"fold{fold}"
        start = timeit.default_timer()
        for epoch in range(start_from, num_epochs):

            self.net.train()
            loss_all, loss_record2, loss_record3, loss_record4, loss_record5 = (
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
            )
            for i, pack in enumerate(train_loader, start=1):
                for rate in size_rates:
                    self.optimizer.zero_grad()

                    # ---- data prepare ----
                    images, gts = pack
                    # images, gts, paths, oriimgs = pack

                    images = Variable(images).cuda()
                    gts = Variable(gts).cuda()

                    trainsize = int(round(img_size * rate / 32) * 32)

                    if rate != 1:
                        images = F.upsample(
                            images,
                            size=(trainsize, trainsize),
                            mode="bilinear",
                            align_corners=True,
                        )
                        gts = F.upsample(
                            gts,
                            size=(trainsize, trainsize),
                            mode="bilinear",
                            align_corners=True,
                        )

                    (
                        lateral_map_5,
                        lateral_map_4,
                        lateral_map_3,
                        lateral_map_2,
                    ) = self.net(images)

                    loss5 = self.loss(lateral_map_5, gts)
                    loss4 = self.loss(lateral_map_4, gts)
                    loss3 = self.loss(lateral_map_3, gts)
                    loss2 = self.loss(lateral_map_2, gts)

                    # loss = loss2 + loss3 + loss4 + loss5

                    loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4

                    loss.backward()
                    clip_gradient(self.optimizer, clip)
                    self.optimizer.step()

                    if rate == 1:
                        loss_record2.update(loss2.data, batchsize)
                        loss_record3.update(loss3.data, batchsize)
                        loss_record4.update(loss4.data, batchsize)
                        loss_record5.update(loss5.data, batchsize)
                        loss_all.update(loss.data, batchsize)

                        self.writer.add_scalar(
                            "Loss2",
                            loss_record2.show(),
                            (epoch - 1) * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss3",
                            loss_record3.show(),
                            (epoch - 1) * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss4",
                            loss_record4.show(),
                            (epoch - 1) * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss5",
                            loss_record5.show(),
                            (epoch - 1) * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss", loss_all.show(), (epoch - 1) * len(train_loader) + i
                        )

                total_step = len(train_loader)
                if i % 25 == 0 or i == total_step:
                    self.logger.info(
                        "{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                        [loss_record2: {:.4f},loss_record3: {:.4f},loss_record4: {:.4f},loss_record5: {:.4f}, loss_all: {:.4f}]".format(
                            datetime.now(),
                            epoch,
                            epoch,
                            self.optimizer.param_groups[0]["lr"],
                            i,
                            total_step,
                            loss_record2.show(),
                            loss_record3.show(),
                            loss_record4.show(),
                            loss_record5.show(),
                            loss_all.show(),
                        )
                    )

            if is_val:
                self.val(val_loader, epoch)

            os.makedirs(self.save_dir, exist_ok=True)
            if epoch > self.save_from or epoch == 23:
                torch.save(
                    {
                        "model_state_dict": self.net.state_dict(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    os.path.join(
                        self.save_dir, "PraNetDG-" + val_fold + "-%d.pth" % epoch
                    ),
                )
                self.logger.info(
                    "[Saving Snapshot:]"
                    + os.path.join(
                        self.save_dir, "PraNetDG-" + val_fold + "-%d.pth" % epoch
                    )
                )

            self.scheduler.step()

        self.writer.flush()
        self.writer.close()
        end = timeit.default_timer()

        self.logger.info("Training cost: " + str(end - start) + "seconds")


class TransUnetTrainer:
    def __init__(self, net, optimizer, loss, scheduler, save_dir, save_from, logger):
        self.net = net
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_from = save_from
        self.writer = SummaryWriter()
        self.logger = logger

    def val(self, val_loader, epoch):
        len_val = len(val_loader)

        for i, pack in enumerate(val_loader, start=1):
            image, gt = pack
            self.net.eval()

            # gt = gt[0][0]
            # gt = np.asarray(gt, np.float32)
            res2 = 0
            image = image.cuda()
            gt = gt.cuda()

            (
                loss_recordx2,
                loss_recordx3,
                loss_recordx4,
                loss_record2,
                loss_record3,
                loss_record4,
                loss_record5,
            ) = (
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
            )

            res5, res4, res3, res2 = self.net(image)

            loss5 = self.loss(res5, gt)
            loss4 = self.loss(res4, gt)
            loss3 = self.loss(res3, gt)
            loss2 = self.loss(res2, gt)
            loss = loss2 + loss3 + loss4 + loss5

            loss_record2.update(loss2.data, 1)
            loss_record3.update(loss3.data, 1)
            loss_record4.update(loss4.data, 1)
            loss_record5.update(loss5.data, 1)

            self.writer.add_scalar(
                "Loss1_val", loss_record2.show(), (epoch - 1) * len(val_loader) + i
            )
            # writer.add_scalar("Loss2", loss_record3.show(), (epoch-1)*len(train_loader) + i)
            # writer.add_scalar("Loss3", loss_record4.show(), (epoch-1)*len(train_loader) + i)
            # writer.add_scalar("Loss4", loss_record5.show(), (epoch-1)*len(train_loader) + i)

            if i == len_val - 1:
                self.logger.info(
                    "val:{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}],\
                    [loss_record2: {:.4f},loss_record3: {:.4f},loss_record4: {:.4f},loss_record5: {:.4f}]".format(
                        datetime.now(),
                        epoch,
                        epoch,
                        self.optimizer.param_groups[0]["lr"],
                        i,
                        loss_record2.show(),
                        loss_record3.show(),
                        loss_record4.show(),
                        loss_record5.show(),
                    )
                )

    def fit(
        self,
        train_loader,
        is_val=False,
        val_loader=None,
        img_size=352,
        start_from=0,
        num_epochs=200,
        batchsize=16,
        clip=0.5,
        fold=4,
    ):

        size_rates = [0.75, 1, 1.25]
        rate = 1

        val_fold = f"fold{fold}"
        start = timeit.default_timer()
        for epoch in range(start_from, num_epochs):

            self.net.train()
            loss_all, loss_record2, loss_record3, loss_record4, loss_record5 = (
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
            )
            for i, pack in enumerate(train_loader, start=1):

                self.optimizer.zero_grad()

                # ---- data prepare ----
                images, gts = pack
                # images, gts, paths, oriimgs = pack

                images = Variable(images).cuda()
                gts = Variable(gts).cuda()

                lateral_map_5 = self.net(images)
                loss5 = self.loss(lateral_map_5, gts)

                loss5.backward()
                clip_gradient(self.optimizer, clip)
                self.optimizer.step()

                if rate == 1:
                    loss_record5.update(loss5.data, batchsize)
                    self.writer.add_scalar(
                        "Loss5",
                        loss_record5.show(),
                        (epoch - 1) * len(train_loader) + i,
                    )

                total_step = len(train_loader)
                if i % 25 == 0 or i == total_step:
                    self.logger.info(
                        "{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                        [loss_record5: {:.4f}]".format(
                            datetime.now(),
                            epoch,
                            epoch,
                            self.optimizer.param_groups[0]["lr"],
                            i,
                            total_step,
                            loss_record5.show(),
                        )
                    )

            if is_val:
                self.val(val_loader, epoch)

            os.makedirs(self.save_dir, exist_ok=True)
            if (epoch + 1) % 3 == 0 and epoch > self.save_from or epoch == 23:
                torch.save(
                    {
                        "model_state_dict": self.net.state_dict(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    os.path.join(
                        self.save_dir, "PraNetDG-" + val_fold + "-%d.pth" % epoch
                    ),
                )
                self.logger.info(
                    "[Saving Snapshot:]"
                    + os.path.join(
                        self.save_dir, "PraNetDG-" + val_fold + "-%d.pth" % epoch
                    )
                )

            self.scheduler.step()

        self.writer.flush()
        self.writer.close()
        end = timeit.default_timer()

        self.logger.info("Training cost: " + str(end - start) + "seconds")


class TrainerGCPAGALD:
    def __init__(self, net, optimizer, loss, scheduler, save_dir, save_from, logger):
        self.net = net
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_from = save_from
        self.writer = SummaryWriter()
        self.logger = logger

    def val(self, val_loader, epoch):
        len_val = len(val_loader)

        for i, pack in enumerate(val_loader, start=1):
            image, gt = pack
            self.net.eval()

            # gt = gt[0][0]
            # gt = np.asarray(gt, np.float32)
            res2 = 0
            image = image.cuda()
            gt = gt.cuda()

            (
                loss_record_head5,
                loss_record2,
                loss_record3,
                loss_record4,
                loss_record5,
                loss_all,
            ) = (
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
            )

            x5_head_out, res5, res4, res3, res2 = self.net(image)

            loss_head_5 = self.loss(x5_head_out, gt)
            loss5 = self.loss(res5, gt)
            loss4 = self.loss(res4, gt)
            loss3 = self.loss(res3, gt)
            loss2 = self.loss(res2, gt)
            loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4 + loss_head_5

            loss_record2.update(loss2.data, 1)

            self.writer.add_scalar(
                "Loss1_val", loss_record2.show(), (epoch - 1) * len(val_loader) + i
            )
            if i == len_val - 1:
                self.logger.info(
                    "val:{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}],\
                    [loss_record2: {:.4f}]".format(
                        datetime.now(),
                        epoch,
                        epoch,
                        self.optimizer.param_groups[0]["lr"],
                        i,
                        loss_record2.show(),
                    )
                )

    def fit(
        self,
        train_loader,
        is_val=False,
        val_loader=None,
        img_size=352,
        start_from=0,
        num_epochs=200,
        batchsize=16,
        clip=0.5,
        fold=4,
    ):

        # size_rates = [1, 1.25]
        size_rates = [0.75, 1, 1.25]

        val_fold = f"fold{fold}"
        start = timeit.default_timer()
        # from network.optim.schedulers import GradualWarmupScheduler
        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, 200, eta_min=0.0001, last_epoch=-1
        # )
        # scheduler = GradualWarmupScheduler(
        #     self.optimizer,
        #     multiplier=8,
        #     total_epoch=8,
        #     after_scheduler=cosine_scheduler,
        # )

        for epoch in range(start_from, num_epochs):
            self.net.train()
            (
                loss_record_head5,
                loss_all,
                loss_record2,
                loss_record3,
                loss_record4,
                loss_record5,
            ) = (
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
                AvgMeter(),
            )
            for i, pack in enumerate(train_loader, start=1):
                for rate in size_rates:
                    self.optimizer.zero_grad()

                    # ---- data prepare ----
                    images, gts = pack
                    # images, gts, paths, oriimgs = pack

                    images = Variable(images).cuda()
                    gts = Variable(gts).cuda()

                    trainsize = int(round(img_size * rate / 32) * 32)

                    if rate != 1:
                        images = F.upsample(
                            images,
                            size=(trainsize, trainsize),
                            mode="bilinear",
                            align_corners=True,
                        )
                        gts = F.upsample(
                            gts,
                            size=(trainsize, trainsize),
                            mode="bilinear",
                            align_corners=True,
                        )

                    (
                        x5_head_out,
                        lateral_map_5,
                        lateral_map_4,
                        lateral_map_3,
                        lateral_map_2,
                    ) = self.net(images)

                    loss_head_5 = self.loss(x5_head_out, gts)
                    loss5 = self.loss(lateral_map_5, gts)
                    loss4 = self.loss(lateral_map_4, gts)
                    loss3 = self.loss(lateral_map_3, gts)
                    loss2 = self.loss(lateral_map_2, gts)

                    # loss = loss2 + loss3 + loss4 + loss5

                    loss = (
                        loss2 * 1
                        + loss3 * 0.8
                        + loss4 * 0.6
                        + loss5 * 0.4
                        + loss_head_5
                    )

                    loss.backward()
                    clip_gradient(self.optimizer, clip)
                    self.optimizer.step()

                    if rate == 1:
                        loss_record2.update(loss2.data, batchsize)
                        loss_record3.update(loss3.data, batchsize)
                        loss_record4.update(loss4.data, batchsize)
                        loss_record5.update(loss5.data, batchsize)
                        loss_record_head5.update(loss_head_5.data, batchsize)
                        loss_all.update(loss.data, batchsize)

                        self.writer.add_scalar(
                            "Loss2",
                            loss_record2.show(),
                            (epoch - 1) * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss3",
                            loss_record3.show(),
                            (epoch - 1) * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss4",
                            loss_record4.show(),
                            (epoch - 1) * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss5",
                            loss_record5.show(),
                            (epoch - 1) * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss_head5",
                            loss_record_head5.show(),
                            (epoch - 1) * len(train_loader) + i,
                        )

                        self.writer.add_scalar(
                            "Loss", loss_all.show(), (epoch - 1) * len(train_loader) + i
                        )

                total_step = len(train_loader)
                if i % 25 == 0 or i == total_step:
                    self.logger.info(
                        "{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                        [loss_record2: {:.4f},loss_record3: {:.4f},loss_record4: {:.4f},loss_record5: {:.4f},loss_record_head5: {:.4f}]".format(
                            datetime.now(),
                            epoch,
                            epoch,
                            self.optimizer.param_groups[0]["lr"],
                            i,
                            total_step,
                            loss_record2.show(),
                            loss_record3.show(),
                            loss_record4.show(),
                            loss_record5.show(),
                            loss_record_head5.show(),
                        )
                    )

            if is_val:
                self.val(val_loader, epoch)

            os.makedirs(self.save_dir, exist_ok=True)
            if (epoch + 1) % 3 == 0 and epoch > self.save_from or epoch == 23:
                torch.save(
                    {
                        "model_state_dict": self.net.state_dict(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    os.path.join(
                        self.save_dir, "PraNetDG-" + val_fold + "-%d.pth" % epoch
                    ),
                )
                self.logger.info(
                    "[Saving Snapshot:]"
                    + os.path.join(
                        self.save_dir, "PraNetDG-" + val_fold + "-%d.pth" % epoch
                    )
                )

            self.scheduler.step()

        self.writer.flush()
        self.writer.close()
        end = timeit.default_timer()

        self.logger.info("Training cost: " + str(end - start) + "seconds")
