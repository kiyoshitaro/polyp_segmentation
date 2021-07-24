from torch.autograd import Variable
import torch
from utils.utils import clip_gradient, AvgMeter
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import timeit
import torch.nn.functional as F
from datetime import datetime
import torchvision
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

        mean_F2 = 0
        mean_acc = 0

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
        images = []
        for i, pack in enumerate(val_loader, start=1):
            image, gt, gt_resize = pack
            self.net.eval()

            gt_ = gt.cuda()
            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)

            res2 = 0
            image_ = image
            image = image.cuda()
            gt_resize = gt_resize.cuda()

            res5, res4, res3, res2 = self.net(image)

            # loss2 = self.loss(res2, gt_resize)
            # loss_record2.update(loss2.data, 1)
            # self.writer.add_scalar(
            #     "Loss2_val", loss_record2.show(), epoch * len(val_loader) + i
            # )

            res = res2
            res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)

            loss2 = self.loss(res, gt_)
            loss_record2.update(loss2.data, 1)
            self.writer.add_scalar(
                "Loss2_val", loss_record2.show(), epoch * len(val_loader) + i
            )
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
                    )
                )

            # if i == len_val - 1:
            #     self.logger.info(
            #         "Val :{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}],\
            #         [loss_record2: {:.4f}]".format(
            #             datetime.now(),
            #             epoch,
            #             self.optimizer.param_groups[0]["lr"],
            #             i,
            #             loss_record2.show(),
            #         )
            #     )

            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            if i <= 0:
                res2 = res2.data.cpu().numpy().round()[0][0]
                gt_resize = gt_resize.data.cpu().numpy()[0][0]
                mask_img = (
                    np.asarray(image_.data.cpu().numpy()[0])
                    + 180
                    * np.array(
                        (
                            np.zeros_like(res2),
                            res2,
                            np.zeros_like(res2),
                        )
                    ).transpose((0, 1, 2))
                    + 180
                    * np.array(
                        (gt_resize, np.zeros_like(gt_resize), np.zeros_like(gt_resize))
                    ).transpose((0, 1, 2))
                )
                images.append(mask_img)

            pr = res.round()
            # gt = gt.round()
            tp = np.sum(gt * pr)
            fp = np.sum(pr) - tp
            fn = np.sum(gt) - tp
            tp_all += tp
            fp_all += fp
            fn_all += fn
            # mean_acc +=
            mean_precision += precision_m(gt, pr)
            mean_recall += recall_m(gt, pr)
            mean_iou += jaccard_m(gt, pr)
            mean_dice += dice_m(gt, pr)
            mean_F2 += (5 * precision_m(gt, pr) * recall_m(gt, pr)) / (
                4 * precision_m(gt, pr) + recall_m(gt, pr)
            )
            # mean_acc +=

        mean_precision /= len_val
        mean_recall /= len_val
        mean_iou /= len_val
        mean_dice /= len_val
        mean_F2 /= len_val
        print(len_val)
        self.logger.info(
            "scores ver1: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                mean_iou, mean_precision, mean_recall, mean_dice, mean_F2
            )
        )

        self.writer.add_scalar("mean_dice", mean_dice, epoch)

        self.writer.add_scalar("mean_iou", mean_iou, epoch)
        precision_all = tp_all / (tp_all + fp_all + 1e-07)
        recall_all = tp_all / (tp_all + fn_all + 1e-07)
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
        self.writer.add_scalar("dice_all", dice_all, epoch)
        self.writer.add_scalar("iou_all", iou_all, epoch)

        # grid = torchvision.utils.make_grid(torch.Tensor(images))
        # self.writer.add_image("images", grid)

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
                            epoch * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss3",
                            loss_record3.show(),
                            epoch * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss4",
                            loss_record4.show(),
                            epoch * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss5",
                            loss_record5.show(),
                            epoch * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss", loss_all.show(), epoch * len(train_loader) + i
                        )

                total_step = len(train_loader)
                if i % 25 == 0 or i == total_step:
                    self.logger.info(
                        "{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                        [loss_record2: {:.4f},loss_record3: {:.4f},loss_record4: {:.4f},loss_record5: {:.4f}, loss_all: {:.4f}]".format(
                            datetime.now(),
                            epoch,
                            num_epochs,
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
            if epoch > self.save_from and (epoch + 1) % 5 == 0 or epoch == 2:
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
            if self.scheduler:
                self.scheduler.step(epoch)

        self.writer.flush()
        self.writer.close()
        end = timeit.default_timer()

        self.logger.info("Training cost: " + str(end - start) + "seconds")


class TrainerDistillation:
    def __init__(self, net,net1, optimizer, loss, scheduler, save_dir, save_from, logger):
        self.net = net
        self.net1 = net1
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

        mean_F2 = 0
        mean_acc = 0

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
        images = []
        for i, pack in enumerate(val_loader, start=1):
            image, gt, gt_resize = pack
            self.net.eval()

            gt_ = gt.cuda()
            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)

            res2 = 0
            image_ = image
            image = image.cuda()
            gt_resize = gt_resize.cuda()

            res5, res4, res3, res2 = self.net(image)

            res = res2
            res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)

            # loss2 = self.loss(res, gt_, )
            # loss_record2.update(loss2.data, 1)
            # self.writer.add_scalar(
            #     "Loss2_val", loss_record2.show(), epoch * len(val_loader) + i
            # )
            if i == len_val - 1:
                self.logger.info(
                    "Val :{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}],\
                    [loss_record2: {:.4f}]".format(
                        datetime.now(),
                        epoch,
                        epoch,
                        self.optimizer.param_groups[0]["lr"],
                        i,
                        0,
                    )
                )

            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            if i <= 0:
                res2 = res2.data.cpu().numpy().round()[0][0]
                gt_resize = gt_resize.data.cpu().numpy()[0][0]
                mask_img = (
                    np.asarray(image_.data.cpu().numpy()[0])
                    + 180
                    * np.array(
                        (
                            np.zeros_like(res2),
                            res2,
                            np.zeros_like(res2),
                        )
                    ).transpose((0, 1, 2))
                    + 180
                    * np.array(
                        (gt_resize, np.zeros_like(gt_resize), np.zeros_like(gt_resize))
                    ).transpose((0, 1, 2))
                )
                images.append(mask_img)

            pr = res.round()
            # gt = gt.round()
            tp = np.sum(gt * pr)
            fp = np.sum(pr) - tp
            fn = np.sum(gt) - tp
            tp_all += tp
            fp_all += fp
            fn_all += fn
            # mean_acc +=
            mean_precision += precision_m(gt, pr)
            mean_recall += recall_m(gt, pr)
            mean_iou += jaccard_m(gt, pr)
            mean_dice += dice_m(gt, pr)
            mean_F2 += (5 * precision_m(gt, pr) * recall_m(gt, pr)) / (
                4 * precision_m(gt, pr) + recall_m(gt, pr)
            )
            # mean_acc +=

        mean_precision /= len_val
        mean_recall /= len_val
        mean_iou /= len_val
        mean_dice /= len_val
        mean_F2 /= len_val
        print(len_val)
        self.logger.info(
            "scores ver1: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                mean_iou, mean_precision, mean_recall, mean_dice, mean_F2
            )
        )

        self.writer.add_scalar("mean_dice", mean_dice, epoch)

        self.writer.add_scalar("mean_iou", mean_iou, epoch)
        precision_all = tp_all / (tp_all + fp_all + 1e-07)
        recall_all = tp_all / (tp_all + fn_all + 1e-07)
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
        self.writer.add_scalar("dice_all", dice_all, epoch)
        self.writer.add_scalar("iou_all", iou_all, epoch)

        # grid = torchvision.utils.make_grid(torch.Tensor(images))
        # self.writer.add_image("images", grid)

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
                    images, gts  = pack
                    # images, gts, softlabel = pack
                    # images, gts, paths, oriimgs = pack

                    images = Variable(images).cuda()
                    gts = Variable(gts).cuda()
                    # softlabel = Variable(softlabel).cuda()

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
                        # softlabel = F.upsample(
                        #     softlabel,
                        #     size=(trainsize, trainsize),
                        #     mode="bilinear",
                        #     align_corners=True,
                        # )

                    (
                        lateral_map_5,
                        lateral_map_4,
                        lateral_map_3,
                        lateral_map_2,
                    ) = self.net(images)
                    self.net1.eval()
                    (
                        soft_lateral_map_5,
                        soft_lateral_map_4,
                        soft_lateral_map_3,
                        soft_lateral_map_2,
                    ) = self.net1(images)
                    

                    # loss5 = self.loss(lateral_map_5, gts, softlabel)
                    loss5 = self.loss(lateral_map_5, gts, soft_lateral_map_5)
                    loss4 = self.loss(lateral_map_4, gts, soft_lateral_map_4)
                    loss3 = self.loss(lateral_map_3, gts, soft_lateral_map_3)
                    loss2 = self.loss(lateral_map_2, gts, soft_lateral_map_2)

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
                            epoch * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss3",
                            loss_record3.show(),
                            epoch * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss4",
                            loss_record4.show(),
                            epoch * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss5",
                            loss_record5.show(),
                            epoch * len(train_loader) + i,
                        )
                        self.writer.add_scalar(
                            "Loss", loss_all.show(), epoch * len(train_loader) + i
                        )

                total_step = len(train_loader)
                if i % 25 == 0 or i == total_step:
                    self.logger.info(
                        "{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                        [loss_record2: {:.4f},loss_record3: {:.4f},loss_record4: {:.4f},loss_record5: {:.4f}, loss_all: {:.4f}]".format(
                            datetime.now(),
                            epoch,
                            num_epochs,
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
            if epoch > self.save_from and (epoch + 1) % 5 == 0 or epoch == 50:
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
            if self.scheduler:
                self.scheduler.step(epoch)

        self.writer.flush()
        self.writer.close()
        end = timeit.default_timer()

        self.logger.info("Training cost: " + str(end - start) + "seconds")


class TrainerOne:
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

        mean_F2 = 0
        mean_acc = 0

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
        images = []
        for i, pack in enumerate(val_loader, start=1):
            image, gt, gt_resize = pack
            self.net.eval()

            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)

            res2 = 0
            image_ = image
            image = image.cuda()
            gt_resize = gt_resize.cuda()

            res2 = self.net(image)
            loss2 = self.loss(res2, gt_resize)

            loss_record2.update(loss2.data, 1)

            self.writer.add_scalar(
                "Loss2_val", loss_record2.show(), epoch * len(val_loader) + i
            )

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
                    )
                )

            res = F.upsample(res2, size=gt.shape, mode="bilinear", align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            res2 = res2.data.cpu().numpy().round()[0][0]
            gt_resize = gt_resize.data.cpu().numpy()[0][0]

            mask_img = (
                np.asarray(image_.data.cpu().numpy()[0])
                + 180
                * np.array(
                    (
                        np.zeros_like(res2),
                        res2,
                        np.zeros_like(res2),
                    )
                ).transpose((0, 1, 2))
                + 180
                * np.array(
                    (gt_resize, np.zeros_like(gt_resize), np.zeros_like(gt_resize))
                ).transpose((0, 1, 2))
            )

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
            mean_F2 += (5 * precision_m(gt, pr) * recall_m(gt, pr)) / (
                4 * precision_m(gt, pr) + recall_m(gt, pr)
            )

        mean_precision /= len_val
        mean_recall /= len_val
        mean_iou /= len_val
        mean_dice /= len_val
        mean_F2 /= len_val
        self.logger.info(
            "scores ver1: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                mean_iou, mean_precision, mean_recall, mean_dice, mean_F2
            )
        )

        self.writer.add_scalar("mean_dice", mean_dice, epoch)

        self.writer.add_scalar("mean_iou", mean_iou, epoch)
        precision_all = tp_all / (tp_all + fp_all + 1e-07)
        recall_all = tp_all / (tp_all + fn_all + 1e-07)
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
        self.writer.add_scalar("dice_all", dice_all, epoch)
        self.writer.add_scalar("iou_all", iou_all, epoch)

        grid = torchvision.utils.make_grid(torch.Tensor(mask_img))
        self.writer.add_image("images", grid)

        # grid = torchvision.utils.make_grid(images)
        # self.writer.add_image("images", grid)

        # self.writer.add_graph(self.net, images)

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
            (loss_record2,) = (AvgMeter(),)
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

                    lateral_map_2 = self.net(images)

                    loss2 = self.loss(lateral_map_2, gts)

                    loss = loss2

                    loss.backward()
                    clip_gradient(self.optimizer, clip)
                    self.optimizer.step()

                    if rate == 1:
                        loss_record2.update(loss2.data, batchsize)

                        self.writer.add_scalar(
                            "Loss2",
                            loss_record2.show(),
                            epoch * len(train_loader) + i,
                        )

                total_step = len(train_loader)
                if i % 25 == 0 or i == total_step:
                    self.logger.info(
                        "{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                        [loss_record2: {:.4f},]".format(
                            datetime.now(),
                            epoch,
                            num_epochs,
                            self.optimizer.param_groups[0]["lr"],
                            i,
                            total_step,
                            loss_record2.show(),
                        )
                    )

            if is_val:
                self.val(val_loader, epoch)

            os.makedirs(self.save_dir, exist_ok=True)
            if epoch > self.save_from and (epoch + 1) % 10 == 0 or epoch == 2:
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
            if self.scheduler:
                self.scheduler.step(epoch)

        self.writer.flush()
        self.writer.close()
        end = timeit.default_timer()

        self.logger.info("Training cost: " + str(end - start) + "seconds")


class Trainer3D:
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

        vals = AvgMeter()
        H, W, T = 240, 240, 155
        keys = ["whole", "core", "enhancing", "loss"]
        msg = ""

        (loss_record2,) = (AvgMeter(),)
        for i, pack in enumerate(val_loader, start=1):
            image, gt, gt_resize = pack
            self.net.eval()

            gt = gt[0]
            gt = np.asarray(gt, np.float32)
            res2 = 0
            image = image.cuda()
            gt_resize = gt_resize.cuda()

            res5, res4, res3, res2 = self.net(image)

            loss2 = self.loss(res2, gt_resize)
            loss_record2.update(loss2.data, 1)

            self.writer.add_scalar(
                "Loss2_val", loss_record2.show(), epoch * len(val_loader) + i
            )

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
                    )
                )

            output = res2[0, :, :H, :W, :T].cpu().detach().numpy()
            output = output.argmax(0)  # (channels,height,width,depth)

            target_cpu = gt[:H, :W, :T]
            scores = softmax_output_dice(output, target_cpu)
            vals.update(np.array(scores))

        msg = "Average scores:\n"
        msg += ", ".join(["{}: {:.4f}".format(k, v) for k, v in zip(keys, vals.avg)])
        self.logger.info(msg)

        # self.writer.add_scalar("mean_dice", mean_dice, epoch)

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
                self.optimizer.zero_grad()

                # ---- data prepare ----
                images, gts = pack
                # images, gts, paths, oriimgs = pack

                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # print(images.shape, gts.shape,"llll")  # (240, 240, 155, 4) (240, 240, 155)
                # import sys
                # sys.exit()

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

                loss_record2.update(loss2.data, batchsize)
                loss_record3.update(loss3.data, batchsize)
                loss_record4.update(loss4.data, batchsize)
                loss_record5.update(loss5.data, batchsize)
                loss_all.update(loss.data, batchsize)

                self.writer.add_scalar(
                    "Loss2",
                    loss_record2.show(),
                    epoch * len(train_loader) + i,
                )
                self.writer.add_scalar(
                    "Loss3",
                    loss_record3.show(),
                    epoch * len(train_loader) + i,
                )
                self.writer.add_scalar(
                    "Loss4",
                    loss_record4.show(),
                    epoch * len(train_loader) + i,
                )
                self.writer.add_scalar(
                    "Loss5",
                    loss_record5.show(),
                    epoch * len(train_loader) + i,
                )
                self.writer.add_scalar(
                    "Loss", loss_all.show(), epoch * len(train_loader) + i
                )

                total_step = len(train_loader)
                if i % 25 == 0 or i == total_step:
                    self.logger.info(
                        "{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                        [loss_record2: {:.4f},loss_record3: {:.4f},loss_record4: {:.4f},loss_record5: {:.4f}, loss_all: {:.4f}]".format(
                            datetime.now(),
                            epoch,
                            num_epochs,
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
            if epoch > self.save_from and (epoch + 1) % 25 == 0 or epoch == 2:
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

            self.scheduler.step(epoch)

        self.writer.flush()
        self.writer.close()
        end = timeit.default_timer()

        self.logger.info("Training cost: " + str(end - start) + "seconds")


from ..optim.losses import LocalSaliencyCoherence, SaliencyStructureConsistency


class TrainerSCWS:
    def __init__(self, net, optimizer, loss, scheduler, save_dir, save_from, logger):
        self.net = net
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_from = save_from
        self.writer = SummaryWriter()
        self.logger = logger

        self.loss_lsc = LocalSaliencyCoherence().cuda()
        self.criterion = torch.nn.CrossEntropyLoss(weight=None, reduction="mean")
        self.l = 0.3
        self.loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
        self.loss_lsc_radius = 5

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

            res5, res4, res3, res2 = self.net(image)
            loss2 = self.loss(res2, gt_resize)
            # loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4
            # loss = loss2 + loss3 + loss4 + loss5

            loss_record2.update(loss2.data, 1)

            self.writer.add_scalar(
                "Loss2_val", loss_record2.show(), epoch * len(val_loader) + i
            )

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
                    )
                )

            res = res2
            res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
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
        self.writer.add_scalar("mean_dice", mean_dice, epoch)

        self.writer.add_scalar("mean_iou", mean_iou, epoch)
        precision_all = tp_all / (tp_all + fp_all + 1e-07)
        recall_all = tp_all / (tp_all + fn_all + 1e-07)
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
        self.writer.add_scalar("dice_all", dice_all, epoch)
        self.writer.add_scalar("iou_all", iou_all, epoch)

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
                self.optimizer.zero_grad()

                # ---- data prepare ----
                images, gts = pack

                images = Variable(images).cuda()
                gts = Variable(gts).cuda()

                image_scale = F.interpolate(
                    images, scale_factor=0.8, mode="bilinear", align_corners=True
                )
                out5, out4, out3, out2 = self.net(images)
                out5_s, out4_s, out3_s, out2_s = self.net(image_scale)
                out2_scale = F.interpolate(
                    out2, scale_factor=0.8, mode="bilinear", align_corners=True
                )
                loss_ssc = SaliencyStructureConsistency(out2_s, out2_scale, 0.85)
                loss2 = loss_ssc + self.loss(out2, gts)  ## dominant loss
                loss4 = self.loss(out4, gts)
                loss3 = self.loss(out3, gts)
                loss5 = self.loss(out5, gts)

                # ######  saliency structure consistency loss  ######
                # image_scale = F.interpolate(images, scale_factor=0.8, mode='bilinear', align_corners=True)
                # out2, out3, out4, out5 = self.net(images)
                # out2_s, out3_s, out4_s, out5_s = self.net(image_scale)
                # out2_scale = F.interpolate(out2[:, 1:2], scale_factor=0.8, mode='bilinear', align_corners=True)
                # loss_ssc = SaliencyStructureConsistency(out2_s[:, 1:2], out2_scale, 0.85)
                # loss2 = loss_ssc + self.criterion(out2, fg_label) + self.criterion(out2, bg_label) + self.l * loss2_lsc  ## dominant loss

                # ######   label for partial cross-entropy loss  ######
                # gt = gts.squeeze(1).long()
                # bg_label = gt.clone()
                # fg_label = gt.clone()
                # bg_label[gt != 0] = 255
                # fg_label[gt == 0] = 255

                # ######   local saliency coherence loss (scale to realize large batchsize)  ######
                # image_ = F.interpolate(images, scale_factor=0.25, mode='bilinear', align_corners=True)
                # sample = {'rgb': image_}
                # out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
                # loss2_lsc = self.loss_lsc(out2_, self.loss_lsc_kernels_desc_defaults, self.loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
                # loss2 = loss_ssc + self.criterion(out2, fg_label) + self.criterion(out2, bg_label) + self.l * loss2_lsc  ## dominant loss

                # ######  auxiliary losses  ######
                # out3_ = F.interpolate(out3[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
                # loss3_lsc = self.loss_lsc(out3_, self.loss_lsc_kernels_desc_defaults, self.loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
                # loss3 = self.criterion(out3, fg_label) + self.criterion(out3, bg_label) + self.l * loss3_lsc
                # out4_ = F.interpolate(out4[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
                # loss4_lsc = self.loss_lsc(out4_, self.loss_lsc_kernels_desc_defaults, self.loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
                # loss4 = self.criterion(out4, fg_label) + self.criterion(out4, bg_label) + self.l * loss4_lsc
                # out5_ = F.interpolate(out5[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
                # loss5_lsc = self.loss_lsc(out5_, self.loss_lsc_kernels_desc_defaults, self.loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
                # loss5 = self.criterion(out5, fg_label) + self.criterion(out5, bg_label) + self.l * loss5_lsc

                ######  objective function  ######
                loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4

                loss.backward()
                clip_gradient(self.optimizer, clip)
                self.optimizer.step()

                loss_record2.update(loss2.data, batchsize)
                loss_record3.update(loss3.data, batchsize)
                loss_record4.update(loss4.data, batchsize)
                loss_record5.update(loss5.data, batchsize)
                loss_all.update(loss.data, batchsize)

                self.writer.add_scalar(
                    "Loss2",
                    loss_record2.show(),
                    epoch * len(train_loader) + i,
                )
                self.writer.add_scalar(
                    "Loss3",
                    loss_record3.show(),
                    epoch * len(train_loader) + i,
                )
                self.writer.add_scalar(
                    "Loss4",
                    loss_record4.show(),
                    epoch * len(train_loader) + i,
                )
                self.writer.add_scalar(
                    "Loss5",
                    loss_record5.show(),
                    epoch * len(train_loader) + i,
                )
                self.writer.add_scalar(
                    "Loss", loss_all.show(), epoch * len(train_loader) + i
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
