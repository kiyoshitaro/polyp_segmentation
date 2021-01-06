import torch
from torch.autograd import Variable
import os
from datetime import datetime
import torch.nn.functional as F
import cv2
from utils.utils import clip_gradient, AvgMeter
from glob import glob
from skimage.io import imread
import numpy as np
import sys
from utils.logger import Logger as Log



class Dataset_test(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = imread(img_path)
        mask = imread(mask_path)
        image = cv2.resize(image, (352, 352))

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        
        mask = mask.astype('float32')
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)
  
  
class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = imread(img_path)
        mask = imread(mask_path)

        if self.aug:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        
        mask = mask.astype('float32')
        mask = mask.transpose((2, 0, 1))
        return np.asarray(image), np.asarray(mask)

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader,test_loader, model, optimizer, epoch, test_fold,writer,args):
    test_size = len(test_loader)
    clip = args["clip"]
    batchsize = args["batchsize"]
    trainsize_init = args["trainsize_init"]
    total_step = args["total_step"]
    train_save = args["train_save"]
    version = args["version"]
    save_from = args["save_from"]
    model.train()
    # ---- multi-scale training ----6
    size_rates = [0.75, 1, 1.25]

    if(version == 0 or version == 1 or version == 2 or version == 3):
        loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(trainsize_init*rate/32)*32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)      

                # ---- forward ----
                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)

                # ---- loss function ----
                loss5 = structure_loss(lateral_map_5, gts)
                loss4 = structure_loss(lateral_map_4, gts)
                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)
                loss = loss2 + loss3 + loss4 + loss5

                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record2.update(loss2.data, batchsize)
                    loss_record3.update(loss3.data, batchsize)
                    loss_record4.update(loss4.data, batchsize)
                    loss_record5.update(loss5.data, batchsize)                  
                    
                    writer.add_scalar("Loss1", loss_record2.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss2", loss_record3.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss3", loss_record4.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss4", loss_record5.show(), (epoch-1)*len(train_loader) + i)
                    

            if i % 25 == 0 or i == total_step:
                Log.info('{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                    [loss_record2: {:.4f},loss_record3: {:.4f},loss_record4: {:.4f},loss_record5: {:.4f}]'.
                    format(datetime.now(), epoch, epoch, optimizer.param_groups[0]["lr"],i, total_step,\
                            loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()
                            ))
    elif(version == 4 or version == 5 or version == 13):
        loss_recordx4, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(trainsize_init*rate/32)*32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)      

                # ---- forward ----
                x4_head_out, lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)

                # ---- loss function ----
                lossx4 = structure_loss(x4_head_out, gts)
                loss5 = structure_loss(lateral_map_5, gts)
                loss4 = structure_loss(lateral_map_4, gts)
                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)
                loss = lossx4*2 + loss2 + loss3 + loss4 + loss5

                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_recordx4.update(lossx4.data, batchsize)
                    loss_record2.update(loss2.data, batchsize)
                    loss_record3.update(loss3.data, batchsize)
                    loss_record4.update(loss4.data, batchsize)
                    loss_record5.update(loss5.data, batchsize)

                    
                    writer.add_scalar("loss_recordx4", loss_recordx4.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss1", loss_record2.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss2", loss_record3.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss3", loss_record4.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss4", loss_record5.show(), (epoch-1)*len(train_loader) + i)
                    

            if i % 25 == 0 or i == total_step:
                Log.info('{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                    [loss_recordx4: {:.4f},loss_record2: {:.4f},loss_record3: {:.4f},loss_record4: {:.4f},loss_record5: {:.4f}]'.
                    format(datetime.now(), epoch, epoch, optimizer.param_groups[0]["lr"],i, total_step,\
                             loss_recordx4.show(),\
                            loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()
                            ))

    elif(version == "GALD" or version == 14 or version == 15 or version == 16):
        loss_recordx3, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(trainsize_init*rate/32)*32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)      

                # ---- forward ----
                x3_head_out, lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)

                # ---- loss function ----
                lossx3 = structure_loss(x3_head_out, gts)
                loss5 = structure_loss(lateral_map_5, gts)
                loss4 = structure_loss(lateral_map_4, gts)
                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)
                loss = lossx3*2 + loss2 + loss3 + loss4 + loss5

                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_recordx3.update(lossx3.data, batchsize)
                    loss_record2.update(loss2.data, batchsize)
                    loss_record3.update(loss3.data, batchsize)
                    loss_record4.update(loss4.data, batchsize)
                    loss_record5.update(loss5.data, batchsize)

                    
                    writer.add_scalar("loss_recordx3", loss_recordx3.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss1", loss_record2.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss2", loss_record3.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss3", loss_record4.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss4", loss_record5.show(), (epoch-1)*len(train_loader) + i)
                    

            if i % 25 == 0 or i == total_step:
                Log.info('{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                    [loss_recordx3: {:.4f},loss_record2: {:.4f},loss_record3: {:.4f},loss_record4: {:.4f},loss_record5: {:.4f}]'.
                    format(datetime.now(), epoch, epoch, optimizer.param_groups[0]["lr"],i, total_step,\
                             loss_recordx3.show(),\
                            loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()
                            ))


    elif(version == 6 or version == 7):
        loss_recordx_GALD, loss_recordx_DUAL, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(trainsize_init*rate/32)*32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)      

                # ---- forward ----
                x_gald_head_out, x_dual_head_out, lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)

                # ---- loss function ----
                lossx_gald = structure_loss(x_gald_head_out, gts)
                lossx_dual = structure_loss(x_dual_head_out, gts)
                loss5 = structure_loss(lateral_map_5, gts)
                loss4 = structure_loss(lateral_map_4, gts)
                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)
                loss = (lossx_dual + lossx_gald)*2 + loss2 + loss3 + loss4 + loss5

                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_recordx_GALD.update(lossx_gald.data, batchsize)
                    loss_recordx_DUAL.update(lossx_dual.data, batchsize)
                    loss_record2.update(loss2.data, batchsize)
                    loss_record3.update(loss3.data, batchsize)
                    loss_record4.update(loss4.data, batchsize)
                    loss_record5.update(loss5.data, batchsize)

                    
                    writer.add_scalar("loss_recordx_GALD", loss_recordx_GALD.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("loss_recordx_DUAL", loss_recordx_DUAL.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss1", loss_record2.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss2", loss_record3.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss3", loss_record4.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss4", loss_record5.show(), (epoch-1)*len(train_loader) + i)
                    

            if i % 25 == 0 or i == total_step:
                Log.info('{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                    [loss_recordx_DUAL: {:.4f},loss_recordx_GALD: {:.4f},loss_record2: {:.4f},loss_record3: {:.4f},loss_record4: {:.4f},loss_record5: {:.4f}]'.
                    format(datetime.now(), epoch, epoch, optimizer.param_groups[0]["lr"],i, total_step,\
                             loss_recordx_DUAL.show(),\
                             loss_recordx_GALD.show(),\
                            loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()
                            ))



    elif(version == "W"):
        loss_record5_new,loss_record4_new,loss_record3_new,loss_record2_new,loss_record1_new,loss_record0_new,loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(trainsize_init*rate/32)*32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)      

                # ---- forward ----        
                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, new_lateral_map_5, new_lateral_map_4, new_lateral_map_3, new_lateral_map_2, new_lateral_map_1, new_lateral_map_0 = model(images)

                # ---- loss function ----
                # lossx4 = structure_loss(x4_head_out, gts)
                lossx3 = structure_loss(x3_head_out, gts)
                # lossx2 = structure_loss(x2_head_out, gts)
                loss5 = structure_loss(lateral_map_5, gts)
                loss4 = structure_loss(lateral_map_4, gts)
                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)
                loss5_new = structure_loss(new_lateral_map_5, gts)
                loss4_new = structure_loss(new_lateral_map_4, gts)
                loss3_new = structure_loss(new_lateral_map_3, gts)
                loss2_new = structure_loss(new_lateral_map_2, gts)
                loss1_new = structure_loss(new_lateral_map_1, gts)
                loss0_new = structure_loss(new_lateral_map_0, gts)

                loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss
                loss += loss5_new + loss4_new + loss3_new + loss2_new + loss1_new + loss0_new
        
                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record2.update(loss2.data, batchsize)
                    loss_record3.update(loss3.data, batchsize)
                    loss_record4.update(loss4.data, batchsize)
                    loss_record5.update(loss5.data, batchsize)
                    
                    loss_record5_new.update(loss5_new.data, batchsize)		
                    loss_record4_new.update(loss4_new.data, batchsize)		
                    loss_record3_new.update(loss3_new.data, batchsize)		
                    loss_record2_new.update(loss2_new.data, batchsize)		
                    loss_record1_new.update(loss1_new.data, batchsize)		
                    loss_record0_new.update(loss0_new.data, batchsize)	
                    
                    
                    writer.add_scalar("Loss1", loss_record2.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss2", loss_record3.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss3", loss_record4.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss4", loss_record5.show(), (epoch-1)*len(train_loader) + i)
                    

                    writer.add_scalar("loss_record5_new", loss_record5_new.show(), (epoch-1)*len(train_loader) + i)		
                    writer.add_scalar("loss_record4_new", loss_record4_new.show(), (epoch-1)*len(train_loader) + i)		
                    writer.add_scalar("loss_record3_new", loss_record3_new.show(), (epoch-1)*len(train_loader) + i)		
                    writer.add_scalar("loss_record2_new", loss_record2_new.show(), (epoch-1)*len(train_loader) + i)		
                    writer.add_scalar("loss_record1_new", loss_record1_new.show(), (epoch-1)*len(train_loader) + i)		
                    writer.add_scalar("loss_record0_new", loss_record0_new.show(), (epoch-1)*len(train_loader) + i)	

            if i % 25 == 0 or i == total_step:
                Log.info('{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                    [loss_record2: {:.4f},loss_record3: {:.4f},loss_record4: {:.4f},loss_record5: {:.4f}]'.
                    format(datetime.now(), epoch, epoch, optimizer.param_groups[0]["lr"],i, total_step,\
                            loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()
                            ))



    elif(version == 8):
        loss_recordx2, loss_recordx3, loss_recordx4, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter()
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(trainsize_init*rate/32)*32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)      

                # ---- forward ----
                x4_head_out, x3_head_out, x2_head_out, lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)

                # ---- loss function ----
                lossx4 = structure_loss(x4_head_out, gts)
                lossx3 = structure_loss(x3_head_out, gts)
                lossx2 = structure_loss(x2_head_out, gts)
                loss5 = structure_loss(lateral_map_5, gts)
                loss4 = structure_loss(lateral_map_4, gts)
                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)
                loss = (lossx4 + lossx3 + lossx2)*2 + loss2 + loss3 + loss4 + loss5

                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_recordx4.update(lossx4.data, batchsize)
                    loss_recordx3.update(lossx3.data, batchsize)
                    loss_recordx2.update(lossx2.data, batchsize)
                    loss_record2.update(loss2.data, batchsize)
                    loss_record3.update(loss3.data, batchsize)
                    loss_record4.update(loss4.data, batchsize)
                    loss_record5.update(loss5.data, batchsize)

                    
                    writer.add_scalar("loss_recordx4", loss_recordx4.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("loss_recordx3", loss_recordx3.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("loss_recordx2", loss_recordx2.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss1", loss_record2.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss2", loss_record3.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss3", loss_record4.show(), (epoch-1)*len(train_loader) + i)
                    writer.add_scalar("Loss4", loss_record5.show(), (epoch-1)*len(train_loader) + i)
                    

            if i % 25 == 0 or i == total_step:
                Log.info('{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                    [loss_recordx4: {:.4f},loss_recordx3: {:.4f},loss_recordx2: {:.4f},loss_record2: {:.4f},loss_record3: {:.4f},loss_record4: {:.4f},loss_record5: {:.4f}]'.
                    format(datetime.now(), epoch, epoch, optimizer.param_groups[0]["lr"],i, total_step,\
                             loss_recordx4.show(),\
                             loss_recordx3.show(),\
                             loss_recordx2.show(),\
                            loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()
                            ))





    elif(version == 12):

        loss_recordx4,\
        loss_recordx3,\
        loss_recordx2,\
        loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(trainsize_init*rate/32)*32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)      

                # ---- forward ----
                x3_head_out = model(images)
                # ---- loss function ----
                lossx3 = structure_loss(x3_head_out, gts)
                loss = lossx3
                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_recordx3.update(lossx3.data, batchsize)
                    writer.add_scalar("loss_recordx3", loss_recordx3.show(), (epoch-1)*len(train_loader) + i)

            if i % 25 == 0 or i == total_step:
                Log.info('{} Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}/{:04d}],\
                    [loss_recordx3: {:.4f}]'.
                    format(datetime.now(), epoch, epoch, optimizer.param_groups[0]["lr"],i, total_step,\
                            loss_recordx3.show(),\
                            ))
    else:
        print("No have" + version + " version")

    save_path = 'snapshots/{}/'.format(train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 3 == 0 and epoch > save_from or epoch == 23:
      torch.save({"model_state_dict":model.state_dict(), "lr":optimizer.param_groups[0]["lr"]}, save_path + 'PraNetDG-' + test_fold +'-%d.pth' % epoch)
      Log.info('[Saving Snapshot:]'+  save_path + 'PraNetDG-' + test_fold +'-%d.pth' % epoch)




if __name__ == "__main__":

    from lib.PraNet_Res2Net import PraNetv16
    import os
    from utils.logger import Logger as Log
    # from train import Dataset, Dataset_test, train
    from albumentations.core.composition import Compose, OneOf
    from glob import glob
    from utils.utils import clip_gradient, adjust_lr, AvgMeter
    import timeit
    from albumentations.augmentations import transforms
    import torch

    lr = 1e-4
    batchsize = 16
    trainsize_init = 352
    clip = 0.5
    decay_rate = 0.1
    decay_epoch = 50
    start_from = 0
    save_from = 60
    name = [[1,2,3,4], [0,2,3,4], [0,1,3,4], [0,1,2,4], [0,1,2,3]]
    start = timeit.default_timer()
    v = 16
    i = 0
    train_save = 'PraNetv{}_Res2Net_kfold'.format(v)
    save_path = 'snapshots/{}/'.format(train_save)
    log_file = 'PraNetv{}_Res2Net_fold{}.log'.format(v,i)
    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = PraNetv16().cuda()
    if start_from != 0: 
    restore_from = "./snapshots/PraNetv{}_Res2Net_kfold/PraNetDG-fold{}-{}.pth".format(v,i,start_from)
    saved_state_dict = torch.load(restore_from)["model_state_dict"]
    lr = torch.load(restore_from)["lr"]
    model.load_state_dict(saved_state_dict, strict=False)


    train1 = 'fold_' + str(name[i][0])
    train2 = 'fold_' + str(name[i][1])
    train3 = 'fold_' + str(name[i][2])
    train4 = 'fold_' + str(name[i][3])
    test_fold = 'fold' + str(i)
    train_img_paths =[]
    train_mask_paths = []
    train_img_path_1 = glob('Kvasir_fold_new/' + train1 + "/images/*")
    train_img_paths.extend(train_img_path_1)
    train_img_path_2 = glob('Kvasir_fold_new/' + train2 + "/images/*")
    train_img_paths.extend(train_img_path_2)
    train_img_path_3 = glob('Kvasir_fold_new/' + train3 + "/images/*")
    train_img_paths.extend(train_img_path_3)
    train_img_path_4 = glob('Kvasir_fold_new/' + train4 + "/images/*")
    train_img_paths.extend(train_img_path_4)
    train_mask_path_1 = glob('Kvasir_fold_new/' + train1 + "/masks/*")
    train_mask_paths.extend(train_mask_path_1)
    train_mask_path_2 = glob('Kvasir_fold_new/' + train2 + "/masks/*")
    train_mask_paths.extend(train_mask_path_2)
    train_mask_path_3 = glob('Kvasir_fold_new/' + train3 + "/masks/*")
    train_mask_paths.extend(train_mask_path_3)
    train_mask_path_4 = glob('Kvasir_fold_new/' + train4 + "/masks/*")
    train_mask_paths.extend(train_mask_path_4)
    train_img_paths.sort()
    train_mask_paths.sort()

    train_transform = Compose([
            transforms.RandomRotate90(),
            transforms.Flip(),
            transforms.HueSaturationValue(),
            transforms.RandomBrightnessContrast(),
            transforms.Transpose(),
            OneOf([
            transforms.RandomCrop(220,220, p=0.5),
            transforms.CenterCrop(220,220, p=0.5)
            ], p=0.5),
            transforms.Resize(352,352)
        ])

    train_dataset = Dataset(train_img_paths, train_mask_paths, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    total_step = len(train_loader)

    data_path = 'Kvasir_fold_new/' + 'fold_' + str(i)
    X_test = glob('{}/images/*'.format(data_path))
    X_test.sort()
    y_test = glob('{}/masks/*'.format(data_path))
    y_test.sort()
    test_dataset = Dataset_test(X_test, y_test,aug=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=True)

    # ---- flops and params ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr)
    Log.init(
        log_level="info",
        log_file=os.path.join(save_path, log_file),
        log_format="%(asctime)s %(levelname)-7s %(message)s",
        rewrite=False,
        stdout_level="info"
    )
    Log.info("#"*20 + f"Start Training Fold{i}" + "#"*20)
    print("#"*20, f"Start Training Fold{i}", "#"*20)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    args = {
    'lr' :lr, 'batchsize':batchsize, "trainsize_init" : trainsize_init,"clip" : clip,
    'decay_rate':decay_rate,'start_from' : start_from, "total_step": total_step, "train_save" : train_save,
    'version' : v, 'save_from' : save_from,
    }

    for epoch in range(start_from, 100):
        adjust_lr(optimizer, lr, epoch, decay_rate, decay_epoch)
        train(train_loader, test_loader , model, optimizer, epoch, test_fold,writer,args)

    writer.flush()
    writer.close()
    end = timeit.default_timer()

    Log.info("Training cost: "+ str(end - start) + 'seconds')

