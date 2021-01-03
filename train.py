import torch
from torch.autograd import Variable
import os
from datetime import datetime
import torch.nn.functional as F
import cv2
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from glob import glob
from tqdm import tqdm
from skimage.io import imread
import numpy as np
import sys
import matplotlib.pyplot as plt
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

    elif(version == "GALD" or version == 14):
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
              


    save_path = 'snapshots/{}/'.format(train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 3 == 0 and epoch > save_from or epoch == 23:
      torch.save({"model_state_dict":model.state_dict(), "lr":optimizer.param_groups[0]["lr"]}, save_path + 'PraNetDG-' + test_fold +'-%d.pth' % epoch)
      Log.info('[Saving Snapshot:]'+  save_path + 'PraNetDG-' + test_fold +'-%d.pth' % epoch)

