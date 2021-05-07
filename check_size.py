from torchsummary import summary
import network.models as models

# GCPAGALDNetv8
# model = models.__dict__["GCPAGALDNetv4"]()

n_skip = 3
vit_name = "R50-ViT-B_16"
vit_patches_size = 16
img_size = 352
import torch.backends.cudnn as cudnn
from network.models.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import numpy as np

config_vit = CONFIGS_ViT_seg[vit_name]
config_vit.n_classes = 1
config_vit.n_skip = n_skip
if vit_name.find("R50") != -1:
    config_vit.patches.grid = (
        int(img_size / vit_patches_size),
        int(img_size / vit_patches_size),
    )



import network.models as models

# from network.contextagg.aspp import DeepLabv3
# model = DeepLabv3(2)
import torch

device = torch.device("cuda")
# SCWSPSPResNet
# SCWSPSP3DNet
# model = models.__dict__["GCPACCNet"]().to(device)  # Pranet
model = models.__dict__["GCPATrans"](
    config_vit, img_size=img_size
).to(device)  # TransUnet
# model = models.__dict__["TransUnet"](
#     config_vit, img_size=img_size, num_classes=config_vit.n_classes
# )  # TransUnet

# model.cuda()



from torchsummaryX import summary
summary(model, torch.rand((1, 3, 352, 352)).cuda())

# summary(model, (3, 352, 352))


# from network.encoders import resnet3D50
# device = torch.device("cuda")
# model = resnet3D50().to(device)
# summary(model, (3, 16, 352, 352))
