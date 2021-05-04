from .pranet import PraNetvFAMGALD, PraNet, PraNetvGALD, PraNetvFAM2GALD
from .trainer import Trainer, TransUnetTrainer, TrainerGCPAGALD, TrainerSCWS, Trainer3D
from .hardnet_mseg import HardnetMSEG
from .transunet import *
from .gcpanet import (
    GCPANet,
    GCPAGALDNet,
    GCPAGALDNetv2,
    GCPAGALDNetv3,
    GCPAGALDNetv4,
    GCPAGALDNetv5,
    GCPAGALDNetv6,
    GCPAGALDNetv7,
    GCPAGALDNetv8,
    GCPAGALDNetv9,
    GCPAASPPNet,
    GCPAPSPNet,
    GCPAPSPSmallNet,
    GCPAPSPAGNet,
    GCPACCNet,
    GCPACC3GANet,
    GCPACC2Net,
    GCPACCv2Net,
    GCPACCv3Net,
    GCPACCv4Net,
    GCPACCv5Net,
    GCPACCDualNet,
    GCPACCASPPNet,
    GCPAPSPAGv2Net,
    SCWSCCNet,
    SCWSCC3GANet,
    SCWSPSPNet,
    SCWSPSPAGNet,
    SCWSPSPResNet,
    SCWSPSPHardNet,
    SCWSCC2Net,
    SCWSASPPNet,
)
from .v3D import SCWSPSPRes3DNet, SCWSPSPHard3DNet

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


from .gcpanet.gcpanet import ResNet


class UNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()
        self.bkbone = ResNet()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up3 = double_conv(1024 + 2048, 1024)
        self.dconv_up2 = double_conv(512 + 1024, 512)
        self.dconv_up1 = double_conv(256 + 512, 256)

        self.conv_out2 = nn.Conv2d(2048, n_class, 1)
        self.conv_out3 = nn.Conv2d(1024, n_class, 1)
        self.conv_out4 = nn.Conv2d(512, n_class, 1)
        self.conv_out5 = nn.Conv2d(256, n_class, 1)

    def forward(self, inp):
        out1, out2_, out3_, out4_, out5_ = self.bkbone(inp)
        # print(out1.shape, out2.shape, out3.shape, out4.shape, out5.shape, "ppppp")

        # conv1 = self.dconv_down1(inp)
        # x = self.maxpool(conv1)

        # conv2 = self.dconv_down2(x)
        # x = self.maxpool(conv2)

        # conv3 = self.dconv_down3(x)
        # x = self.maxpool(conv3)

        # x = self.dconv_down4(x)
        x = out5_
        out5 = F.interpolate(self.conv_out2(x), size=inp.size()[2:], mode="bilinear")

        x = self.upsample(x)
        # print(x.shape, out4.shape)
        x = torch.cat([x, out4_], dim=1)
        x = self.dconv_up3(x)
        out4 = F.interpolate(self.conv_out3(x), size=inp.size()[2:], mode="bilinear")

        x = self.upsample(x)
        x = torch.cat([x, out3_], dim=1)
        x = self.dconv_up2(x)
        out3 = F.interpolate(self.conv_out4(x), size=inp.size()[2:], mode="bilinear")

        x = self.upsample(x)
        x = torch.cat([x, out2_], dim=1)
        x = self.dconv_up1(x)
        out2 = F.interpolate(self.conv_out5(x), size=inp.size()[2:], mode="bilinear")

        return out5, out4, out3, out2