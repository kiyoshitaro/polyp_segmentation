from .pranet import PraNetvFAMGALD, PraNet, PraNetvGALD, PraNetvFAM2GALD
from .trainer import (
    Trainer,
    TransUnetTrainer,
    TrainerGCPAGALD,
    TrainerSCWS,
    Trainer3D,
    TrainerOne,
    TrainerDistillation,
)
from .dunet import DUNet
from .hardnet_mseg import HardnetMSEG
from .transunet import (
    TransHarD, TransUnet
)
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
    GCPAPSP2Net,
    GCPAPSPSmallNet,
    GCPAPSPAGNet,
    GCPACCNet,
    GCPATrans,
    GCPACC3GANet,
    GCPACC2Net,
    GCPACCv2Net,
    GCPACCv3Net,
    GCPACCv4Net,
    GCPACCv5Net,
    GCPACCDualNet,
    GCPACCASPPNet,
    GCPAPSPAGv2Net,
    GCPARCCANet,
    SCWSCCNet,
    SCWSCC3GANet,
    SCWSPSPNet,
    SCWSPSPv3Net,
    SCWSPSP2Net,
    SCWSPSPAGNet,
    SCWSPSPResNet,
    SCWSPSPHardNet,
    SCWSCC2Net,
    SCWSASPPNet,
    SCWSRCCANet,
    SCWSLambdaNet,
    SCWSBottleStackNet,
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


# class UNet(nn.Module):
#     def __init__(self, n_class=1):
#         super().__init__()
#         self.bkbone = ResNet()

#         self.dconv_down1 = double_conv(3, 64)
#         self.dconv_down2 = double_conv(64, 128)
#         self.dconv_down3 = double_conv(128, 256)
#         self.dconv_down4 = double_conv(256, 512)

#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

#         self.dconv_up3 = double_conv(1024 + 2048, 1024)
#         self.dconv_up2 = double_conv(512 + 1024, 512)
#         self.dconv_up1 = double_conv(256 + 512, 256)

#         self.conv_out2 = nn.Conv2d(2048, n_class, 1)
#         self.conv_out3 = nn.Conv2d(1024, n_class, 1)
#         self.conv_out4 = nn.Conv2d(512, n_class, 1)
#         self.conv_out5 = nn.Conv2d(256, n_class, 1)

#     def forward(self, inp):
#         out1, out2_, out3_, out4_, out5_ = self.bkbone(inp)
#         # print(out1.shape, out2.shape, out3.shape, out4.shape, out5.shape, "ppppp")

#         # conv1 = self.dconv_down1(inp)
#         # x = self.maxpool(conv1)

#         # conv2 = self.dconv_down2(x)
#         # x = self.maxpool(conv2)

#         # conv3 = self.dconv_down3(x)
#         # x = self.maxpool(conv3)

#         # x = self.dconv_down4(x)
#         x = out5_
#         out5 = F.interpolate(self.conv_out2(x), size=inp.size()[2:], mode="bilinear")

#         x = self.upsample(x)
#         # print(x.shape, out4.shape)
#         x = torch.cat([x, out4_], dim=1)
#         x = self.dconv_up3(x)
#         out4 = F.interpolate(self.conv_out3(x), size=inp.size()[2:], mode="bilinear")

#         x = self.upsample(x)
#         x = torch.cat([x, out3_], dim=1)
#         x = self.dconv_up2(x)
#         out3 = F.interpolate(self.conv_out4(x), size=inp.size()[2:], mode="bilinear")

#         x = self.upsample(x)
#         x = torch.cat([x, out2_], dim=1)
#         x = self.dconv_up1(x)
#         out2 = F.interpolate(self.conv_out5(x), size=inp.size()[2:], mode="bilinear")

#         return out5, out4, out3, out2


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
