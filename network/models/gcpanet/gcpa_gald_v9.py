import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...contextagg import GALDHead, GALDBlock, SpatialCGNL, LocalAttenModule
from torch.nn import BatchNorm2d, BatchNorm1d

# from .gcpa_gald import *
from ...encoders import res2net50_v1b_26w_4s, hardnet
from .gcpanet import ResNet


class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, in_channel_right):
        super(FAM, self).__init__()
        # self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(
            in_channel_right, 256, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(256)

        self.conv_d1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256 * 2, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True)  # 256

        down_1 = self.conv_d1(down)

        w1 = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode="bilinear")
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)  # down is mask

        # if down_1.size()[2:] != left.size()[2:]:
        #     down_1 = F.interpolate(down_1, size=left.size()[2:], mode="bilinear")

        # z2 = F.relu(down_1 * left, inplace=True)  # left is mask

        # z3
        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode="bilinear")
        z3 = F.relu(down_2 * left, inplace=True)  # down_2 is mask

        out = torch.cat((z1, z3), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)


class GCPAGALDNetv9(nn.Module):
    def __init__(self):
        super(GCPAGALDNetv9, self).__init__()

        self.hardnet = hardnet(arch=68)

        self.fam45 = FAM(640, 256, 256)
        self.fam34 = FAM(320, 256, 256)
        self.fam23 = FAM(128, 256, 256)

        # self.ca55 = CA(256, 1024)
        # self.sa55 = SA(1024, 1024)

        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        inplanes = 1024
        interplanes = 256
        self.conva = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )

        self.long_relation = SpatialCGNL(interplanes, interplanes // 2)
        self.local_attention_4 = LocalAttenModule(interplanes)
        self.local_attention_3 = LocalAttenModule(interplanes)
        self.local_attention_2 = LocalAttenModule(interplanes)
        self.local_attention = LocalAttenModule(interplanes)

    def forward(self, x):

        hardnetout = self.hardnet(x)
        # out1 = self.resnet.maxpool(out1)  # bs, 64, 88, 88

        # out2 = self.resnet.layer1(out1)  # bs, 256, 88, 88
        # out3 = self.resnet.layer2(out2)  # bs, 512, 44, 44
        # out4 = self.resnet.layer3(out3)  # bs, 1024, 22, 22
        # out5_ = self.resnet.layer4(out4)  # bs, 2048, 11, 11

        out2 = hardnetout[0]  # [24, 128, 88, 88]
        out3 = hardnetout[1]  # [24, 320, 44, 44]
        out4 = hardnetout[2]  # [24, 640, 22, 22]
        out5_ = hardnetout[3]  # [24, 1024, 11, 11]

        out5_ = self.conva(out5_)  # bs, 256, 11, 11
        out5_c = self.long_relation(out5_)  # bs, 256, 11, 11

        # GCF
        out4_c = self.local_attention_4(out5_c)  # bs, 256, 11, 11

        out3_c = self.local_attention_3(out5_c)  # bs, 256, 11, 11

        out2_c = self.local_attention_2(out5_c)  # bs, 256, 11, 11

        # HA
        out5 = out5_  # bs, 256, 11, 11

        # out
        out4 = self.fam45(out4, out5, out4_c)   # bs, 256, 22, 22
        out3 = self.fam34(out3, out4, out3_c)  # bs, 256, 44, 44
        out2 = self.fam23(out2, out3, out2_c)   # bs, 256, 88, 88
        # we use bilinear interpolation instead of transpose convolution
        out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode="bilinear")
        out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode="bilinear")
        out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode="bilinear")
        out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode="bilinear")
        return out5, out4, out3, out2
