import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...contextagg import LocalAttenModule, PSPModule
from ...encoders import hardnet


class FAMSCWSAG(nn.Module):
    def __init__(
        self, in_channel_left, in_channel_down, in_channel_right, interplanes=256
    ):
        super(FAMSCWSAG, self).__init__()
        # self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)

        self.conv_l0 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnl0 = nn.BatchNorm2d(interplanes)
        self.conv_d0 = nn.Conv2d(
            in_channel_down, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnd0 = nn.BatchNorm2d(interplanes)

        self.conv_l1 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnl1 = nn.BatchNorm2d(interplanes)
        self.conv_d1 = nn.Conv2d(
            in_channel_down, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnd1 = nn.BatchNorm2d(interplanes)

        self.conv_l2 = nn.Conv2d(
            in_channel_left, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnl2 = nn.BatchNorm2d(interplanes)

        self.conv_r2 = nn.Conv2d(
            in_channel_right, interplanes, kernel_size=1, stride=1, padding=1
        )
        self.bnr2 = nn.BatchNorm2d(interplanes)

        self.psi_1 = nn.Sequential(
            nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.psi_2 = nn.Sequential(
            nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.psi_3 = nn.Sequential(
            nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.conv_out = nn.Conv2d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn_out = nn.BatchNorm2d(interplanes)

        self.conv_att1 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)	
        self.conv_att2 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)	
        self.conv_att3 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)	

    def forward(self, left, down, right):

        # BRANCH 1: LOW GUIDE HIGH
        left1 = self.bnl0(self.conv_l0(left))  # 256 channels
        down1 = self.bnd0(self.conv_d0(down))  # 256 channels

        if down1.size()[2:] != left1.size()[2:]:
            down1 = F.interpolate(down1, size=left1.size()[2:], mode="bilinear")
        psi_1 = F.relu(left1 + down1)
        psi_1 = self.psi_1(psi_1)
        zdl = down1 * psi_1

        z1_att = F.adaptive_avg_pool2d(self.conv_att1(zdl), (1,1))	
        zdl = z1_att * zdl	

        # BRANCH 2: HIGH GUIDE LOW
        left2 = self.bnl1(self.conv_l1(left))  # 256 channels
        down2 = self.bnd1(self.conv_d1(down))  # 256 channels
        # w1 = self.conv_l(left)
        if down2.size()[2:] != left2.size()[2:]:
            down2 = F.interpolate(down2, size=left2.size()[2:], mode="bilinear")
        psi_2 = F.relu(left2 + down2)
        psi_2 = self.psi_2(psi_2)
        zld = left2 * psi_2
        # z2 = F.relu(down_1 * left2, inplace=True)  # left is mask

        z2_att = F.adaptive_avg_pool2d(self.conv_att2(zld), (1,1))	
        zld = z2_att * zld	

        # BRANCH 3: CONTEXT GUIDE LOW
        left3 = self.bnl2(self.conv_l2(left))  # 256 channels
        right3 = self.bnr2(self.conv_r2(right))  # 256
        if right3.size()[2:] != left3.size()[2:]:
            right3 = F.interpolate(right3, size=left3.size()[2:], mode="bilinear")
        psi_3 = F.relu(left3 + right3)
        psi_3 = self.psi_3(psi_3)
        zlr = left3 * psi_3

        z3_att = F.adaptive_avg_pool2d(self.conv_att3(zlr), (1,1))	
        zlr = z3_att * zlr	

        out = (zld + zdl + zlr) / (z1_att + z2_att + z3_att)	
        return F.relu(self.bn_out(self.conv_out(out)), inplace=True)	


class SCWSPSPAGNet(nn.Module):
    def __init__(self):
        super(SCWSPSPAGNet, self).__init__()

        self.hardnet = hardnet(arch=68)

        inplanes = 1024
        interplanes = 256

        self.fam45 = FAMSCWSAG(640, interplanes, interplanes, interplanes)
        self.fam34 = FAMSCWSAG(320, interplanes, interplanes, interplanes)
        self.fam23 = FAMSCWSAG(128, interplanes, interplanes, interplanes)

        self.linear5 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)

        self.conva = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )
        # self.long_relation = SpatialCGNL(interplanes, interplanes // 2)
        self.long_relation = PSPModule(inplanes, interplanes)
        self.local_attention_4 = LocalAttenModule(interplanes)
        self.local_attention_3 = LocalAttenModule(interplanes)
        self.local_attention_2 = LocalAttenModule(interplanes)

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

        out5_c = self.long_relation(out5_)  # bs, 256, 11, 11

        # GCF
        out4_c = self.local_attention_4(out5_c)  # bs, 256, 11, 11

        out3_c = self.local_attention_3(out5_c)  # bs, 256, 11, 11

        out2_c = self.local_attention_2(out5_c)  # bs, 256, 11, 11

        # HA
        out5 = self.conva(out5_)  # bs, 256, 11, 11

        # out
        out4 = self.fam45(out4, out5, out4_c)
        out3 = self.fam34(out3, out4, out3_c)
        out2 = self.fam23(out2, out3, out2_c)
        # we use bilinear interpolation instead of transpose convolution
        out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode="bilinear")
        out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode="bilinear")
        out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode="bilinear")
        out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode="bilinear")
        return out5, out4, out3, out2
