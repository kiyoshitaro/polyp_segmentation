import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...contextagg import LocalAttenModule, Atrous_module
from torch.nn import BatchNorm2d, BatchNorm1d
from .gcpa_gald import FAM
from ...encoders import res2net50_v1b_26w_4s, hardnet
from .gcpanet import ResNet


class GCPAASPPNet(nn.Module):
    def __init__(self):
        super(GCPAASPPNet, self).__init__()

        self.hardnet = hardnet(arch=68)

        self.fam45 = FAM(640, 256, 256)
        self.fam34 = FAM(320, 256, 256)
        self.fam23 = FAM(128, 256, 256)

        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        inplanes = 1024
        interplanes = 256
        self.conva = nn.Sequential(
            nn.Conv2d(interplanes*5, interplanes, 3, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )

        rates = [1, 3, 6, 9]
        self.aspp1 = Atrous_module(inplanes , interplanes, rate=rates[0])
        self.aspp2 = Atrous_module(inplanes , interplanes, rate=rates[1])
        self.aspp3 = Atrous_module(inplanes , interplanes, rate=rates[2])
        self.aspp4 = Atrous_module(inplanes , interplanes, rate=rates[3])
        self.image_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Conv2d(inplanes , interplanes, kernel_size=1))

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

        x1 = self.aspp1(out5_) # bs, 256, 11, 11
        x2 = self.aspp2(out5_) # bs, 256, 11, 11
        x3 = self.aspp3(out5_) # bs, 256, 11, 11
        x4 = self.aspp4(out5_) # bs, 256, 11, 11
        x5 = self.image_pool(out5_)
        x5 = F.upsample(x5, size=out5_.size()[2:], mode='nearest')

        out5_c = torch.cat((x1, x2, x3, x4, x5), dim=1)

        out5_c = self.conva(out5_c)  # bs, 256, 11, 11


        # GCF
        out4_c = self.local_attention_4(out5_c)  # bs, 256, 11, 11

        out3_c = self.local_attention_3(out5_c)  # bs, 256, 11, 11

        out2_c = self.local_attention_2(out5_c)  # bs, 256, 11, 11

        # HA
        out5 = out5_c  # bs, 256, 11, 11

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
