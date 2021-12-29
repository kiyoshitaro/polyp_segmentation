import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...contextagg import (
    LocalAttenModule,
    CrissCrossAttention,
    RCCAModule,
)
from torch.nn import BatchNorm2d, BatchNorm1d
from .gcpa_gald import FAMSCWS
from ...encoders import hardnet
from .gcpanet import ResNet

class SCWSRCCAResNet(nn.Module):
    def __init__(self):
        super(SCWSRCCAResNet, self).__init__()

        self.bkbone = ResNet()

        # self.hardnet = hardnet(arch=68)

        inplanes = 2048
        interplanes = 256

        self.fam45 = FAMSCWS(1024, interplanes, 2048, interplanes)
        self.fam34 = FAMSCWS(512, interplanes, interplanes, interplanes)
        self.fam23 = FAMSCWS(256, interplanes, interplanes, interplanes)

        self.linear5 = nn.Conv2d(2048, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)

        self.conva = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )

        self.long_relation = RCCAModule(inplanes, interplanes)
        self.local_attention_4 = LocalAttenModule(interplanes)
        self.local_attention_3 = LocalAttenModule(interplanes)
        self.local_attention_2 = LocalAttenModule(interplanes)

    def forward(self, x):
        # hardnetout = self.bkbone(x)
        out1, out2, out3, out4, out5 = self.bkbone(x)
        # RCCA
        out5_c = self.long_relation(out5)  # bs, 256, 11, 11

        # GCF
        out4_c = self.local_attention_4(out5_c)  # bs, 256, 11, 11

        out3_c = self.local_attention_3(out5_c)  # bs, 256, 11, 11

        out2_c = self.local_attention_2(out5_c)  # bs, 256, 11, 11

        # Decoder
        out4 = self.fam45(out4, out4_c, out5)
        out3 = self.fam34(out3, out3_c, out4)
        out2 = self.fam23(out2, out2_c, out3)

        # we use bilinear interpolation instead of transpose convolution
        out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode="bilinear")
        out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode="bilinear")
        out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode="bilinear")
        out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode="bilinear")

        return out5, out4, out3, out2
