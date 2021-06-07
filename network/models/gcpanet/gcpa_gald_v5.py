import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...contextagg import GALDHead, GALDBlock, SpatialCGNL, LocalAttenModule
from torch.nn import BatchNorm2d, BatchNorm1d
from .gcpa_gald import *
from ...encoders import res2net50_v1b_26w_4s, ResNet


class GCPAGALDNetv5(nn.Module):
    def __init__(self):
        super(GCPAGALDNetv5, self).__init__()
        self.bkbone = ResNet()
        # self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.fam45 = FAM(1024, 256, 256)
        self.fam34 = FAM(512, 256, 256)
        self.fam23 = FAM(256, 256, 256)

        self.srm5 = SRM(256)
        self.srm4 = SRM(256)
        self.srm3 = SRM(256)
        self.srm2 = SRM(256)

        self.ca55 = CA(256, 2048)
        self.sa55 = SA(2048, 2048)

        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        inplanes = 2048
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

        # out1 = self.resnet.conv1(x)
        # out1 = self.resnet.bn1(out1)
        # out1 = self.resnet.relu(out1)
        # out1 = self.resnet.maxpool(out1)  # bs, 64, 88, 88

        # out2 = self.resnet.layer1(out1)  # bs, 256, 88, 88
        # out3 = self.resnet.layer2(out2)  # bs, 512, 44, 44
        # out4 = self.resnet.layer3(out3)  # bs, 1024, 22, 22
        # out5_ = self.resnet.layer4(out4)  # bs, 2048, 11, 11
        out1, out2, out3, out4, out5_ = self.bkbone(x)
        out5_c = self.conva(out5_)  # bs, 256, 11, 11
        out5_c = self.long_relation(out5_c)  # bs, 256, 11, 11

        # GCF
        out4_c = self.local_attention_4(out5_c)  # bs, 256, 11, 11

        out3_c = self.local_attention_3(out5_c)  # bs, 256, 11, 11

        out2_c = self.local_attention_2(out5_c)  # bs, 256, 11, 11

        # HA
        out5_a = self.sa55(out5_, out5_)
        out5 = self.ca55(out5_a, out5_)  # bs, 256, 11, 11

        # out
        out5 = self.srm5(out5)  # bs, 256, 11, 11

        out4 = self.srm4(self.fam45(out4, out5, out4_c))
        out3 = self.srm3(self.fam34(out3, out4, out3_c))
        out2 = self.srm2(self.fam23(out2, out3, out2_c))
        # we use bilinear interpolation instead of transpose convolution
        out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode="bilinear")
        out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode="bilinear")
        out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode="bilinear")
        out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode="bilinear")
        return out5, out4, out3, out2
