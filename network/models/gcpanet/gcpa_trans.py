import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...contextagg import (
    # SpatialCGNL,
    LocalAttenModule,
    CrissCrossAttention,
    # SmallLocalAttenModule,
)
from torch.nn import BatchNorm2d, BatchNorm1d
from .gcpa_gald import FAM
from ...encoders import hardnet
from ..transunet import (Transformer, Conv2dReLU)
class GCPATrans(nn.Module):
    def __init__(self, config, img_size=352):
        super(GCPATrans, self).__init__()

        self.hardnet = hardnet(arch=68)
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size,False)
        self.conv_more = Conv2dReLU(
            768, 256,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.config = config
        
        
        inplanes = 1024
        interplanes = 256
        
        self.conva = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )

        self.fam45 = FAM(640, interplanes, interplanes, interplanes)
        self.fam34 = FAM(320, interplanes, interplanes, interplanes)
        self.fam23 = FAM(128, interplanes, interplanes, interplanes)

        self.linear5 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(interplanes, 1, kernel_size=3, stride=1, padding=1)

        self.local_attention_4 = LocalAttenModule(interplanes)
        self.local_attention_3 = LocalAttenModule(interplanes)
        self.local_attention_2 = LocalAttenModule(interplanes)

    def forward(self, x):
        hardnetout = self.hardnet(x)
        transout, attn_weights, features = self.transformer(x)

        out2 = hardnetout[0]  # [24, 128, 88, 88]
        out3 = hardnetout[1]  # [24, 320, 44, 44]
        out4 = hardnetout[2]  # [24, 640, 22, 22]
        out5 = hardnetout[3]  # [24, 1024, 11, 11]
        out5 = self.conva(out5)

        z = transout
        print(z.shape)
        B, n_patch, hidden = z.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        z = z.permute(0, 2, 1)
        z = z.contiguous().view(B, hidden, h, w)
        z = self.conv_more(z)  # bs, 256, 11, 11
        z = F.interpolate(z, scale_factor=0.5, mode='bilinear')
        
        # LD
        out4_c = self.local_attention_4(z)  # bs, 256, 11, 11

        out3_c = self.local_attention_3(z)  # bs, 256, 11, 11

        out2_c = self.local_attention_2(z)  # bs, 256, 11, 11

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
                        
