import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...contextagg import (
    # SpatialCGNL,
    LocalAttenModule,
    PSPModule,
    # SmallLocalAttenModule,
)
from ...encoders import resnet3D50



class LocalAtten3DModule(nn.Module):
    def __init__(self, inplane):
        super(LocalAtten3DModule, self).__init__()
        self.dconv1 = nn.Sequential(
            nn.Conv3d(inplane, inplane, kernel_size=1, groups=inplane, stride=2),
            nn.BatchNorm3d(inplane),
            nn.ReLU(inplace=False),
        )
        self.dconv2 = nn.Sequential(
            nn.Conv3d(inplane, inplane, kernel_size=1, groups=inplane, stride=2),
            nn.BatchNorm3d(inplane),
            nn.ReLU(inplace=False),
        )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=1, groups=inplane, stride=2),
            nn.BatchNorm3d(inplane),
            nn.ReLU(inplace=False),
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, t, h, w = x.size()
        res1 = x
        res2 = x
        print(x.shape,"oooooo")
        x = self.dconv1(x)
        x = self.dconv2(x)
        # x = self.dconv3(x)
        x = F.upsample(x, size=(t, h, w), mode="trilinear", align_corners=True)
        x_mask = self.sigmoid_spatial(x)

        res1 = res1 * x_mask

        return res2 + res1

class PSP3DModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSP3DModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, out_features, size) for size in sizes]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(
                features + len(sizes) * out_features,
                out_features,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_features),
            nn.ReLU(),
            nn.Dropout3d(0.1),
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool3d(output_size=(size, size,size))
        conv = nn.Conv3d(features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm3d(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        t, h, w = feats.size(2), feats.size(3), feats.size(4)
        print(feats.size(),"llllllll")
        # print(self.stages,'kkkk',feats)
        # import sys
        # sys.exit()

        priors = [
            F.interpolate(
                input=stage(feats), size=(t, h, w), mode="trilinear", align_corners=True
            )
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class FAMSCWS(nn.Module):
    def __init__(
        self, in_channel_left, in_channel_down, in_channel_right, interplanes=256
    ):
        super(FAMSCWS, self).__init__()
        # self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv0 = nn.Conv3d(
            in_channel_left, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn0 = nn.BatchNorm3d(interplanes)
        self.conv1 = nn.Conv3d(
            in_channel_down, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm3d(interplanes)
        self.conv2 = nn.Conv3d(
            in_channel_right, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm3d(interplanes)

        self.conv_d1 = nn.Conv3d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.conv_d2 = nn.Conv3d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.conv_l = nn.Conv3d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv3d(
            interplanes, interplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm3d(interplanes)

        self.conv_att1 = nn.Conv3d(interplanes, 1, kernel_size=3, stride=1, padding=1)	
        self.conv_att2 = nn.Conv3d(interplanes, 1, kernel_size=3, stride=1, padding=1)	
        self.conv_att3 = nn.Conv3d(interplanes, 1, kernel_size=3, stride=1, padding=1)	

    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True)  # 256

        down_1 = self.conv_d1(down)

        w1 = self.conv_l(left)
        if down.size()[-3:] != left.size()[-3:]:
            down_ = F.interpolate(down, size=left.size()[-3:], mode="trilinear")
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)  # down is mask

        z1_att = F.adaptive_avg_pool2d(self.conv_att1(z1), (1,1))	
        z1 = z1_att * z1	

        if down_1.size()[-3:] != left.size()[-3:]:
            down_1 = F.interpolate(down_1, size=left.size()[-3:], mode="trilinear")

        z2 = F.relu(down_1 * left, inplace=True)  # left is mask
        z2_att = F.adaptive_avg_pool2d(self.conv_att2(z2), (1,1))	
        z2 = z2_att * z2	

        # z3
        down_2 = self.conv_d2(right)
        if down_2.size()[-3:] != left.size()[-3:]:
            down_2 = F.interpolate(down_2, size=left.size()[-3:], mode="trilinear")
        z3 = F.relu(down_2 * left, inplace=True)  # down_2 is mask

        z3_att = F.adaptive_avg_pool2d(self.conv_att3(z3), (1,1))	
        z3 = z3_att * z3	
        out = (z1 + z2 + z3) / (z1_att + z2_att + z3_att)	
        return F.relu(self.bn3(self.conv3(out)), inplace=True)	

        # out = torch.cat((z1, z2, z3), dim=1)
        # return F.relu(self.bn3(self.conv3(out)), inplace=True)



class SCWSPSPHard3DNet(nn.Module):
    def __init__(self):
        super(SCWSPSPHard3DNet, self).__init__()

        self.resnet3d = resnet3D50()

        inplanes = 512
        interplanes = 256

        self.fam45 = FAMSCWS(256, interplanes, interplanes, interplanes)
        self.fam34 = FAMSCWS(128, interplanes, interplanes, interplanes)
        self.fam23 = FAMSCWS(64, interplanes, interplanes, interplanes)

        self.linear5 = nn.Conv3d(interplanes, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv3d(interplanes, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv3d(interplanes, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv3d(interplanes, 1, kernel_size=3, stride=1, padding=1)

        self.conva = nn.Sequential(
            nn.Conv3d(inplanes, interplanes, 3, padding=1, bias=False),
            nn.BatchNorm3d(interplanes),
            nn.ReLU(interplanes),
        )

        self.long_relation = PSP3DModule(inplanes, interplanes)
        self.local_attention_4 = LocalAtten3DModule(interplanes)
        self.local_attention_3 = LocalAtten3DModule(interplanes)
        self.local_attention_2 = LocalAtten3DModule(interplanes)

    def forward(self, x):
        print(x.size(),"qqqqqqqq")
        out1, out2, out3, out4, out5_ = self.resnet3d(x)
        print(out1.shape,out2.shape, out3.shape,out4.shape,out5_.shape,"ppppp")

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
        out5 = F.interpolate(self.linear5(out5), size=x.size()[-3:], mode="trilinear")
        out4 = F.interpolate(self.linear4(out4), size=x.size()[-3:], mode="trilinear")
        out3 = F.interpolate(self.linear3(out3), size=x.size()[-3:], mode="trilinear")
        out2 = F.interpolate(self.linear2(out2), size=x.size()[-3:], mode="trilinear")

        # out5 = torch.cat((1 - out5, out5), 1)	
        # out4 = torch.cat((1 - out4, out4), 1)	
        # out3 = torch.cat((1 - out3, out3), 1)	
        # out2 = torch.cat((1 - out2, out2), 1)	

        return out5, out4, out3, out2

