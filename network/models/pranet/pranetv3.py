from .pranet import BasicConv2d, RFB_modified, aggregation
from torch.nn import BatchNorm2d, BatchNorm1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...encoders import res2net50_v1b_26w_4s
from ...contextagg import DualGCNHead, DualGCN


class PraNetv3(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(PraNetv3, self).__init__()
        # ---- ResNet Backbone ----
        print("PraNetv3")

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.head = DualGCNHead(2048, 512, 1)
        self.dualgcn_2 = DualGCN(64)
        self.dualgcn_3 = DualGCN(64)
        self.dualgcn_4 = DualGCN(256)

        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.bottleneck_2 = nn.Sequential(
            nn.Conv2d(512 + 64, 64, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(64),
            nn.ReLU(64),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.bottleneck_3 = nn.Sequential(
            nn.Conv2d(1024 + 64, 64, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(64),
            nn.ReLU(64),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.bottleneck_4 = nn.Sequential(
            nn.Conv2d(
                2048 + 256, 256, kernel_size=3, padding=1, dilation=1, bias=False
            ),
            BatchNorm2d(256),
            nn.ReLU(256),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11
        # print("x1",x1.shape,"x2",x2.shape,"x3",x3.shape,"x4",x4.shape)

        # ra5_feat = self.head(x4)

        x2_rfb = self.rfb2_1(x2)  # channel --> 32  [bs, 32, 44, 44]
        x3_rfb = self.rfb3_1(x3)  # channel --> 32  [bs, 32, 22, 22]
        x4_rfb = self.rfb4_1(x4)  # channel --> 32  [bs, 32, 11, 11]
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)  # Sg = [bs, 1, 44, 44]

        # print("ra5_feat",x3_rfb.shape,x4_rfb.shape)

        lateral_map_5 = F.interpolate(
            ra5_feat, scale_factor=8, mode="bilinear"
        )  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # lateral_map_5 = F.upsample(input=ra5_feat, size=(352,352), mode='bilinear', align_corners=True)
        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode="bilinear")
        # print(crop_4,"crop_4")
        x = -1 * (torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1).mul(x4)

        x_ = self.ra4_conv1(x)
        x_ = self.dualgcn_4(x_)

        x_ = F.relu(self.ra4_conv2(x_))
        # print("ssss",x_.shape)
        x = self.bottleneck_4(torch.cat([x, x_], 1))  # [bs, 64, 45, 45]

        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)  # R5
        x = ra4_feat + crop_4  # S5 = (bs, 1, 11, 11)

        lateral_map_4 = F.interpolate(
            x, scale_factor=32, mode="bilinear"
        )  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode="bilinear")  # (bs, 1, 22, 22)
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)

        x_ = self.ra3_conv1(x)
        x_ = self.dualgcn_3(x_)
        x_ = F.relu(self.ra3_conv2(x_))
        # print("iiiii",x_.shape)
        x = self.bottleneck_3(torch.cat([x, x_], 1))  # [bs, 64, 45, 45]

        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)  # R4
        x = ra3_feat + crop_3  # S4 = (bs, 1, 22, 22)

        lateral_map_3 = F.interpolate(
            x, scale_factor=16, mode="bilinear"
        )  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode="bilinear")  # (bs, 1, 44, 44)
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x_ = self.ra2_conv1(x)
        x_ = self.dualgcn_2(x_)
        x_ = F.relu(self.ra2_conv2(x_))
        x = self.bottleneck_2(torch.cat([x, x_], 1))  # [bs, 64, 45, 45]
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)  # R3
        x = ra2_feat + crop_2  # S3 = (bs, 1, 44, 44)

        lateral_map_2 = F.interpolate(
            x, scale_factor=8, mode="bilinear"
        )  # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2

    def restore_weights(self, restore_from):
        saved_state_dict = torch.load(restore_from)["model_state_dict"]
        lr = torch.load(restore_from)["lr"]
        self.load_state_dict(saved_state_dict, strict=False)
        return lr


if __name__ == "__main__":
    ras = PraNetv3().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)
