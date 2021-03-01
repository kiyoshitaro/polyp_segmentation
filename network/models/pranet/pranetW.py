from .pranet import BasicConv2d, RFB_modified, aggregation
from torch.nn import BatchNorm2d, BatchNorm1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...encoders import res2net50_v1b_26w_4s


class PraNetv8(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(PraNetW, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.new_resnet = res2net50_v1b_26w_4s(pretrained=True)

        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- new Receptive Field Block like module ----
        self.new_rfb2_1 = RFB_modified(512, channel)
        self.new_rfb3_1 = RFB_modified(1024, channel)
        self.new_rfb4_1 = RFB_modified(2048, channel)

        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)
        self.agg2 = aggregation(channel)

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

        # ---- new reverse attention branch 4 ----
        self.new_ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.new_ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.new_ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.new_ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.new_ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- new reverse attention branch 3 ----
        self.new_ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.new_ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.new_ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.new_ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- new reverse attention branch 2 ----
        self.new_ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.new_ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.new_ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.new_ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- new reverse attention branch 1 ----
        self.new_ra1_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.new_ra1_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.new_ra1_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.new_ra1_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- new reverse attention branch 0 ----
        self.new_ra0_conv1 = BasicConv2d(64, 64, kernel_size=1)
        self.new_ra0_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.new_ra0_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.new_ra0_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        x0 = self.resnet.maxpool(x0)  # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x0)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11
        x2_rfb = self.rfb2_1(x2)  # channel -> 32
        x3_rfb = self.rfb3_1(x3)  # channel -> 32
        x4_rfb = self.rfb4_1(x4)  # channel -> 32

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(
            ra5_feat, scale_factor=8, mode='bilinear'
        )  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(
            x, scale_factor=32, mode='bilinear'
        )  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(
            x, scale_factor=16, mode='bilinear'
        )  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(
            x, scale_factor=8, mode='bilinear'
        )  # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- Double branch ----
        new_x = F.interpolate(x, scale_factor=2, mode='bilinear')
        new_x = new_x.expand(-1, 256, -1, -1)  # bs, 256, 88, 88

        # ---- NEW low-level features ----
        new_x2 = self.new_resnet.layer2(new_x)  # bs, 512, 44, 44
        new_x3 = self.new_resnet.layer3(new_x2)  # bs, 1024, 22, 22
        new_x4 = self.new_resnet.layer4(new_x3)  # bs, 2048, 11, 11
        new_x2_rfb = self.new_rfb2_1(new_x2)  # channel -> 32
        new_x3_rfb = self.new_rfb3_1(new_x3)  # channel -> 32
        new_x4_rfb = self.new_rfb4_1(new_x4)  # channel -> 32

        new_ra5_feat = self.agg2(new_x4_rfb, new_x3_rfb, new_x2_rfb)
        new_lateral_map_5 = F.interpolate(
            new_ra5_feat, scale_factor=8, mode='bilinear'
        )  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- NEW reverse attention branch_4 ----
        new_crop_4 = F.interpolate(new_ra5_feat,
                                   scale_factor=0.25,
                                   mode='bilinear')
        new_x = -1 * (torch.sigmoid(new_crop_4)) + 1
        new_x = new_x.expand(-1, 2048, -1, -1).mul(new_x4)
        new_x = self.new_ra4_conv1(new_x)
        new_x = F.relu(self.new_ra4_conv2(new_x))
        new_x = F.relu(self.new_ra4_conv3(new_x))
        new_x = F.relu(self.new_ra4_conv4(new_x))
        new_ra4_feat = self.new_ra4_conv5(new_x)
        new_x = new_ra4_feat + new_crop_4
        new_lateral_map_4 = F.interpolate(
            new_x, scale_factor=32, mode='bilinear'
        )  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- NEW reverse attention branch_3 ----
        new_crop_3 = F.interpolate(new_x, scale_factor=2, mode='bilinear')
        new_x = -1 * (torch.sigmoid(new_crop_3)) + 1
        new_x = new_x.expand(-1, 1024, -1, -1).mul(new_x3)
        new_x = self.new_ra3_conv1(new_x)
        new_x = F.relu(self.new_ra3_conv2(new_x))
        new_x = F.relu(self.new_ra3_conv3(new_x))
        new_ra3_feat = self.new_ra3_conv4(new_x)
        new_x = new_ra3_feat + new_crop_3
        new_lateral_map_3 = F.interpolate(
            new_x, scale_factor=16, mode='bilinear'
        )  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- NEW reverse attention branch_2 ----
        new_crop_2 = F.interpolate(new_x, scale_factor=2, mode='bilinear')
        new_x = -1 * (torch.sigmoid(new_crop_2)) + 1
        new_x = new_x.expand(-1, 512, -1, -1).mul(new_x2)
        new_x = self.new_ra2_conv1(new_x)
        new_x = F.relu(self.new_ra2_conv2(new_x))
        new_x = F.relu(self.new_ra2_conv3(new_x))
        new_ra2_feat = self.new_ra2_conv4(new_x)
        new_x = new_ra2_feat + new_crop_2
        new_lateral_map_2 = F.interpolate(
            new_x, scale_factor=8, mode='bilinear'
        )  # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- NEW reverse attention branch_1 ----
        new_crop_1 = F.interpolate(new_x, scale_factor=2, mode='bilinear')
        new_x = -1 * (torch.sigmoid(new_crop_1)) + 1
        new_x = new_x.expand(-1, 256, -1, -1).mul(x1)
        new_x = self.new_ra1_conv1(new_x)
        new_x = F.relu(self.new_ra1_conv2(new_x))
        new_x = F.relu(self.new_ra1_conv3(new_x))
        new_ra1_feat = self.new_ra1_conv4(new_x)
        new_x = new_ra1_feat + new_crop_1
        new_lateral_map_1 = F.interpolate(
            new_x, scale_factor=4, mode='bilinear'
        )  # NOTES: Sup-5 (bs, 1, 88, 88) -> (bs, 1, 352, 352)

        # ---- NEW reverse attention branch_0 ----
        new_crop_0 = new_x
        new_x = -1 * (torch.sigmoid(new_crop_0)) + 1
        new_x = new_x.expand(-1, 64, -1, -1).mul(x0)
        new_x = self.new_ra0_conv1(new_x)
        new_x = F.relu(self.new_ra0_conv2(new_x))
        new_x = F.relu(self.new_ra0_conv3(new_x))
        new_ra0_feat = self.new_ra0_conv4(new_x)
        new_x = new_ra0_feat + new_crop_0
        new_lateral_map_0 = F.interpolate(
            new_x, scale_factor=4, mode='bilinear'
        )  # NOTES: Sup-6 (bs, 1, 88, 88) -> (bs, 1, 352, 352)

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, new_lateral_map_5, new_lateral_map_4, new_lateral_map_3, new_lateral_map_2, new_lateral_map_1, new_lateral_map_0

    def restore_weights(self, restore_from):
        saved_state_dict = torch.load(restore_from)["model_state_dict"]
        lr = torch.load(restore_from)["lr"]
        self.load_state_dict(saved_state_dict, strict=False)
        return lr


if __name__ == '__main__':
    ras = PraNetv8().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)
