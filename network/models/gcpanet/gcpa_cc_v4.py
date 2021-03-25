from ..pranet.pranet import BasicConv2d, RFB_modified, aggregation
from torch.nn import BatchNorm2d, BatchNorm1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...encoders import res2net50_v1b_26w_4s, hardnet
from ...contextagg import (
    GALDHead,
    GALDBlock,
    SpatialCGNL,
    LocalAttenModule,
    CrissCrossAttention,
)
from .gcpa_gald import FAMPra, CA, SA


class GCPACCv4Net(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(GCPACCv4Net, self).__init__()
        # ---- ResNet Backbone ----
        print("GCPACCv4Net")

        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(320, channel)
        self.rfb3_1 = RFB_modified(640, channel)
        self.rfb4_1 = RFB_modified(1024, channel)

        self.hardnet = hardnet(arch=68)

        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)

        inplanes = 1024
        interplanes = 256

        self.conva = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )

        self.long_relation = CrissCrossAttention(interplanes)
        self.local_attention_4 = LocalAttenModule(interplanes)
        self.local_attention_3 = LocalAttenModule(interplanes)
        self.local_attention_2 = LocalAttenModule(interplanes)

        self.fam45 = FAMPra(640, interplanes, interplanes, interplanes)
        self.fam34 = FAMPra(320, interplanes, interplanes, interplanes)
        self.fam23 = FAMPra(128, interplanes, interplanes, interplanes)

    def forward(self, x):

        hardnetout = self.hardnet(x)

        enc2 = hardnetout[0]  # [24, 128, 88, 88]
        enc3 = hardnetout[1]  # [24, 320, 44, 44]
        enc4 = hardnetout[2]  # [24, 640, 22, 22]
        enc5 = hardnetout[3]  # [24, 1024, 11, 11]

        dec5 = self.conva(enc5)  # bs, 256, 11, 11
        # out4_c = self.long_relation(x4)  # bs, 256, 11, 11
        context = self.long_relation(dec5)  # bs, 256, 11, 11

        # GCF
        context_4 = self.local_attention_4(context)  # bs, 256, 11, 11

        context_3 = self.local_attention_3(context)  # bs, 256, 11, 11

        context_2 = self.local_attention_2(context)  # bs, 256, 11, 11

        enc3_rfb = self.rfb2_1(enc3)  # channel --> 32  [bs, 32, 44, 44]
        enc4_rfb = self.rfb3_1(enc4)  # channel --> 32  [bs, 32, 22, 22]
        enc5_rfb = self.rfb4_1(enc5)  # channel --> 32  [bs, 32, 11, 11]
        ra5_feat = self.agg1(enc5_rfb, enc4_rfb, enc3_rfb)  # [bs, 1, 44, 44]

        # print("ra5_feat",x3_rfb.shape,x4_rfb.shape)
        origin_shape = x.size()[2:]

        lateral_map_5 = F.interpolate(
            ra5_feat, size=origin_shape, mode="bilinear"
        )  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_4 ----
        crop_5 = F.interpolate(ra5_feat, scale_factor=0.5, mode="bilinear")
        dec4, crop_4 = self.fam45(enc4, dec5, context_4, crop_5)  # bs, 256, 22, 22
        lateral_map_4 = F.interpolate(crop_4, size=origin_shape, mode="bilinear")

        crop_4 = F.interpolate(crop_4, scale_factor=2, mode="bilinear")
        dec3, crop_3 = self.fam34(enc3, dec4, context_3, crop_4)  # bs, 256, 44, 44
        lateral_map_3 = F.interpolate(crop_3, size=origin_shape, mode="bilinear")

        crop_3 = F.interpolate(crop_3, scale_factor=2, mode="bilinear")
        dec2, crop_2 = self.fam23(enc2, dec3, context_2, crop_3)  # bs, 256, 88, 88
        lateral_map_2 = F.interpolate(crop_2, size=origin_shape, mode="bilinear")

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2

    def restore_weights(self, restore_from):
        saved_state_dict = torch.load(restore_from)["model_state_dict"]
        lr = torch.load(restore_from)["lr"]
        self.load_state_dict(saved_state_dict, strict=False)
        return lr


if __name__ == "__main__":
    ras = GCPACCv4Net().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)
