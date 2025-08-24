import math
import torch
import torch.nn as nn
import numpy as np
import copy

# your DCN import (keeps your original)
from lib.DCNv2.dcn_v2 import DCN
from nets.kaf.fpn import get_fpn
import timm

BN_MOMENTUM = 0.1


class FeatureFusion(nn.Module):
    """Feature fusion module to combine RGB and mask features"""

    def __init__(self, rgb_channels, mask_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.fusion_conv = nn.Conv2d(
            rgb_channels + mask_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.fusion_bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Additional refinement layer
        self.refine_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.refine_bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

    def forward(self, rgb_feat, mask_feat):
        # Concatenate features
        fused = torch.cat([rgb_feat, mask_feat], dim=1)

        # Apply fusion layers
        fused = self.fusion_conv(fused)
        fused = self.fusion_bn(fused)
        fused = self.relu(fused)

        # Refine features
        fused = self.refine_conv(fused)
        fused = self.refine_bn(fused)
        fused = self.relu(fused)

        return fused


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ----------------- NEW: DCNLayer wrapper for heads -----------------
class DCNLayer(nn.Module):
    """
    Simple wrapper: offset conv -> DCN main conv
    Uses DCN class (from lib.DCNv2.dcn_v2 import DCN).
    """

    def __init__(
        self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, deformable_groups=1
    ):
        super(DCNLayer, self).__init__()
        self.k = kernel_size
        # main DCN conv; signature follows your DCN constructor
        # (in_channels, out_channels, kernel_size=(k,k), stride=..., padding=..., deformable_groups=...)
        self.dcn = DCN(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            deformable_groups=deformable_groups,
        )

    def forward(self, x):
        return self.dcn(x)


class KAF_SwinT(nn.Module):
    def __init__(self, head_conv, num_classes, num_rel):
        super().__init__()
        self.num_classes = num_classes
        self.num_rel = num_rel

        swin = timm.create_model(
            "swinv2_tiny_window8_256.ms_in1k", pretrained=True, img_size=512
        )
        # 分离 Swin 组件
        self.patch_embed = swin.patch_embed
        self.stage1 = swin.layers[0]
        self.stage2 = swin.layers[1]
        self.stage3 = swin.layers[2]
        self.stage4 = swin.layers[3]
        self.patch_embed_mask = swin.patch_embed
        self.stage1_mask = copy.deepcopy(swin.layers[0])
        self.feature_fusion = FeatureFusion(96, 96, 96)  # Keep for stage1 fusion
        self.fpn_1 = get_fpn(96)
        self.fpn_2 = get_fpn(96)

        if head_conv > 0:
            # **Replace first conv in each head with DCNLayer**
            # heatmap layers
            self.hmap = nn.Sequential(
                DCNLayer(256, head_conv, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, self.num_classes, kernel_size=1, bias=True),
            )
            self.hmap[-1].bias.data.fill_(-2.19)

            self.raf = nn.Sequential(
                DCNLayer(256, head_conv, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, self.num_rel * 2, kernel_size=1, bias=True),
            )
            self.raf[-1].bias.data.fill_(-2.19)

            # regression layers
            self.regs = nn.Sequential(
                DCNLayer(256, head_conv, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, 2, kernel_size=1, bias=True),
            )
            self.w_h_ = nn.Sequential(
                DCNLayer(256, head_conv, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, 2, kernel_size=1, bias=True),
            )
        else:
            # fallback: keep original plain conv heads if head_conv==0
            self.hmap = nn.Conv2d(64, num_classes, kernel_size=1, bias=True)
            self.raf = nn.Conv2d(64, num_rel * 2, kernel_size=1, bias=True)
            # regression layers
            self.regs = nn.Conv2d(64, 2, kernel_size=1, bias=True)
            self.w_h_ = nn.Conv2d(64, 2, kernel_size=1, bias=True)

        fill_fc_weights(self.regs)
        fill_fc_weights(self.w_h_)

        for m in self.w_h_.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 50.0)

    def forward(self, x):
        # Split input into RGB and mask channels
        rgb = x[:, :3, :, :]  # First 3 channels (RGB)
        mask = x[:, 3:, :, :]  # Last 3 channels (masks)

        # ---- RGB branch stage1 ----
        rgb = self.patch_embed(rgb)
        rgb_feat1 = self.stage1(rgb)

        # ---- Mask branch stage1 ----
        mask = self.patch_embed_mask(mask)
        mask_feat1 = self.stage1_mask(mask)

        rgb_feat1 = rgb_feat1.permute(0, 3, 1, 2).contiguous()
        mask_feat1 = mask_feat1.permute(0, 3, 1, 2).contiguous()
        # ---- Fusion ----
        fused_feat1 = self.feature_fusion(rgb_feat1, mask_feat1)
        fused_feat1 = fused_feat1.permute(0, 2, 3, 1)

        # ---- Shared stage2–4 ----
        feat2 = self.stage2(fused_feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)

        # ---- FPN ----
        feats = [
            fused_feat1.permute(0, 3, 1, 2).contiguous(),
            feat2.permute(0, 3, 1, 2).contiguous(),
            feat3.permute(0, 3, 1, 2).contiguous(),
            feat4.permute(0, 3, 1, 2).contiguous(),
        ]
        p2_1, p3_1, p4_1, p5_1 = self.fpn_1(feats)
        p2_2, p3_2, p4_2, p5_2 = self.fpn_2(feats)

        # ---- Heads ----
        out = [
            [self.hmap(p5_1), self.hmap(p4_1), self.hmap(p3_1), self.hmap(p2_1)],
            [self.regs(p5_1), self.regs(p4_1), self.regs(p3_1), self.regs(p2_1)],
            [self.w_h_(p5_1), self.w_h_(p4_1), self.w_h_(p3_1), self.w_h_(p2_1)],
            [self.raf(p5_2), self.raf(p4_2), self.raf(p3_2), self.raf(p2_2)],
        ]
        return out

    def init_weights(self):
        print("=> init deconv weights from normal distribution")
        for name, m in self.fpn_1.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, m in self.fpn_2.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def get_kaf_swint(head_conv=64, num_classes=13, num_rel=14):
    model = KAF_SwinT(head_conv, num_classes, num_rel)
    model.init_weights()
    return model


if __name__ == "__main__":
    import torch

    def hook(self, input, output):
        try:
            print(output.data.cpu().numpy().shape)
        except Exception:
            pass

    net = get_kaf_swint().cuda()

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, DCN):
            m.register_forward_hook(hook)

    # Example: load pretrained ResNet50 weights into model (will copy matching params)

    with torch.no_grad():
        y = net(torch.randn(4, 6, 512, 512).cuda())
        print("Result dimensions")
        print(type(y[0]))
        for level in range(4):
            print(f"FPN level: {level}")
            print((y[0][level].cpu().numpy()).shape)  # hmap
            print((y[1][level].cpu().numpy()).shape)  # reg
            print((y[2][level].cpu().numpy()).shape)  # wh
            print((y[3][level].cpu().numpy()).shape)  # raf
