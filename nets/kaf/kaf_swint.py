import math
import torch
import torch.nn as nn
import numpy as np
from timm.models import swin_transformer

# your DCN import (keeps your original)
from lib.DCNv2.dcn_v2 import DCN
from nets.kaf.fpn import get_fpn

BN_MOMENTUM = 0.1


class LightweightMaskBackbone(nn.Module):
    """Lightweight backbone for processing mask channels"""

    def __init__(self, in_channels=3, out_channels=256):
        super(LightweightMaskBackbone, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)

        self.conv4 = nn.Conv2d(
            128, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        return x


class FeatureFusion(nn.Module):
    """Feature fusion module to combine Swin and mask features"""

    def __init__(self, swin_channels, mask_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.fusion_conv = nn.Conv2d(
            swin_channels + mask_channels,
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

    def forward(self, swin_feat, mask_feat):
        # Ensure spatial dimensions match
        if swin_feat.shape[2:] != mask_feat.shape[2:]:
            mask_feat = nn.functional.interpolate(
                mask_feat,
                size=swin_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Concatenate features
        fused = torch.cat([swin_feat, mask_feat], dim=1)

        # Apply fusion layers
        fused = self.fusion_conv(fused)
        fused = self.fusion_bn(fused)
        fused = self.relu(fused)

        # Refine features
        fused = self.refine_conv(fused)
        fused = self.refine_bn(fused)
        fused = self.relu(fused)

        return fused


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DCNLayer(nn.Module):
    """
    Simple wrapper: DCN layer for heads
    """

    def __init__(
        self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, deformable_groups=1
    ):
        super(DCNLayer, self).__init__()
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


class KAF_SwinTransformer(nn.Module):
    def __init__(self, swin_model_name, head_conv, num_classes, num_rel):
        super(KAF_SwinTransformer, self).__init__()

        self.num_classes = num_classes
        self.head_conv = head_conv

        # Swin Transformer backbone for RGB
        self.swin_backbone = swin_transformer.__dict__[swin_model_name](
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # Get features from all 4 stages
        )

        # Get the feature dimensions from Swin
        # For swin_tiny_patch4_window7_224: [96, 192, 384, 768]
        # For swin_small_patch4_window7_224: [96, 192, 384, 768]
        # For swin_base_patch4_window7_224: [128, 256, 512, 1024]
        self.swin_dims = self.swin_backbone.feature_info.channels()

        # Mask branch (lightweight backbone)
        self.mask_backbone = LightweightMaskBackbone(in_channels=3, out_channels=256)

        # Feature fusion modules for each stage
        self.fusion_modules = nn.ModuleList(
            [FeatureFusion(self.swin_dims[i], 256, self.swin_dims[i]) for i in range(4)]
        )

        # Channel adjustment layers to make all stages have 256 channels for FPN
        self.channel_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.swin_dims[i], 256, kernel_size=1, bias=False),
                    nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                )
                for i in range(4)
            ]
        )

        # FPN modules
        self.fpn_1 = get_fpn()
        self.fpn_2 = get_fpn()

        if head_conv > 0:
            # DCN-based heads
            self.hmap = nn.Sequential(
                DCNLayer(256, head_conv, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_classes, kernel_size=1, bias=True),
            )
            self.hmap[-1].bias.data.fill_(-2.19)

            self.raf = nn.Sequential(
                DCNLayer(256, head_conv, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_rel * 2, kernel_size=1, bias=True),
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
            # fallback: plain conv heads
            self.hmap = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
            self.raf = nn.Conv2d(256, num_rel * 2, kernel_size=1, bias=True)
            self.regs = nn.Conv2d(256, 2, kernel_size=1, bias=True)
            self.w_h_ = nn.Conv2d(256, 2, kernel_size=1, bias=True)

        fill_fc_weights(self.regs)
        fill_fc_weights(self.w_h_)

        for m in self.w_h_.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 50.0)

    def forward(self, x):
        # Split input into RGB and mask channels
        rgb_input = x[:, :3, :, :]  # First 3 channels (RGB)
        mask_input = x[:, 3:, :, :]  # Last 3 channels (masks)

        # Extract features using Swin Transformer
        swin_features = self.swin_backbone(rgb_input)

        # Process mask features
        mask_feat = self.mask_backbone(mask_input)

        # Fuse Swin features with mask features at each stage
        fused_features = []
        for i, swin_feat in enumerate(swin_features):
            # Resize mask features to match Swin feature spatial dimensions
            if mask_feat.shape[2:] != swin_feat.shape[2:]:
                resized_mask_feat = nn.functional.interpolate(
                    mask_feat,
                    size=swin_feat.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                resized_mask_feat = mask_feat

            fused_feat = self.fusion_modules[i](swin_feat, resized_mask_feat)

            # Adapt channels to 256 for FPN
            adapted_feat = self.channel_adapters[i](fused_feat)
            fused_features.append(adapted_feat)

        # Apply FPN
        p2_1, p3_1, p4_1, p5_1 = self.fpn_1(fused_features)
        p2_2, p3_2, p4_2, p5_2 = self.fpn_2(fused_features)

        # Generate outputs
        out = [
            [self.hmap(p5_1), self.hmap(p4_1), self.hmap(p3_1), self.hmap(p2_1)],
            [self.regs(p5_1), self.regs(p4_1), self.regs(p3_1), self.regs(p2_1)],
            [self.w_h_(p5_1), self.w_h_(p4_1), self.w_h_(p3_1), self.w_h_(p2_1)],
            [self.raf(p5_2), self.raf(p4_2), self.raf(p3_2), self.raf(p2_2)],
        ]
        return out

    def init_weights(self):
        """Initialize weights for custom layers"""
        print("=> init weights from normal distribution")
        for name, m in self.fpn_1.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, m in self.fpn_2.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def get_kaf_swin_tiny(head_conv=64, num_classes=13, num_rel=14):
    """Get KAF model with Swin-Tiny backbone"""
    model = KAF_SwinTransformer(
        swin_model_name="swin_tiny_patch4_window7_224",
        head_conv=head_conv,
        num_classes=num_classes,
        num_rel=num_rel,
    )
    model.init_weights()
    return model


def get_kaf_swin_small(head_conv=64, num_classes=13, num_rel=14):
    """Get KAF model with Swin-Small backbone"""
    model = KAF_SwinTransformer(
        swin_model_name="swin_small_patch4_window7_224",
        head_conv=head_conv,
        num_classes=num_classes,
        num_rel=num_rel,
    )
    model.init_weights()
    return model


def get_kaf_swin_base(head_conv=64, num_classes=13, num_rel=14):
    """Get KAF model with Swin-Base backbone"""
    model = KAF_SwinTransformer(
        swin_model_name="swin_base_patch4_window7_224",
        head_conv=head_conv,
        num_classes=num_classes,
        num_rel=num_rel,
    )
    model.init_weights()
    return model


if __name__ == "__main__":
    import torch

    def hook(self, input, output):
        try:
            print(output.data.cpu().numpy().shape)
        except Exception:
            pass

    # Test with Swin-Tiny
    net = get_kaf_swin_tiny().cuda()

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, DCN):
            m.register_forward_hook(hook)

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
