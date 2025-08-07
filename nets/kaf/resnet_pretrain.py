"""
ResNet-based model with FPN neck and prediction heads for KAF-Net.
Uses pretrained ResNet backbone from torchvision with custom FPN and heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict


class ResNetBackbone(nn.Module):
    """ResNet backbone that extracts multi-scale features."""

    def __init__(self, arch="resnet50", pretrained=True, input_channels=4):
        super(ResNetBackbone, self).__init__()

        # Load pretrained ResNet
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.feature_channels = [64, 128, 256, 512]
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_channels = [64, 128, 256, 512]
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_channels = [256, 512, 1024, 2048]
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
            self.feature_channels = [256, 512, 1024, 2048]
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=pretrained)
            self.feature_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # Modify first conv layer for 4-channel input (RGBD)
        if input_channels != 3:
            original_conv1 = resnet.conv1
            self.conv1 = nn.Conv2d(
                input_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias,
            )

            # Initialize new conv1 weights
            with torch.no_grad():
                if pretrained:
                    # Copy RGB weights and initialize depth channel
                    self.conv1.weight[:, :3, :, :] = original_conv1.weight
                    self.conv1.weight[:, 3:, :, :] = original_conv1.weight[
                        :, :1, :, :
                    ].mean(dim=1, keepdim=True)
                else:
                    nn.init.kaiming_normal_(
                        self.conv1.weight, mode="fan_out", nonlinearity="relu"
                    )
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # ResNet layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        """Forward pass to extract multi-scale features."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Extract features from different stages
        c2 = self.layer1(x)  # 1/4 resolution
        c3 = self.layer2(c2)  # 1/8 resolution
        c4 = self.layer3(c3)  # 1/16 resolution
        c5 = self.layer4(c4)  # 1/32 resolution

        return [c2, c3, c4, c5]


class CustomFPN(nn.Module):
    """Custom Feature Pyramid Network for multi-scale feature fusion."""

    def __init__(self, in_channels_list, out_channels=256):
        super(CustomFPN, self).__init__()

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        # Lateral convolutions to reduce channel dimensions
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            self.lateral_convs.append(lateral_conv)

        # Output convolutions to smooth features
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.fpn_convs.append(fpn_conv)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize FPN weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """Forward pass for FPN."""
        assert len(inputs) == len(self.in_channels_list)

        # Build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))

        # Build top-down path
        for i in range(len(laterals) - 2, -1, -1):
            prev_shape = laterals[i].shape[-2:]
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=prev_shape, mode="nearest"
            )

        # Build outputs
        outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            outs.append(fpn_conv(laterals[i]))

        return outs


class PredictionHead(nn.Module):
    """Prediction head for specific task (hmap, reg, wh, raf)."""

    def __init__(self, in_channels, out_channels, head_conv=64, bias_init=None):
        super(PredictionHead, self).__init__()

        if head_conv > 0:
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, out_channels, 1, bias=True),
            )
        else:
            self.head = nn.Conv2d(in_channels, out_channels, 1, bias=True)

        # Initialize bias if specified
        if bias_init is not None:
            if head_conv > 0:
                self.head[-1].bias.data.fill_(bias_init)
            else:
                self.head.bias.data.fill_(bias_init)

        # Initialize other weights
        self._init_weights()

    def _init_weights(self):
        """Initialize head weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is None or not hasattr(m.bias, "data"):
                    continue
                # Don't reinitialize bias if it was set in __init__
                if torch.all(m.bias.data == 0):
                    nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        return self.head(x)


class KAFResNetFPN(nn.Module):
    """KAF-Net with ResNet backbone and FPN neck."""

    def __init__(
        self,
        arch="resnet50",
        pretrained=True,
        input_channels=4,
        num_classes=13,
        num_rel=14,
        head_conv=64,
    ):
        super(KAFResNetFPN, self).__init__()

        self.num_classes = num_classes
        self.num_rel = num_rel

        # Backbone
        self.backbone = ResNetBackbone(arch, pretrained, input_channels)

        # FPN necks (two separate FPNs like in original resdcn.py)
        fpn_channels = 256
        self.fpn1 = CustomFPN(self.backbone.feature_channels, fpn_channels)
        self.fpn2 = CustomFPN(self.backbone.feature_channels, fpn_channels)

        # Prediction heads
        # Following the original pattern: hmap on p5_1, reg on p4_1, wh on p3_1, raf on p2_2
        self.hmap_head = PredictionHead(
            fpn_channels, num_classes, head_conv, bias_init=-2.19
        )
        self.reg_head = PredictionHead(fpn_channels, 2, head_conv)
        self.wh_head = PredictionHead(fpn_channels, 2, head_conv)
        self.raf_head = PredictionHead(
            fpn_channels, num_rel * 2, head_conv, bias_init=-2.19
        )

    def forward(self, x):
        """Forward pass."""
        # Extract multi-scale features
        features = self.backbone(x)  # [c2, c3, c4, c5]

        # Apply FPN
        fpn1_features = self.fpn1(features)  # [p2_1, p3_1, p4_1, p5_1]
        fpn2_features = self.fpn2(features)  # [p2_2, p3_2, p4_2, p5_2]

        # Apply prediction heads to appropriate FPN levels
        hmap = self.hmap_head(fpn1_features[3])  # p5_1
        reg = self.reg_head(fpn1_features[2])  # p4_1
        wh = self.wh_head(fpn1_features[1])  # p3_1
        raf = self.raf_head(fpn2_features[0])  # p2_2

        return [[hmap, reg, wh, raf]]


# Factory functions for different ResNet variants
def get_resnet18_fpn(
    pretrained=True, input_channels=4, num_classes=13, num_rel=14, head_conv=64
):
    """Create ResNet-18 with FPN."""
    return KAFResNetFPN(
        "resnet18", pretrained, input_channels, num_classes, num_rel, head_conv
    )


def get_resnet34_fpn(
    pretrained=True, input_channels=4, num_classes=13, num_rel=14, head_conv=64
):
    """Create ResNet-34 with FPN."""
    return KAFResNetFPN(
        "resnet34", pretrained, input_channels, num_classes, num_rel, head_conv
    )


def get_resnet50_fpn(
    pretrained=True, input_channels=4, num_classes=13, num_rel=14, head_conv=64
):
    """Create ResNet-50 with FPN."""
    return KAFResNetFPN(
        "resnet50", pretrained, input_channels, num_classes, num_rel, head_conv
    )


def get_resnet101_fpn(
    pretrained=True, input_channels=4, num_classes=13, num_rel=14, head_conv=64
):
    """Create ResNet-101 with FPN."""
    return KAFResNetFPN(
        "resnet101", pretrained, input_channels, num_classes, num_rel, head_conv
    )


def get_resnet152_fpn(
    pretrained=True, input_channels=4, num_classes=13, num_rel=14, head_conv=64
):
    """Create ResNet-152 with FPN."""
    return KAFResNetFPN(
        "resnet152", pretrained, input_channels, num_classes, num_rel, head_conv
    )


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = get_resnet50_fpn(pretrained=True).to(device)

    # Test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 4, 512, 512).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)

    print("Model output shapes:")
    print(f"hmap: {outputs[0][0].shape}")  # [batch, num_classes, H, W]
    print(f"reg:  {outputs[0][1].shape}")  # [batch, 2, H, W]
    print(f"wh:   {outputs[0][2].shape}")  # [batch, 2, H, W]
    print(f"raf:  {outputs[0][3].shape}")  # [batch, num_rel*2, H, W]

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
