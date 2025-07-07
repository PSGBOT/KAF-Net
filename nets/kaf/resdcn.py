import math

import torch.nn as nn
import numpy as np

from lib.DCNv2.dcn_v2 import DCN
from .fpn import get_fpn

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ConvBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ConvBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DeformBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(DeformBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        # self.conv2 = nn.Conv2d(
        #     out_channels,
        #     out_channels,
        #     kernel_size=3,
        #     stride=stride,
        #     padding=1,
        #     bias=False,
        # )
        self.dcn2 = DCN(
            out_channels,  # Changed from in_channels to out_channels
            out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
            dilation=1,
            deformable_groups=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dcn2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class KAF_ResDCN(nn.Module):
    def __init__(self, blocks, num_layers, head_conv, num_classes, num_rel):
        self.in_channel = 64
        self.deconv_with_bias = False
        self.num_classes = num_classes

        super(KAF_ResDCN, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[0], 64, num_layers[0])
        self.layer2 = self._make_layer(blocks[0], 128, num_layers[1], stride=2)
        self.layer3 = self._make_layer(blocks[1], 256, num_layers[2], stride=2)
        self.layer4 = self._make_layer(blocks[1], 512, num_layers[3], stride=2)
        self.fpn_1 = get_fpn()
        self.fpn_2 = get_fpn()

        if head_conv > 0:
            # heatmap layers
            self.hmap = nn.Sequential(
                nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_classes, kernel_size=1, bias=True),
            )
            self.hmap[-1].bias.data.fill_(-2.19)
            self.raf = nn.Sequential(
                nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_rel * 2, kernel_size=1, bias=True),
            )
            self.raf[-1].bias.data.fill_(-2.19)
            # regression layers
            self.regs = nn.Sequential(
                nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, 2, kernel_size=1, bias=True),
            )
            self.w_h_ = nn.Sequential(
                nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, 2, kernel_size=1, bias=True),
            )
        else:
            # heatmap layers
            self.hmap = nn.Conv2d(64, num_classes, kernel_size=1, bias=True)
            self.raf = nn.Conv2d(64, num_rel * 2, kernel_size=1, bias=True)
            # regression layers
            self.regs = nn.Conv2d(64, 2, kernel_size=1, bias=True)
            self.w_h_ = nn.Conv2d(64, 2, kernel_size=1, bias=True)

        fill_fc_weights(self.regs)
        fill_fc_weights(self.w_h_)

    def _make_layer(self, block, out_channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    out_channel * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channel * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channel, out_channel))
        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        padding = 0
        output_padding = 0
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        p2_1, p3_1, p4_1, p5_1 = self.fpn_1([out1, out2, out3, out4])
        p2_2, p3_2, p4_2, p5_2 = self.fpn_2([out1, out2, out3, out4])
        out = [self.hmap(p5_1), self.regs(p4_1), self.w_h_(p3_1), self.raf(p2_2)]
        return [out]

    def init_weights(self, num_layers):
        print("=> init deconv weights from normal distribution")
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: ([BasicBlock], [2, 2, 2, 2]),
    34: ([BasicBlock], [3, 4, 6, 3]),
    50: ([ConvBottleneck], [3, 4, 6, 3]),
    101: ([ConvBottleneck], [3, 4, 23, 3]),
    152: ([ConvBottleneck], [3, 8, 36, 3]),
}

resdcn_spec = {
    50: ([ConvBottleneck, DeformBottleneck], [3, 4, 6, 3]),
    101: ([ConvBottleneck, DeformBottleneck], [3, 4, 23, 3]),
    152: ([ConvBottleneck, DeformBottleneck], [3, 8, 36, 3]),
}


def get_kaf_resnet(num_layers, head_conv=64, num_classes=13, num_rel=14):
    block_classes, layers = resnet_spec[num_layers]
    model = KAF_ResDCN(block_classes, layers, head_conv, num_classes, num_rel)
    # model.init_weights(num_layers) # Commented out as deconv_layers is not defined
    return model


def get_kaf_resdcn(num_layers, head_conv=64, num_classes=13, num_rel=14):
    block_classes, layers = resdcn_spec[num_layers]
    model = KAF_ResDCN(block_classes, layers, head_conv, num_classes, num_rel)
    # model.init_weights(num_layers) # Already commented out, keeping it that way
    return model


if __name__ == "__main__":
    import torch

    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)
        # pass

    net = get_kaf_resdcn(50).cuda()

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, DCN):
            m.register_forward_hook(hook)

    with torch.no_grad():
        y = net(torch.randn(4, 4, 512, 512).cuda())
        print("Result dimensions")
        print(type(y[0]))
        print((y[0][0].cpu().numpy()).shape)  # hmap [2,80,128,128]
        print((y[0][1].cpu().numpy()).shape)  # reg [2,2,128,128]
        print((y[0][2].cpu().numpy()).shape)  # wh [2,2,128,128]
        print((y[0][3].cpu().numpy()).shape)  # raf [2,14,128,128]
