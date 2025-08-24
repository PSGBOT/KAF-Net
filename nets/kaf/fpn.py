"""FPN in PyTorch.
From github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
See the paper "Feature Pyramid Networks for Object Detection" for more details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class FPN(nn.Module):
    def __init__(self, size):
        super(FPN, self).__init__()

        # Top layer
        self.top_layer = nn.Conv2d(
            size * 8, 256, kernel_size=1, stride=1, padding=0
        )  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.lat_layer1 = nn.Conv2d(size * 4, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Conv2d(size * 2, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer3 = nn.Conv2d(size, 256, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode="bilinear") + y

    def forward(self, x):
        c2, c3, c4, c5 = x
        # Top-down
        p5 = self.top_layer(c5)
        p4 = self._upsample_add(p5, self.lat_layer1(c4))
        p3 = self._upsample_add(p4, self.lat_layer2(c3))
        p2 = self._upsample_add(p3, self.lat_layer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return [p2, p3, p4, p5]


def get_fpn(size=256):
    return FPN(size)
