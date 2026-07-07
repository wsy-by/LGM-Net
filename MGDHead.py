import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Detect
from ultralytics.nn.modules.block import DFL

__all__ = ('MGDHead', 'MorphologyAttention', 'ProgressiveDecoupling',
           'EqualDecoupling', 'StandardDecoupling')


class MorphologyAttention(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()

        self.color_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.shape_branch = nn.ModuleList([
            nn.Conv2d(
                channels,
                channels // reduction,
                kernel_size=k,
                padding=k // 2,
                groups=channels // reduction,  
                bias=False
            )
            for k in (3, 5)
        ])

        self.shape_fusion = nn.Sequential(
            nn.Conv2d(channels // reduction * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )


        self.morph_weight = nn.Parameter(
            torch.tensor([0.6, 0.4], dtype=torch.float32)
        )

    def forward(self, x):


        color_att = self.color_branch(x)
        shape_feats = [branch(x) for branch in self.shape_branch]
        shape_att = self.shape_fusion(torch.cat(shape_feats, dim=1))
        weights = F.softmax(self.morph_weight, dim=0)
        attention = weights[0] * color_att + weights[1] * shape_att
        return x * (1.0 + attention)


class ProgressiveDecoupling(nn.Module):

    def __init__(self, channels, nc=1, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.cls_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, nc, 1),
        )

        self.reg_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, 4 * reg_max, 1),
        )

    def forward(self, x):
        cls_out = self.cls_path(x)
        reg_out = self.reg_path(x)
        return cls_out, reg_out


class EqualDecoupling(nn.Module):

    def __init__(self, channels, nc=1, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        unified_channels = channels // 2

        self.cls_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, unified_channels, 1, bias=False),
            nn.BatchNorm2d(unified_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(unified_channels, nc, 1),
        )

        self.reg_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, unified_channels, 1, bias=False),
            nn.BatchNorm2d(unified_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(unified_channels, 4 * reg_max, 1),
        )

    def forward(self, x):
        return self.cls_path(x), self.reg_path(x)


class StandardDecoupling(nn.Module):

    def __init__(self, channels, nc=1, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max

        self.cls_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, nc, 1),
        )

        self.reg_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 4 * reg_max, 1),
        )

    def forward(self, x):
        return self.cls_path(x), self.reg_path(x)


class MGDHead(Detect):

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=1, ch=(64, 128, 256), decoupling_type='progressive'):
        nn.Module.__init__(self)

        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = self.nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        if isinstance(decoupling_type, bool):
            decoupling_type = 'progressive' if decoupling_type else 'standard'

        self.decoupling_type = str(decoupling_type).lower()
        print(f"[MGDHead] Initialized with decoupling_type='{self.decoupling_type}'")

        unified_channels = 64

        self.channel_align = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, unified_channels, 1, bias=False),
                nn.BatchNorm2d(unified_channels),
                nn.SiLU(inplace=True),
            )
            for c in ch
        ])

        self.morph_attention = nn.ModuleList([
            MorphologyAttention(unified_channels, reduction=8)
            for _ in range(self.nl)
        ])

        head_mapping = {
            'progressive': ProgressiveDecoupling,
            'equal': EqualDecoupling,
            'standard': StandardDecoupling,
        }
        if self.decoupling_type not in head_mapping:
            raise ValueError(
                f"Unknown decoupling_type: {self.decoupling_type}. "
                f"Expected one of {list(head_mapping.keys())}"
            )

        HeadClass = head_mapping[self.decoupling_type]
        self.decouple_heads = nn.ModuleList([
            HeadClass(unified_channels, nc, self.reg_max)
            for _ in range(self.nl)
        ])

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):

        aligned = [align(feat) for align, feat in zip(self.channel_align, x)]

        enhanced = [attn(feat) for attn, feat in zip(self.morph_attention, aligned)]

        outputs = []
        for feat, head in zip(enhanced, self.decouple_heads):
            cls_out, reg_out = head(feat)
            outputs.append(torch.cat([reg_out, cls_out], dim=1))

        if self.training:
            return outputs

        return self.inference(outputs)

    def inference(self, x):
        from ultralytics.utils.tal import make_anchors, dist2bbox

        b = x[0].shape[0]
        x_cat = torch.cat([xi.view(b, self.no, -1) for xi in x], dim=2)

        if self.dynamic or self.shape != x[0].shape:
            self.anchors, self.strides = (
                t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5)
            )
            self.shape = x[0].shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)

        box = self.dfl(box)

        dbox = dist2bbox(box, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), dim=1)

        return y if self.export else (y, x)

    def bias_init(self):
        for head in self.decouple_heads:
            cls_conv = head.cls_path[-1]
            if hasattr(cls_conv, 'bias') and cls_conv.bias is not None:
                b = cls_conv.bias.view(-1)
                b.data.fill_(-math.log((1 - 0.01) / 0.01))
                cls_conv.bias = nn.Parameter(b.view(-1), requires_grad=True)

            reg_conv = head.reg_path[-1]
            if hasattr(reg_conv, 'bias') and reg_conv.bias is not None:
                b = reg_conv.bias.view(4, self.reg_max)
                b.data[:] = 1.0
                reg_conv.bias = nn.Parameter(b.view(-1), requires_grad=True)