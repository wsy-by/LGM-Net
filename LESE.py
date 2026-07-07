import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ('LESE', 'C2fLESE', 'Bottleneck_LESE')


class LESE(nn.Module):

    def __init__(self, channels, reduction=None):
        super().__init__()

        channels = int(channels)


        if reduction is None:
            if channels <= 256:
                reduction = 4
            elif channels <= 512:
                reduction = 8
            else:
                reduction = 16

        mid_channels = max(int(channels // reduction), 16)  


        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
        )


        self.local_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )
            for k in [3, 5]
        ])


        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )


        self.scale = nn.Parameter(torch.ones(1) * 0.1)

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_att = self.global_branch(x)

        local_feats = [branch(x) for branch in self.local_branch]
        local_att = [F.adaptive_avg_pool2d(f, 1) for f in local_feats]

        combined = torch.cat([global_att] + local_att, dim=1)
        att = self.fusion(combined)

        out = x + self.scale * (x * att)

        return out


class Bottleneck_LESE(nn.Module):


    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()


        c1 = int(c1)
        c2 = int(c2)
        c_ = int(c2 * e)
        g = max(1, int(g)) 


        if isinstance(k, (list, tuple)):
            k = tuple(int(x) for x in k)
        else:
            k = (int(k), int(k))

        try:
            from ultralytics.nn.modules.conv import Conv
        except ImportError:
            from .conv import Conv

        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.lese = LESE(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.lese(out)
        return x + out if self.add else out


class C2fLESE(nn.Module):


    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()


        c1 = int(c1)
        c2 = int(c2)
        n = max(1, int(n))  
        g = max(1, int(g))  
        self.c = int(c2 * e)


        try:
            from ultralytics.nn.modules.conv import Conv
        except ImportError:
            from .conv import Conv

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_LESE(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):

        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
