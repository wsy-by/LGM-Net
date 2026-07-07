import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv, autopad


class DySample(nn.Module):

    def __init__(self, in_channels, scale=2, style='lp', groups=4):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups

        if style == 'lp':  
            self.offset = nn.Conv2d(in_channels, 2 * groups * scale * scale, 1)
            self.scope = nn.Conv2d(in_channels, groups * scale * scale, 1)
            nn.init.constant_(self.offset.weight, 0)
            nn.init.constant_(self.offset.bias, 0)
            nn.init.constant_(self.scope.weight, 0)
            nn.init.constant_(self.scope.bias, 0)
        elif style == 'pl':  
            self.pixel_weight = nn.Conv2d(in_channels, groups * scale * scale, 1)
            nn.init.constant_(self.pixel_weight.weight, 0)
            nn.init.constant_(self.pixel_weight.bias, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.style == 'lp':
            offset = self.offset(x)
            scope = torch.sigmoid(self.scope(x))

            offset = offset.view(B, self.groups, 2, self.scale, self.scale, H, W)
            scope = scope.view(B, self.groups, 1, self.scale, self.scale, H, W)


            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, H * self.scale, device=x.device),
                torch.arange(0, W * self.scale, device=x.device),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=0).float()
            grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

            x = F.interpolate(x, scale_factor=self.scale, mode='nearest')

        else:  
            weight = self.pixel_weight(x)
            weight = weight.view(B, self.groups, self.scale * self.scale, H, W)
            weight = F.softmax(weight, dim=2)

            x = F.interpolate(x, scale_factor=self.scale, mode='nearest')

        return x


class SpatialGate(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)


class GSSPA(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels

        self.channel_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_branch = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(k, stride=1, padding=k // 2) if k > 1 else nn.Identity(),
                nn.Conv2d(channels, channels // reduction, 1),
                nn.BatchNorm2d(channels // reduction),
                nn.SiLU(inplace=True)
            ) for k in [1, 3, 5, 7]  
        ])

        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(channels // reduction * 4, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 3, padding=1),
            nn.Sigmoid()
        )


        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, 2, 1),             nn.Softmax(dim=1)
        )

    def forward(self, x):


        channel_att = self.channel_branch(x)


        spatial_feats = [branch(x) for branch in self.spatial_branch]
        spatial_feat = torch.cat(spatial_feats, dim=1)
        spatial_att = self.spatial_fusion(spatial_feat)

        gate_weights = self.gate(x)  
        w_c = gate_weights[:, 0:1, :, :]  
        w_s = gate_weights[:, 1:2, :, :]  

        out = x * (w_c * channel_att + w_s * spatial_att)
        return out


class MiniSPP(nn.Module):


    def __init__(self, channels, pool_sizes=(3, 5, 7)):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.MaxPool2d(k, stride=1, padding=k // 2) for k in pool_sizes
        ])
        self.conv = Conv(channels * (len(pool_sizes) + 1), channels, 1)

    def forward(self, x):
        pooled = [x] + [pool(x) for pool in self.pools]
        return self.conv(torch.cat(pooled, dim=1))


class GSSPAN(nn.Module):


    def __init__(self, in_channels_list, out_channels=64, use_dysample=True):

        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.use_dysample = use_dysample


        self.up_p5_to_p4 = DySample(in_channels_list[2], scale=2) if use_dysample else nn.Upsample(scale_factor=2,
                                                                                                   mode='nearest')
        self.gsspa_p4 = GSSPA(in_channels_list[1] + in_channels_list[2])
        self.reduce_p4 = Conv(in_channels_list[1] + in_channels_list[2], out_channels, 1)


        self.up_p4_to_p3 = DySample(out_channels, scale=2) if use_dysample else nn.Upsample(scale_factor=2,
                                                                                            mode='nearest')
        self.gsspa_p3 = GSSPA(in_channels_list[0] + out_channels)
        self.reduce_p3 = Conv(in_channels_list[0] + out_channels, out_channels, 1)


        self.down_p3_to_p4 = Conv(out_channels, out_channels, 3, 2)
        self.gsspa_p4_down = GSSPA(out_channels * 2)
        self.reduce_p4_down = Conv(out_channels * 2, out_channels, 1)


        self.down_p4_to_p5 = Conv(out_channels, out_channels, 3, 2)
        self.gsspa_p5 = GSSPA(out_channels * 2)
        self.reduce_p5 = Conv(out_channels * 2, out_channels, 1)


        self.spp_p3 = MiniSPP(out_channels)
        self.spp_p4 = MiniSPP(out_channels)
        self.spp_p5 = MiniSPP(out_channels)

    def forward(self, x):

        p3, p4, p5 = x

        p5_up = self.up_p5_to_p4(p5)
        p4_in = torch.cat([p4, p5_up], dim=1)
        p4_in = self.gsspa_p4(p4_in)
        p4_out = self.reduce_p4(p4_in)

        p4_up = self.up_p4_to_p3(p4_out)
        p3_in = torch.cat([p3, p4_up], dim=1)
        p3_in = self.gsspa_p3(p3_in)
        p3_out = self.reduce_p3(p3_in)

        p3_down = self.down_p3_to_p4(p3_out)
        p4_in2 = torch.cat([p4_out, p3_down], dim=1)
        p4_in2 = self.gsspa_p4_down(p4_in2)
        p4_out_final = self.reduce_p4_down(p4_in2)

        p4_down = self.down_p4_to_p5(p4_out_final)
        p5_in = torch.cat([self.reduce_p5(torch.cat([p5, torch.zeros_like(p5)], dim=1)), p4_down], dim=1)
        p5_in = self.gsspa_p5(p5_in)
        p5_out_final = self.reduce_p5(p5_in)

        p3_final = self.spp_p3(p3_out)
        p4_final = self.spp_p4(p4_out_final)
        p5_final = self.spp_p5(p5_out_final)

        return [p3_final, p4_final, p5_final]



class GSSPANSimple(nn.Module):

    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.gsspa = nn.ModuleList([
            GSSPA(c) for c in in_channels_list
        ])
        self.reduce = nn.ModuleList([
            Conv(c, out_channels, 1) for c in in_channels_list
        ])

    def forward(self, x):
        return [self.reduce[i](self.gsspa[i](feat)) for i, feat in enumerate(x)]
