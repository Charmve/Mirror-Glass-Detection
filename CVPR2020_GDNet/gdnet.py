"""
 @Time    : 2020/3/15 22:09
 @Author  : TaylorMei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2020_GDNet
 @File    : gdnet.py
 @Function:
 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.resnext.resnext101_regular import ResNeXt101


###################################################################
# ########################## CBAM #################################
###################################################################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        # original
        # return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        # max
        # torch.max(x, 1)[0].unsqueeze(1)
        # avg
        return torch.mean(x, 1).unsqueeze(1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels=128, reduction_ratio=16, pool_types=['avg'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


###################################################################
# ########################## LCFI #################################
###################################################################
class LCFI(nn.Module):
    def __init__(self, input_channels, dr1=1, dr2=2, dr3=3, dr4=4):
        super(LCFI, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)
        self.channels_double = int(input_channels / 2)
        self.dr1 = dr1
        self.dr2 = dr2
        self.dr3 = dr3
        self.dr4 = dr4
        self.padding1 = 1 * dr1
        self.padding2 = 2 * dr2
        self.padding3 = 3 * dr3
        self.padding4 = 4 * dr4

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1_d1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, (3, 1), 1, padding=(self.padding1, 0),
                      dilation=(self.dr1, 1)),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 3), 1, padding=(0, self.padding1),
                      dilation=(1, self.dr1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_d2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, (1, 3), 1, padding=(0, self.padding1),
                      dilation=(1, self.dr1)),
            nn.Conv2d(self.channels_single, self.channels_single, (3, 1), 1, padding=(self.padding1, 0),
                      dilation=(self.dr1, 1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2_d1 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (5, 1), 1, padding=(self.padding2, 0),
                      dilation=(self.dr2, 1)),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 5), 1, padding=(0, self.padding2),
                      dilation=(1, self.dr2)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_d2 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (1, 5), 1, padding=(0, self.padding2),
                      dilation=(1, self.dr2)),
            nn.Conv2d(self.channels_single, self.channels_single, (5, 1), 1, padding=(self.padding2, 0),
                      dilation=(self.dr2, 1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3_d1 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (7, 1), 1, padding=(self.padding3, 0),
                      dilation=(self.dr3, 1)),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 7), 1, padding=(0, self.padding3),
                      dilation=(1, self.dr3)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_d2 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (1, 7), 1, padding=(0, self.padding3),
                      dilation=(1, self.dr3)),
            nn.Conv2d(self.channels_single, self.channels_single, (7, 1), 1, padding=(self.padding3, 0),
                      dilation=(self.dr3, 1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4_d1 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (9, 1), 1, padding=(self.padding4, 0),
                      dilation=(self.dr4, 1)),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 9), 1, padding=(0, self.padding4),
                      dilation=(1, self.dr4)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_d2 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (1, 9), 1, padding=(0, self.padding4),
                      dilation=(1, self.dr4)),
            nn.Conv2d(self.channels_single, self.channels_single, (9, 1), 1, padding=(self.padding4, 0),
                      dilation=(self.dr4, 1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 3, 1, 1, dilation=1),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.cbam = CBAM(self.input_channels)

        self.channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1_fusion(torch.cat((self.p1_d1(p1_input), self.p1_d2(p1_input)), 1))

        p2_input = torch.cat((self.p2_channel_reduction(x), p1), 1)
        p2 = self.p2_fusion(torch.cat((self.p2_d1(p2_input), self.p2_d2(p2_input)), 1))

        p3_input = torch.cat((self.p3_channel_reduction(x), p2), 1)
        p3 = self.p3_fusion(torch.cat((self.p3_d1(p3_input), self.p3_d2(p3_input)), 1))

        p4_input = torch.cat((self.p4_channel_reduction(x), p3), 1)
        p4 = self.p4_fusion(torch.cat((self.p4_d1(p4_input), self.p4_d2(p4_input)), 1))

        channel_reduction = self.channel_reduction(self.cbam(torch.cat((p1, p2, p3, p4), 1)))

        return channel_reduction


###################################################################
# ########################## NETWORK ##############################
###################################################################
class GDNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(GDNet, self).__init__()
        # params

        # backbone
        resnext = ResNeXt101(backbone_path)
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.h5_conv = LCFI(2048, 1, 2, 3, 4)
        self.h4_conv = LCFI(1024, 1, 2, 3, 4)
        self.h3_conv = LCFI(512, 1, 2, 3, 4)
        self.l2_conv = LCFI(256, 1, 2, 3, 4)

        # h fusion
        self.h5_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.h3_down = nn.AvgPool2d((2, 2), stride=2)
        self.h_fusion = CBAM(896)
        self.h_fusion_conv = nn.Sequential(nn.Conv2d(896, 896, 3, 1, 1), nn.BatchNorm2d(896), nn.ReLU())

        # l fusion
        self.l_fusion_conv = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.h2l = nn.ConvTranspose2d(896, 1, 8, 4, 2)

        # final fusion
        self.h_up_for_final_fusion = nn.ConvTranspose2d(896, 256, 8, 4, 2)
        self.final_fusion = CBAM(320)
        self.final_fusion_conv = nn.Sequential(nn.Conv2d(320, 320, 3, 1, 1), nn.BatchNorm2d(320), nn.ReLU())

        # predict conv
        self.h_predict = nn.Conv2d(896, 1, 3, 1, 1)
        self.l_predict = nn.Conv2d(64, 1, 3, 1, 1)
        self.final_predict = nn.Conv2d(320, 1, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        layer0 = self.layer0(x)  # [-1, 64, h/2, w/2]
        layer1 = self.layer1(layer0)  # [-1, 256, h/4, w/4]
        layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
        layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
        layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]

        h5_conv = self.h5_conv(layer4)
        h4_conv = self.h4_conv(layer3)
        h3_conv = self.h3_conv(layer2)
        l2_conv = self.l2_conv(layer1)

        # h fusion
        h5_up = self.h5_up(h5_conv)
        h3_down = self.h3_down(h3_conv)
        h_fusion = self.h_fusion(torch.cat((h5_up, h4_conv, h3_down), 1))
        h_fusion = self.h_fusion_conv(h_fusion)

        # l fusion
        l_fusion = self.l_fusion_conv(l2_conv)
        h2l = self.h2l(h_fusion)
        l_fusion = F.sigmoid(h2l) * l_fusion

        # final fusion
        h_up_for_final_fusion = self.h_up_for_final_fusion(h_fusion)
        final_fusion = self.final_fusion(torch.cat((h_up_for_final_fusion, l_fusion), 1))
        final_fusion = self.final_fusion_conv(final_fusion)

        # h predict
        h_predict = self.h_predict(h_fusion)

        # l predict
        l_predict = self.l_predict(l_fusion)

        # final predict
        final_predict = self.final_predict(final_fusion)

        # rescale to original size
        h_predict = F.upsample(h_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        l_predict = F.upsample(l_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        final_predict = F.upsample(final_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        return torch.sigmoid(h_predict), torch.sigmoid(l_predict), torch.sigmoid(final_predict)
