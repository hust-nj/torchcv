#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Mobilenet models.

import torch.nn as nn
from model.backbone.mobilenet.mobilenet_models import MobileNetModels
import torch.nn.functional as F

class MobileNetBackbone(object):
    def __init__(self, configer):
        self.configer = configer
        self.mobile_models = MobileNetModels(self.configer)

    def __call__(self, backbone=None, pretrained=None):
        arch = self.configer.get('network.backbone') if backbone is None else backbone
        pretrained = self.configer.get('network.pretrained') if pretrained is None else pretrained

        if arch == 'mobilenetv2':
            arch_net = self.mobile_models.mobilenetv2(pretrained=pretrained)

        elif arch == 'mobilenetv2_dilated8':
            #arch_net = self.mobile_models.mobilenetv2_dilated8(pretrained=pretrained)
            orig_mobilenet = self.mobile_models.mobilenetv2(pretrained=pretrained)
            arch_net = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)

        elif arch == 'mobilenetv2_dilated16':
            orig_mobilenet = self.mobile_models.mobilenetv2(pretrained=pretrained)
            arch_net = MobileNetV2Dilated(orig_mobilenet, dilate_scale=16)

        elif arch == 'mobilenetv2_fpn':
            orig_mobilenet = self.mobile_models.mobilenetv2(pretrained=pretrained)
            arch_net = MobileNetV2FPN(orig_mobilenet)

        else:
            raise Exception('Architecture undefined!')

        return arch_net


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        print("===================================================")
        print(self.features)
        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        self.features = nn.Sequential(*self.features)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    #m.dilation = (dilate//2, dilate//2)
                    #m.padding = (dilate//2, dilate//2)
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def get_num_features(self):
        return 320

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            f1 = self.features[:15](x)
            f2 = self.features[15:](f1)
            return [f1, f2]

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=False)
    )

def conv_3x3_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=False)
    )


class MobileNetV2FPN(nn.Module):
    def __init__(self, orig_net):
        super(MobileNetV2FPN, self).__init__()
        self.features = orig_net.features[:-1]
        self.total_idx = len(self.features)
        # self.down_idx = [2, 4, 7, 14]
        self.block_idx = [6, 13, 17]
        self.in_channels = [32, 96, 320]
        self.lout_channels = 32
        self.out_channels = 320
        self.block_num = len(self.block_idx)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        # pyramid
        for i in range(self.block_num):
            l_conv = conv_1x1_bn(self.in_channels[i], self.lout_channels)
            # fpn_conv = conv_3x3_bn(self.out_channels, self.out_channels)
            self.lateral_convs.append(l_conv)
            # self.fpn_convs.append(fpn_conv)

        self.out_conv = conv_3x3_bn(self.lout_channels, self.out_channels)

    def get_num_features(self):
        return 320

    def forward(self, x):
        conv_out = []
        dsn_out = None
        for i in range(self.total_idx):
            x = self.features[i](x)
            if i == 14:
                dsn_out = x
            if i in self.block_idx:
                conv_out.append(x)
        laterals = [
            lateral_conv(conv_out[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        for i in range(self.block_num - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape)

        out = self.out_conv(laterals[0])
        return dsn_out, out
