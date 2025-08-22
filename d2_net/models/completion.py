#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: sirope
# @Contact: cfd.xiangchao.wu@gmail.com
# @License: (c)copyright 2018-2028
# @Create Date: 2025/8/11 13:17
# @Last Modified by: sirope
# @Last Modified time: 2025/8/11 13:17
# @File : completion.py
# @Description:

import torch.nn as nn


class CompletionHead(nn.Module):
    """
    一个简单的卷积头，用于从BEV特征中回归出物体的形状编码。
    """

    def __init__(self, in_channels, latent_dim):
        super(CompletionHead, self).__init__()

        shared_channels = 128
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, shared_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(shared_channels),
            nn.ReLU(True),
        )

        self.final_head = nn.Conv2d(shared_channels, latent_dim, kernel_size=1)

    def forward(self, x):
        x = self.shared_conv(x)
        shape_codes = self.final_head(x)
        return shape_codes
