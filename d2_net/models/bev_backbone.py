# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-29 12:20:00
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-07-29 12:55:00

import torch
import torch.nn as nn
import torch.nn.functional as F # 导入 functional

class BEVBackbone(nn.Module):
    """
    A simple Feature Pyramid Network (FPN) to process the BEV feature map.
    """
    def __init__(self, in_channels, layer_channels, upsample_channels, output_channels):
        super(BEVBackbone, self).__init__()
        
        self.in_channels = in_channels
        self.output_channels = output_channels

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Create downsampling blocks
        in_c = in_channels
        for out_c in layer_channels:
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True),
            ))
            in_c = out_c

        # Create upsampling blocks
        # The number of upsample blocks should match the number of downsample blocks
        for i in range(len(layer_channels)):
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    layer_channels[i], upsample_channels[i], kernel_size=2, stride=2, bias=False
                ),
                nn.BatchNorm2d(upsample_channels[i]),
                nn.ReLU(True),
            ))

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # Downsampling path
        down_features = []
        for block in self.down_blocks:
            x = block(x)
            down_features.append(x)
        
        # 修正FPN的上采样和融合逻辑
        # Upsampling and fusion path
        up_features = []
        for i, block in enumerate(self.up_blocks):
            up_features.append(block(down_features[i]))
            
        # At this point, up_features contains tensors of different spatial resolutions.
        # e.g., [(B, 128, 200, 176), (B, 128, 100, 88)]
        # We must resize them all to match the largest one before concatenation.
        
        # Get the target shape from the first (largest) upsampled feature map
        target_shape = up_features[0].shape[2:]
        
        # Resize all other feature maps to the target shape
        resized_up_features = [up_features[0]]
        for i in range(1, len(up_features)):
            resized_feature = F.interpolate(
                up_features[i], 
                size=target_shape, 
                mode='bilinear', 
                align_corners=False
            )
            resized_up_features.append(resized_feature)
            
        # Concatenate all upsampled features, which now have the same H and W
        final_features = torch.cat(resized_up_features, dim=1)
        
        return final_features