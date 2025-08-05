# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-29 12:20:00
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-07-29 13:20:00

import torch
import torch.nn as nn
import spconv.pytorch as spconv

class UnifiedBackbone(nn.Module):
    """
    A 3D U-Net backbone that provides features for both detection and densification.
    """
    def __init__(self, config):
        super(UnifiedBackbone, self).__init__()
        
        # --- Encoder Path ---
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(config['model']['input_channels'], 16, 3, padding=1, bias=False, indice_key='subm0'),
            nn.BatchNorm1d(16), nn.ReLU(True)
        )
        self.conv1 = spconv.SparseSequential(
            spconv.SparseConv3d(16, 32, 3, 2, padding=1, bias=False, indice_key='spconv1'),
            nn.BatchNorm1d(32), nn.ReLU(True)
        )
        self.conv2 = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3, 2, padding=1, bias=False, indice_key='spconv2'),
            nn.BatchNorm1d(64), nn.ReLU(True)
        )
        self.conv3 = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, 3, 2, padding=1, bias=False, indice_key='spconv3'),
            nn.BatchNorm1d(128), nn.ReLU(True)
        )
        self.conv4 = spconv.SparseSequential(
            spconv.SparseConv3d(128, 128, 3, 2, padding=1, bias=False, indice_key='spconv4'),
            nn.BatchNorm1d(128), nn.ReLU(True)
        )
        
        # --- [核心修改] 修正 SparseInverseConv3d 的参数 ---
        # The signature is (in_channels, out_channels, kernel_size, indice_key)
        self.up1 = spconv.SparseInverseConv3d(128, 128, 3, indice_key='spconv4', bias=False)
        self.up2 = spconv.SparseInverseConv3d(128 + 128, 64, 3, indice_key='spconv3', bias=False)
        self.up3 = spconv.SparseInverseConv3d(64 + 64, 32, 3, indice_key='spconv2', bias=False)
        self.up4 = spconv.SparseInverseConv3d(32 + 32, 16, 3, indice_key='spconv1', bias=False)
        
        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(16 + 16, 16, 3, padding=1, bias=False, indice_key='subm0'),
            nn.BatchNorm1d(16),
            nn.ReLU(True)
        )

    def forward(self, sp_tensor):
        # --- Encoder ---
        x0 = self.conv_input(sp_tensor)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3) # Bottleneck features

        # --- Decoder ---
        u1 = self.up1(x4)
        u1 = u1.replace_feature(torch.cat([u1.features, x3.features], dim=1))
        
        u2 = self.up2(u1)
        u2 = u2.replace_feature(torch.cat([u2.features, x2.features], dim=1))
        
        u3 = self.up3(u2)
        u3 = u3.replace_feature(torch.cat([u3.features, x1.features], dim=1))
        
        u4 = self.up4(u3)
        u4 = u4.replace_feature(torch.cat([u4.features, x0.features], dim=1))

        final_3d_features = self.conv_out(u4)
        
        return {
            'bottleneck_features': x4,
            'final_3d_features': final_3d_features
        }