# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-31 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-08-04 12:20:00

import torch
import torch.nn as nn

class PointNetEncoder(nn.Module):
    """
    一个简化的PointNet编码器，用于将点云压缩为形状编码。
    """
    def __init__(self, point_dim=3, latent_dim=256):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(point_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, latent_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, x.shape[1])
        return x

class PointCloudDecoder(nn.Module):
    """
    增加了网络深度和宽度，并增加了输出点的数量。
    """
    def __init__(self, num_points=4096, latent_dim=256): # [改进1] 增加输出点数
        super(PointCloudDecoder, self).__init__()
        self.num_points = num_points
        
        # 增加网络容量
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048) # 增加一层
        self.fc4 = nn.Linear(2048, num_points * 3)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(2048) # 增加一层
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, latent_dim)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x))) # 增加一层
        x = self.fc4(x)
        # Reshape to (B, num_points, 3)
        x = x.view(-1, self.num_points, 3)
        return x

class ShapeAutoencoder(nn.Module):
    """
    将编码器和增强后的简单解码器组合。
    """
    def __init__(self, point_dim=3, latent_dim=256, num_points=4096):
        super(ShapeAutoencoder, self).__init__()
        self.encoder = PointNetEncoder(point_dim, latent_dim)
        self.decoder = PointCloudDecoder(num_points, latent_dim)

    def forward(self, x):
        latent_code = self.encoder(x)
        reconstructed_points = self.decoder(latent_code)
        return reconstructed_points, latent_code
