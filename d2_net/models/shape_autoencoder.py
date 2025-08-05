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
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # Max pooling over points
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, x.shape[1]) # (B, latent_dim)
        return x

class PointCloudDecoder(nn.Module):
    """
    一个简单的MLP解码器，用于从形状编码生成完整点云。
    """
    def __init__(self, num_points=2048, latent_dim=256):
        super(PointCloudDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, num_points * 3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, latent_dim)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        # Reshape to (B, num_points, 3)
        x = x.view(-1, self.num_points, 3)
        return x

class ShapeAutoencoder(nn.Module):
    """
    将编码器和解码器组合成一个完整的自编码器模型。
    """
    def __init__(self, point_dim=3, latent_dim=256, num_points=2048):
        super(ShapeAutoencoder, self).__init__()
        self.encoder = PointNetEncoder(point_dim, latent_dim)
        self.decoder = PointCloudDecoder(num_points, latent_dim)

    def forward(self, x):
        latent_code = self.encoder(x)
        reconstructed_points = self.decoder(latent_code)
        return reconstructed_points, latent_code
