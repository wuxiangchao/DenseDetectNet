# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-07-29 13:10:00

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import numpy as np

from .backbone import UnifiedBackbone
from .bev_backbone import BEVBackbone
from .detection import DetectionHead # 检测头
from .densification import DensificationHead # 导入稠密化头
from .completion import CompletionHead # 补全头
from .shape_autoencoder import PointCloudDecoder # 形状解码器

class D2Net(nn.Module):
    def __init__(self, config):
        super(D2Net, self).__init__()
        self.config = config
        
        # --- 核心模块 ---
        self.unified_backbone = UnifiedBackbone(config)
        self.bev_backbone = BEVBackbone(
            in_channels=config['model']['bev_backbone']['in_channels'],
            layer_channels=config['model']['bev_backbone']['layer_channels'],
            upsample_channels=config['model']['bev_backbone']['upsample_channels'],
            output_channels=config['model']['bev_backbone']['output_channels']
        )
        
        # --- [核心修改] 实例化所有三个任务头 ---
        self.detection_head = DetectionHead(
            in_channels=config['model']['detection_head']['in_channels'],
            tasks_config=config['model']['detection_head']['tasks']
        )
        self.completion_head = CompletionHead(
            in_channels=config['model']['completion_head']['in_channels'],
            latent_dim=config['model']['completion_head']['latent_dim']
        )
        self.densification_head = DensificationHead(
            in_channels=config['model']['densification_head']['in_channels'],
            num_points_to_predict=config['model']['densification_head']['num_points_to_predict']
        )
        
        # --- 加载并冻结预训练的形状解码器 (用于补全) ---
        self.shape_decoder = PointCloudDecoder(
            num_points=config['model']['shape_autoencoder']['num_points'],
            latent_dim=config['model']['shape_autoencoder']['latent_dim']
        )
        decoder_path = config['model']['shape_autoencoder']['decoder_path']
        print(f"Loading pre-trained shape decoder from: {decoder_path}")
        self.shape_decoder.load_state_dict(torch.load(decoder_path))
        self.shape_decoder.eval()
        for param in self.shape_decoder.parameters():
            param.requires_grad = False

        # ... (sparse_shape 计算不变) ...
        pc_range = np.array(config['data']['point_cloud_range'])
        voxel_size = np.array(config['data']['voxel_size'])
        grid_size = (pc_range[3:6] - pc_range[0:3]) / voxel_size
        self.sparse_shape = np.round(grid_size).astype(np.int64)[::-1]

    def forward(self, batch_dict):
        sp_tensor = spconv.SparseConvTensor(
            features=batch_dict['voxels'].float(),
            indices=batch_dict['coordinates'].int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_dict['batch_size']
        )
        
        # 1. 3D U-Net 骨干网
        backbone_out = self.unified_backbone(sp_tensor)
        
        # --- [核心修改] 并行的数据流 ---
        
        # 2a. 稠密化任务流 (使用3D U-Net解码器末端特征)
        final_3d_features = backbone_out['final_3d_features']
        densification_out = self.densification_head(final_3d_features)
        
        # 2b. 检测与补全任务流 (使用3D U-Net瓶颈特征)
        bottleneck_features = backbone_out['bottleneck_features']
        bev_map = bottleneck_features.dense()
        B, C, Z, Y, X = bev_map.shape
        bev_feature_map = bev_map.view(B, C * Z, Y, X)
        enhanced_bev_map = self.bev_backbone(bev_feature_map)
        
        detection_preds = self.detection_head(enhanced_bev_map)
        completion_preds = self.completion_head(enhanced_bev_map)
        
        # 3. 组装输出字典
        output_dict = {}

        # 稠密化输出
        point_offsets = densification_out['point_offsets']
        voxel_size = torch.tensor(self.config['data']['voxel_size'], device=point_offsets.device)
        pc_range_min = torch.tensor(self.config['data']['point_cloud_range'][:3], device=point_offsets.device)
        voxel_indices_xyz = densification_out['sparse_tensor'].indices[:, [3, 2, 1]] 
        voxel_centers_world = voxel_indices_xyz.float() * voxel_size + pc_range_min + voxel_size / 2.0
        predicted_dense_points_world = voxel_centers_world.unsqueeze(1) + point_offsets
        output_dict['predicted_dense_patches_world'] = predicted_dense_points_world
        output_dict['predicted_dense_offsets'] = point_offsets

        # 补全输出
        output_dict['predicted_shape_codes'] = completion_preds.permute(0, 2, 3, 1).contiguous()
        
        # 检测输出
        task_preds_permuted = detection_preds.permute(0, 2, 3, 1).contiguous()
        num_total_classes = sum(len(task['class_names']) for task in self.config['model']['detection_head']['tasks'])
        all_heatmaps_preds = task_preds_permuted[..., :num_total_classes]
        output_dict['box_preds'] = task_preds_permuted[..., num_total_classes:]
        output_dict['heatmaps'] = []
        start_channel = 0
        for task in self.config['model']['detection_head']['tasks']:
            num_task_classes = len(task['class_names'])
            end_channel = start_channel + num_task_classes
            task_heatmap = all_heatmaps_preds[..., start_channel:end_channel]
            output_dict['heatmaps'].append(task_heatmap.permute(0, 3, 1, 2).contiguous())
            start_channel = end_channel

        return output_dict