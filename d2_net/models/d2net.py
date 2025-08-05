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
from .detection import DetectionHead
from .densification import DensificationHead


class D2Net(nn.Module):
    def __init__(self, config):
        super(D2Net, self).__init__()
        self.config = config
        self.unified_backbone = UnifiedBackbone(config)
        self.bev_backbone = BEVBackbone(
            in_channels=config['model']['bev_backbone']['in_channels'],
            layer_channels=config['model']['bev_backbone']['layer_channels'],
            upsample_channels=config['model']['bev_backbone']['upsample_channels'],
            output_channels=config['model']['bev_backbone']['output_channels']
        )
        self.detection_head = DetectionHead(
            in_channels=config['model']['detection_head']['in_channels'],
            tasks_config=config['model']['detection_head']['tasks']
        )
        self.densification_head = DensificationHead(
            in_channels=16,
            num_points_to_predict=config['model']['densification_head']['num_points_to_predict']
        )
        pc_range = np.array(config['data']['point_cloud_range'])
        voxel_size = np.array(config['data']['voxel_size'])
        grid_size = (pc_range[3:6] - pc_range[0:3]) / voxel_size
        self.sparse_shape = np.round(grid_size).astype(np.int64)[::-1]

    def forward(self, batch_dict):
        # (sp_tensor 和 backbone_out 不变)
        sp_tensor = spconv.SparseConvTensor(
            features=batch_dict['voxels'].float(),
            indices=batch_dict['coordinates'].int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_dict['batch_size']
        )
        backbone_out = self.unified_backbone(sp_tensor)
        
        # 1. 稠密化任务
        final_3d_features = backbone_out['final_3d_features']
        densification_out = self.densification_head(final_3d_features)
        point_offsets = densification_out['point_offsets'] # 这是模型预测的局部偏移

        # 2. 检测任务 (不变)
        bottleneck_features = backbone_out['bottleneck_features']
        bev_map = bottleneck_features.dense()
        B, C, Z, Y, X = bev_map.shape
        bev_feature_map = bev_map.view(B, C * Z, Y, X)
        enhanced_bev_map = self.bev_backbone(bev_feature_map)
        detection_preds = self.detection_head(enhanced_bev_map)
        
        # 3. 组装输出字典
        # --- [核心修改] ---
        # 为推理/可视化计算世界坐标下的点云
        voxel_size = torch.tensor(self.config['data']['voxel_size'], device=point_offsets.device)
        pc_range_min = torch.tensor(self.config['data']['point_cloud_range'][:3], device=point_offsets.device)
        voxel_indices_xyz = densification_out['sparse_tensor'].indices[:, [3, 2, 1]] 
        voxel_centers_world = voxel_indices_xyz.float() * voxel_size + pc_range_min + voxel_size / 2.0
        predicted_dense_points_world = voxel_centers_world.unsqueeze(1) + point_offsets

        output_dict = {
            'predicted_point_patches_world': predicted_dense_points_world, # 用于可视化
            'predicted_point_offsets': point_offsets # 用于新的损失计算
        }
        # --------------------
        
        # 检测部分输出
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