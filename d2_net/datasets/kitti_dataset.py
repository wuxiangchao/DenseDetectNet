# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-07-29 13:20:00

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from ..utils.pointcloud_utils import VoxelGenerator
from ..utils.kitti_utils import Calibration 

USE_MINI_DATASET = False 

class KittiDataset(Dataset):
    def __init__(self, config, split='train'):
        self.root_path = config['data']['root_path']
        self.config = config
        self.split = split
        self.class_names = config['data']['class_names']
        
        split_to_load = split
        if split == 'train' and USE_MINI_DATASET:
            print("--- INFO: Using MINI dataset for training for faster iteration! ---")
            split_to_load = 'train'

        split_file_path = os.path.join(self.root_path, 'ImageSets', f'{split_to_load}.txt')
        if not os.path.exists(split_file_path):
             split_file_path = os.path.join(self.root_path, 'PointCloudSets', f'{split_to_load}.txt')

        self.sample_ids = [x.strip() for x in open(split_file_path).readlines()] if os.path.exists(split_file_path) else []

        self.voxel_generator = VoxelGenerator(
            voxel_size=config['data']['voxel_size'],
            point_cloud_range=config['data']['point_cloud_range'],
            max_num_points=config['data']['max_points_per_voxel'],
            # Note: We now pass max_voxels for both splits
            max_voxels=config['data']['max_number_of_voxels'] if split == 'train' else config['data']['test_max_number_of_voxels']
        )
        self.max_voxels = self.voxel_generator.max_voxels
        self.grid_size = self.voxel_generator.grid_size
        downsample_factor = 16
        self.feature_map_size = [
            (self.grid_size[1] + downsample_factor - 1) // downsample_factor,
            (self.grid_size[0] + downsample_factor - 1) // downsample_factor
        ]

    # ... (__len__, _load_velodyne, _load_calib, _load_labels 不变) ...
    def __len__(self):
        return len(self.sample_ids)
    def _load_velodyne(self, sample_id):
        path = os.path.join(self.root_path, 'training', 'velodyne', f'{sample_id}.bin')
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    def _load_calib(self, sample_id):
        path = os.path.join(self.root_path, 'training', 'calib', f'{sample_id}.txt')
        return Calibration(path)
    def _load_labels(self, sample_id):
        path = os.path.join(self.root_path, 'training', 'label_2', f'{sample_id}.txt')
        annotations = []
        with open(path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                obj_type = parts[0]
                if obj_type not in self.class_names: continue
                loc = np.array([float(parts[11]), float(parts[12]), float(parts[13])], dtype=np.float32)
                dims = np.array([float(parts[10]), float(parts[8]), float(parts[9])], dtype=np.float32)
                ry = float(parts[14])
                annotations.append({'name': obj_type, 'location_cam': loc, 'dimensions': dims, 'rotation_y': ry})
        return annotations

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        points = self._load_velodyne(sample_id)
        annotations = self._load_labels(sample_id)
        calib = self._load_calib(sample_id)

        for ann in annotations:
            loc_cam = ann['location_cam'].reshape(1, 3)
            loc_velo = calib.project_rect_to_velo(loc_cam)
            ann['location'] = loc_velo.flatten()
            ann['yaw'] = -(ann['rotation_y'] + np.pi / 2.0)
            
        # 1. 生成所有体素
        voxel_data = self.voxel_generator.generate(points)
        num_voxels = voxel_data['coordinates'].shape[0]

        # 2. 生成检测目标
        detection_targets = self.generate_detection_targets(annotations)

        # 3. [核心修改] 如果体素数量超限，则执行采样
        if num_voxels > self.max_voxels:
            if self.split == 'train':
                # 训练时使用优先采样
                final_indices = self.get_priority_indices(voxel_data, detection_targets)
            else:
                # 验证/测试时使用简单的随机采样
                final_indices = np.random.choice(num_voxels, self.max_voxels, replace=False)
            
            # 根据采样索引来选择数据
            for key in ['voxels', 'coordinates', 'gt_points', 'gt_points_mask']:
                voxel_data[key] = voxel_data[key][final_indices]
        
        # 4. 归一化稠密化目标
        voxel_coords = voxel_data['coordinates']
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        voxel_centers = (voxel_coords[:, [2, 1, 0]] + 0.5) * voxel_size + pc_range[:3]
        
        gt_points_world = voxel_data['gt_points']
        gt_points_local = gt_points_world - voxel_centers[:, np.newaxis, :]
        voxel_data['gt_points'] = gt_points_local

        # 5. 准备最终的输入和输出字典
        input_dict = {'voxels': voxel_data['voxels'], 'coordinates': voxel_data['coordinates']}
        target_dict = {'gt_points': voxel_data['gt_points'], 'gt_points_mask': voxel_data['gt_points_mask'], **detection_targets}
        return input_dict, target_dict

    def get_priority_indices(self, voxel_data, detection_targets):
        """
        执行优先采样的辅助函数
        """
        pos_mask = detection_targets['pos_mask']
        voxel_coords = voxel_data['coordinates']
        
        downsample_ratio = 16
        map_coords = voxel_coords[:, 1:] // downsample_ratio
        map_coords[:, 0] = np.clip(map_coords[:, 0], 0, self.feature_map_size[0] - 1)
        map_coords[:, 1] = np.clip(map_coords[:, 1], 0, self.feature_map_size[1] - 1)
        
        is_positive = pos_mask[map_coords[:, 0], map_coords[:, 1]]
        
        pos_indices = np.where(is_positive)[0]
        neg_indices = np.where(~is_positive)[0]
        
        num_pos = len(pos_indices)
        num_neg_to_sample = self.max_voxels - num_pos
        
        if num_neg_to_sample > 0 and len(neg_indices) > 0:
            if len(neg_indices) > num_neg_to_sample:
                sampled_neg_indices = np.random.choice(neg_indices, num_neg_to_sample, replace=False)
            else:
                sampled_neg_indices = neg_indices
            final_indices = np.concatenate([pos_indices, sampled_neg_indices])
        else:
            final_indices = pos_indices

        if len(final_indices) > self.max_voxels:
            final_indices = np.random.choice(final_indices, self.max_voxels, replace=False)
            
        return final_indices

    def generate_detection_targets(self, annotations):
        # ... (This function does not need changes) ...
        downsample_ratio = 16
        heatmaps = [np.zeros((len(task['class_names']), *self.feature_map_size), dtype=np.float32) for task in self.config['model']['detection_head']['tasks']]
        box_preds = np.zeros((*self.feature_map_size, 8), dtype=np.float32)
        pos_mask = np.zeros(self.feature_map_size, dtype=np.bool_)
        pc_range = self.config['data']['point_cloud_range']
        voxel_size = self.config['data']['voxel_size']
        for ann in annotations:
            cls_name, x, y, z, l, w, h, yaw = ann['name'], *ann['location'], *ann['dimensions'], ann['yaw']
            if not (pc_range[0] <= x < pc_range[3] and pc_range[1] <= y < pc_range[4]): continue
            coord_x = (x - pc_range[0]) / voxel_size[0] / downsample_ratio
            coord_y = (y - pc_range[1]) / voxel_size[1] / downsample_ratio
            center_int = (int(coord_y), int(coord_x))
            if not (0 <= center_int[0] < self.feature_map_size[0] and 0 <= center_int[1] < self.feature_map_size[1]): continue
            radius = max(0, int(np.sqrt(w * l) / (downsample_ratio * voxel_size[0])))
            self.draw_gaussian(heatmaps, cls_name, center_int, radius)
            pos_mask[center_int] = True
            box_preds[center_int] = [coord_x - center_int[1], coord_y - center_int[0], z, np.log(l), np.log(w), np.log(h), np.sin(yaw), np.cos(yaw)]
        return {'heatmaps': heatmaps, 'box_preds': box_preds, 'pos_mask': pos_mask}

    def draw_gaussian(self, heatmaps, cls_name, center, radius):
        # ... (This function does not need changes) ...
        for task_id, task in enumerate(self.config['model']['detection_head']['tasks']):
            if cls_name in task['class_names']:
                cls_id = task['class_names'].index(cls_name)
                heatmap = heatmaps[task_id][cls_id]
                diameter = 2 * radius + 1
                gaussian = self.gaussian_2d((diameter, diameter), sigma=diameter / 6)
                x, y = int(center[1]), int(center[0])
                height, width = heatmap.shape[0:2]
                left, right = min(x, radius), min(width - 1 - x, radius)
                top, bottom = min(y, radius), min(height - 1 - y, radius)
                masked_heatmap = heatmap[y - top:y + bottom + 1, x - left:x + right + 1]
                masked_gaussian = gaussian[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
                if masked_heatmap.shape == masked_gaussian.shape:
                    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    @staticmethod
    def gaussian_2d(shape, sigma=1.0):
        # ... (This function does not need changes) ...
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    @staticmethod
    def collate_fn(batch_list):
        input_dicts = [d[0] for d in batch_list]
        target_dicts = [d[1] for d in batch_list]
        
        # --- Input processing (unchanged) ---
        batched_coords_list = []
        for i, d in enumerate(input_dicts):
            coo = torch.from_numpy(d['coordinates'])
            batched_coords_list.append(torch.nn.functional.pad(coo, (1, 0), 'constant', i))
        batched_voxels = torch.from_numpy(np.concatenate([d['voxels'] for d in input_dicts], axis=0))
        batched_coords = torch.cat(batched_coords_list, dim=0)
        final_input_dict = {'batch_size': len(batch_list), 'voxels': batched_voxels, 'coordinates': batched_coords}
        
        # --- [CORE FIX] Target processing with explicit type casting ---
        final_target_dict = {}
        target_keys = target_dicts[0].keys()
        for key in target_keys:
            if key == 'heatmaps':
                num_tasks = len(target_dicts[0][key])
                batched_heatmaps = []
                for task_idx in range(num_tasks):
                    # Ensure float32 for heatmaps
                    hms_for_task = [torch.from_numpy(d[key][task_idx]).float() for d in target_dicts]
                    batched_heatmaps.append(torch.stack(hms_for_task, dim=0))
                final_target_dict[key] = batched_heatmaps
            elif isinstance(target_dicts[0][key], np.ndarray):
                if key == 'gt_points':
                    # Ensure float32 for gt_points
                    tensors_to_cat = [torch.from_numpy(d[key]).float() for d in target_dicts if d[key].shape[0] > 0]
                elif key == 'gt_points_mask':
                    # Ensure bool for masks
                    tensors_to_cat = [torch.from_numpy(d[key]).bool() for d in target_dicts if d[key].shape[0] > 0]
                elif key == 'pos_mask':
                    # Ensure bool for masks
                    stacked = [torch.from_numpy(d[key]).bool() for d in target_dicts]
                    final_target_dict[key] = torch.stack(stacked, dim=0)
                    continue # Skip to next key
                else: # Handles box_preds
                    # Ensure float32 for regression targets
                    stacked = [torch.from_numpy(d[key]).float() for d in target_dicts]
                    final_target_dict[key] = torch.stack(stacked, dim=0)
                    continue # Skip to next key
                
                # Concatenate for gt_points and gt_points_mask
                if tensors_to_cat:
                    final_target_dict[key] = torch.cat(tensors_to_cat, dim=0)
                else:
                    shape = list(target_dicts[0][key].shape)
                    shape[0] = 0
                    dtype_map = {'gt_points': torch.float32, 'gt_points_mask': torch.bool}
                    final_target_dict[key] = torch.empty(shape, dtype=dtype_map.get(key, torch.float32))
        
        return final_input_dict, final_target_dict