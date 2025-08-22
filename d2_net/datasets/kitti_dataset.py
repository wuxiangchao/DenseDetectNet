# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-08-11 15:45:00

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm

from ..utils.pointcloud_utils import VoxelGenerator
from ..utils.kitti_utils import Calibration
from ..models.shape_autoencoder import PointNetEncoder

USE_MINI_DATASET = True 

class KittiDataset(Dataset):
    """
    一个轻量级的数据集类，只负责加载原始数据。
    所有繁重的预处理任务都将移交给 KittiCollator 在主进程中完成。
    """
    def __init__(self, config, split='train'):
        self.root_path = config['data']['root_path']
        self.split = split
        
        split_to_load = split
        if split == 'train' and USE_MINI_DATASET:
            print("--- INFO: Using MINI dataset for training for faster iteration! ---")
            split_to_load = 'train_mini'

        split_file_path = os.path.join(self.root_path, 'ImageSets', f'{split_to_load}.txt')
        if not os.path.exists(split_file_path):
             split_file_path = os.path.join(self.root_path, 'PointCloudSets', f'{split_to_load}.txt')

        self.sample_ids = [x.strip() for x in open(split_file_path).readlines()] if os.path.exists(split_file_path) else []
        self.class_names = config['data']['class_names']

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

        # 进行坐标系变换
        for ann in annotations:
            loc_cam = ann['location_cam'].reshape(1, 3)
            loc_velo = calib.project_rect_to_velo(loc_cam)
            ann['location'] = loc_velo.flatten()
            ann['yaw'] = -(ann['rotation_y'] + np.pi / 2.0)
        
        # 返回原始数据，由Collator处理
        return {
            'points': points,
            'annotations': annotations
        }

class KittiCollator:
    """
    一个自定义的 collate_fn 类，在主进程中执行所有繁重的预处理，
    以解决多进程CUDA和RAM溢出问题。
    """
    def __init__(self, config, split):
        self.config = config
        self.split = split
        
        # 在主进程中初始化VoxelGenerator
        self.voxel_generator = VoxelGenerator(
            voxel_size=config['data']['voxel_size'],
            point_cloud_range=config['data']['point_cloud_range'],
            max_num_points=config['data']['max_points_per_voxel'],
            max_voxels=config['data']['max_number_of_voxels'] if split == 'train' else config['data']['test_max_number_of_voxels']
        )
        self.max_voxels = self.voxel_generator.max_voxels
        self.grid_size = self.voxel_generator.grid_size
        downsample_factor = 16
        self.feature_map_size = [
            (self.grid_size[1] + downsample_factor - 1) // downsample_factor,
            (self.grid_size[0] + downsample_factor - 1) // downsample_factor
        ]
        
        # 在主进程中加载一次预训练的形状编码器
        self.shape_encoder = None
        if self.split == 'train':
            self.shape_encoder = PointNetEncoder(
                latent_dim=config['model']['shape_autoencoder']['latent_dim']
            ).cuda()
            encoder_path = config['model']['shape_autoencoder']['encoder_path']
            print(f"Collator: Loading pre-trained shape encoder from: {encoder_path}")
            self.shape_encoder.load_state_dict(torch.load(encoder_path))
            self.shape_encoder.eval()
            for param in self.shape_encoder.parameters():
                param.requires_grad = False

    def __call__(self, batch):
        input_dicts = []
        target_dicts = []

        for item in batch:
            points = item['points']
            annotations = item['annotations']
            
            all_voxel_data = self.voxel_generator.generate(points)
            detection_targets = self.generate_detection_targets(annotations, points)
            
            input_dict, target_dict = self.priority_voxel_sampling(all_voxel_data, detection_targets)
            
            voxel_coords = input_dict['coordinates']
            voxel_size = self.voxel_generator.voxel_size
            pc_range = self.voxel_generator.point_cloud_range
            voxel_centers = (voxel_coords[:, [2, 1, 0]] + 0.5) * voxel_size + pc_range[:3]
            
            gt_points_world = target_dict['gt_points']
            gt_points_local = gt_points_world - voxel_centers[:, np.newaxis, :]
            target_dict['gt_points'] = gt_points_local
            
            input_dicts.append(input_dict)
            target_dicts.append(target_dict)

        # --- 数据打包 ---
        batched_coords_list = []
        for i, d in enumerate(input_dicts):
            coo = torch.from_numpy(d['coordinates'])
            batched_coords_list.append(torch.nn.functional.pad(coo, (1, 0), 'constant', i))
        
        batched_voxels = torch.from_numpy(np.concatenate([d['voxels'] for d in input_dicts], axis=0))
        batched_coords = torch.cat(batched_coords_list, dim=0)
        final_input_dict = {'batch_size': len(batch), 'voxels': batched_voxels, 'coordinates': batched_coords}
        
        final_target_dict = {}
        if not target_dicts:
            return final_input_dict, final_target_dict

        target_keys = target_dicts[0].keys()
        for key in target_keys:
            if key == 'heatmaps':
                num_tasks = len(target_dicts[0][key])
                batched_heatmaps = []
                for task_idx in range(num_tasks):
                    hms_for_task = [torch.from_numpy(d[key][task_idx]).float() for d in target_dicts]
                    batched_heatmaps.append(torch.stack(hms_for_task, dim=0))
                final_target_dict[key] = batched_heatmaps
            elif key == 'voxel_pos_mask':
                tensors_to_cat = [torch.from_numpy(d[key]).bool() for d in target_dicts if d[key].shape[0] > 0]
                if tensors_to_cat:
                    final_target_dict[key] = torch.cat(tensors_to_cat, dim=0)
            elif isinstance(target_dicts[0].get(key), np.ndarray):
                if key in ['gt_points', 'gt_points_mask']:
                    dtype_map = {'gt_points': torch.float32, 'gt_points_mask': torch.bool}
                    tensors_to_cat = [torch.from_numpy(d[key]).to(dtype_map[key]) for d in target_dicts if d[key].shape[0] > 0]
                    if tensors_to_cat:
                        final_target_dict[key] = torch.cat(tensors_to_cat, dim=0)
                else: 
                    stacked = [torch.from_numpy(d[key]).float() for d in target_dicts]
                    final_target_dict[key] = torch.stack(stacked, dim=0)
        
        return final_input_dict, final_target_dict

    def priority_voxel_sampling(self, all_voxel_data, detection_targets):
        if self.split != 'train' or self.max_voxels is None:
            detection_targets.pop('gt_shape_codes', None)
            final_targets = {k: v for k, v in detection_targets.items() if k != 'gt_shape_codes'}
            
            num_voxels = all_voxel_data['coordinates'].shape[0]
            if num_voxels > self.max_voxels:
                final_indices = np.random.choice(num_voxels, self.max_voxels, replace=False)
                for key in ['voxels', 'coordinates', 'gt_points', 'gt_points_mask']:
                    all_voxel_data[key] = all_voxel_data[key][final_indices]

            return {
                'voxels': all_voxel_data['voxels'],
                'coordinates': all_voxel_data['coordinates'],
            }, final_targets

        pos_mask_bev = detection_targets['pos_mask']
        voxel_coords = all_voxel_data['coordinates']
        
        downsample_ratio = 16
        map_coords = voxel_coords[:, 1:] // downsample_ratio
        map_coords[:, 0] = np.clip(map_coords[:, 0], 0, self.feature_map_size[0] - 1)
        map_coords[:, 1] = np.clip(map_coords[:, 1], 0, self.feature_map_size[1] - 1)
        
        is_positive = pos_mask_bev[map_coords[:, 0], map_coords[:, 1]]
        
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

        input_dict = {
            'voxels': all_voxel_data['voxels'][final_indices],
            'coordinates': all_voxel_data['coordinates'][final_indices],
        }
        
        final_is_positive_mask = is_positive[final_indices]

        target_dict = {
            'heatmaps': detection_targets['heatmaps'],
            'box_preds': detection_targets['box_preds'],
            'pos_mask': detection_targets['pos_mask'],
            'gt_shape_codes': detection_targets['gt_shape_codes'],
            'gt_points': all_voxel_data['gt_points'][final_indices],
            'gt_points_mask': all_voxel_data['gt_points_mask'][final_indices],
            'voxel_pos_mask': final_is_positive_mask
        }
        
        return input_dict, target_dict

    def generate_detection_targets(self, annotations, points):
        heatmaps = [np.zeros((len(task['class_names']), *self.feature_map_size), dtype=np.float32) for task in self.config['model']['detection_head']['tasks']]
        box_preds = np.zeros((*self.feature_map_size, 8), dtype=np.float32)
        pos_mask = np.zeros(self.feature_map_size, dtype=np.bool_)
        
        latent_dim = self.config['model']['shape_autoencoder']['latent_dim']
        gt_shape_codes = np.zeros((*self.feature_map_size, latent_dim), dtype=np.float32)

        pc_range = self.config['data']['point_cloud_range']
        voxel_size = self.config['data']['voxel_size']
        downsample_ratio = 16

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

            # 编码一个更稳定的残差目标 ---
            if cls_name in self.config['data']['mean_size']:
                mean_size = self.config['data']['mean_size'][cls_name]
                # 目标: (dim / anchor) - 1
                l_residual = l / mean_size[0] - 1
                w_residual = w / mean_size[1] - 1
                h_residual = h / mean_size[2] - 1
                box_preds[center_int] = [coord_x - center_int[1], coord_y - center_int[0], z, l_residual, w_residual, h_residual, np.sin(yaw), np.cos(yaw)]
            else: # 对于没有定义平均尺寸的类别，保持原样
                box_preds[center_int] = [coord_x - center_int[1], coord_y - center_int[0], z, np.log(l), np.log(w), np.log(h), np.sin(yaw), np.cos(yaw)]
            # ----------------------------------------------------
            # 获取均值和标准差
            reg_mean = np.array(self.config['data']['reg_targets_mean'], dtype=np.float32)
            reg_std = np.array(self.config['data']['reg_targets_std'], dtype=np.float32)
            # 对编码好的box_preds进行标准化
            box_preds[center_int] = (box_preds[center_int] - reg_mean) / reg_std

            if self.split == 'train' and cls_name == 'Car' and self.shape_encoder is not None:
                rot_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                points_in_box = self.get_points_in_box(points, ann['location'], ann['dimensions'], rot_mat)
                
                if points_in_box.shape[0] > 20:
                    points_in_box = self.normalize_and_sample_points(points_in_box, self.config['model']['shape_autoencoder']['num_points'])
                    points_tensor = torch.from_numpy(points_in_box).float().unsqueeze(0).cuda()
                    with torch.no_grad():
                        gt_code = self.shape_encoder(points_tensor).squeeze(0).cpu().numpy()
                    gt_shape_codes[center_int] = gt_code

        return {'heatmaps': heatmaps, 'box_preds': box_preds, 'pos_mask': pos_mask, 'gt_shape_codes': gt_shape_codes}

    def get_points_in_box(self, points, center, dims, rot_mat):
        points_shifted = points[:, :3] - center
        points_local = points_shifted @ np.linalg.inv(rot_mat)
        l, w, h = dims
        mask = (np.abs(points_local[:, 0]) < l / 2) & \
               (np.abs(points_local[:, 1]) < w / 2) & \
               (np.abs(points_local[:, 2]) < h / 2)
        return points_local[mask]

    def normalize_and_sample_points(self, points, num_points):
        max_dims = np.max(np.abs(points), axis=0) + 1e-6
        points /= (max_dims * 2)
        
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        elif len(points) < num_points:
            indices = np.random.choice(len(points), num_points, replace=True)
            points = points[indices]
        return points

    def draw_gaussian(self, heatmaps, cls_name, center, radius):
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
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h
