# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-31 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-08-04 13:20:00

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import trimesh

class ShapeNetDataset(Dataset):
    """
    用于加载和处理ShapeNet汽车模型的数据集类。
    [核心修改] 使用 trimesh 替代 open3d 以获得更强的鲁棒性。
    """
    def __init__(self, root_dir, num_points=2048, split='train'):
        self.root_dir = root_dir
        self.num_points = num_points
        self.car_category_id = '02958343'
        
        self.obj_paths = []
        car_dir = os.path.join(self.root_dir, self.car_category_id)
        if not os.path.isdir(car_dir):
            raise FileNotFoundError(f"Car category directory not found at {car_dir}")
            
        model_ids = sorted(os.listdir(car_dir))
        
        # 简单地按80/20划分训练/验证集
        split_idx = int(len(model_ids) * 0.8)
        if split == 'train':
            model_ids = model_ids[:split_idx]
        else:
            model_ids = model_ids[split_idx:]

        print(f"Loading {split} data...")
        for model_id in tqdm(model_ids):
            model_path = os.path.join(car_dir, model_id, 'models', 'model_normalized.obj')
            if os.path.exists(model_path):
                self.obj_paths.append(model_path)

    def __len__(self):
        return len(self.obj_paths)

    def __getitem__(self, idx):
        obj_path = self.obj_paths[idx]
        
        try:
            # 使用 trimesh 加载 .obj 文件
            mesh = trimesh.load(obj_path, force='mesh')
            
            # 从网格表面均匀采样点
            points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
            points = points.astype(np.float32)

            # 数据增强：随机旋转
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-np.pi, np.pi)
                rot_mat = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
                points = trimesh.transform_points(points, rot_mat)
            
            return torch.from_numpy(points.astype(np.float32))
        except Exception as e:
            print(f"Warning: Could not load or process {obj_path}. Skipping. Error: {e}")
            # 返回一个虚拟数据，在collate_fn中会被过滤掉
            return None # 返回None，方便过滤

def shapenet_collate_fn(batch):
    # 过滤掉加载失败的数据 (返回None的项)
    batch = [data for data in batch if data is not None]
    if not batch:
        return None
    return torch.stack(batch, dim=0)

