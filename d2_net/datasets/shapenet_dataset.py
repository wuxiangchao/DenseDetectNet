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
import trimesh # 仍然需要trimesh做数据增强

class ShapeNetDataset(Dataset):
    """
    现在从预处理好的.npy文件加载点云，速度极快。
    """
    def __init__(self, root_dir, num_points=2048, split='train'):
        self.root_dir = root_dir
        self.num_points = num_points
        self.car_category_id = '02958343'
        
        # 指向我们预处理好的点云文件夹
        processed_dir = os.path.join(self.root_dir, 'processed_pointclouds', self.car_category_id)
        if not os.path.isdir(processed_dir):
            raise FileNotFoundError(f"Processed point cloud directory not found at {processed_dir}. Please run preprocess_shapenet.py first.")
        
        self.npy_paths = []
        model_ids = sorted(os.listdir(processed_dir))
        
        # 简单地按80/20划分训练/验证集
        split_idx = int(len(model_ids) * 0.8)
        if split == 'train':
            model_ids = model_ids[:split_idx]
        else:
            model_ids = model_ids[split_idx:]

        print(f"Loading {split} data from pre-processed .npy files...")
        for model_id in tqdm(model_ids):
            npy_path = os.path.join(processed_dir, model_id, 'pointcloud.npy')
            if os.path.exists(npy_path):
                self.npy_paths.append(npy_path)

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        npy_path = self.npy_paths[idx]
        
        try:
            # 直接加载Numpy数组，速度飞快
            points = np.load(npy_path)

            # 数据增强：随机旋转 (保持不变)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-np.pi, np.pi)
                rot_mat = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
                points = trimesh.transform_points(points, rot_mat)
            
            return torch.from_numpy(points.astype(np.float32))
        except Exception as e:
            print(f"Warning: Could not load or process {npy_path}. Skipping. Error: {e}")
            return None

def shapenet_collate_fn(batch):
    # 过滤掉加载失败的数据 (返回None的项)
    batch = [data for data in batch if data is not None]
    if not batch:
        return None
    return torch.stack(batch, dim=0)

