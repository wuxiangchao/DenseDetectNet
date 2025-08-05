# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-11 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-07-29 15:45:00

import numpy as np
import numba

@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            point_cloud_range,
                            grid_size,
                            max_num_points):
    """
    一个独立的、可被Numba JIT编译的内核函数，用于将点分配到体素中。
    [核心修改] 移除了max_voxels参数，现在处理所有体素。
    """
    num_points = points.shape[0]
    num_features = points.shape[1]
    
    coor_to_voxel_idx = -np.ones(grid_size, dtype=np.int32).flatten()
    
    # 我们不知道最终会有多少体素，所以先用一个足够大的数组
    # KITTI通常不会超过90000个体素
    max_expected_voxels = 90000 
    coors = np.zeros((max_expected_voxels, 3), dtype=np.int32)
    voxels = np.zeros((max_expected_voxels, max_num_points, num_features), dtype=points.dtype)
    voxel_num_points = np.zeros(max_expected_voxels, dtype=np.int32)

    coor = np.zeros(3, dtype=np.int32)
    voxel_idx = 0
    
    for i in range(num_points):
        coor[0] = np.floor((points[i, 0] - point_cloud_range[0]) / voxel_size[0])
        coor[1] = np.floor((points[i, 1] - point_cloud_range[1]) / voxel_size[1])
        coor[2] = np.floor((points[i, 2] - point_cloud_range[2]) / voxel_size[2])

        if not (0 <= coor[0] < grid_size[0] and 0 <= coor[1] < grid_size[1] and 0 <= coor[2] < grid_size[2]):
            continue

        grid_idx = coor[2] * grid_size[1] * grid_size[0] + coor[1] * grid_size[0] + coor[0]

        if coor_to_voxel_idx[grid_idx] == -1:
            if voxel_idx < max_expected_voxels:
                coor_to_voxel_idx[grid_idx] = voxel_idx
                coors[voxel_idx] = coor[::-1] # Store as [z, y, x]
                current_voxel_idx = voxel_idx
                voxel_idx += 1
            else:
                # 如果超出了预估的最大值，忽略这个点
                continue
        else:
            current_voxel_idx = coor_to_voxel_idx[grid_idx]

        current_num_points = voxel_num_points[current_voxel_idx]
        if current_num_points < max_num_points:
            voxels[current_voxel_idx, current_num_points] = points[i]
            voxel_num_points[current_voxel_idx] += 1

    return voxels, coors, voxel_num_points, voxel_idx

class VoxelGenerator:
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels # 这个值现在将在dataset类中使用

        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
        self.grid_size = np.round(grid_size).astype(np.int64)

    def generate(self, points):
        # 调用优化后的JIT函数，它会返回所有找到的体素
        voxels, coordinates, num_points_per_voxel, num_valid_voxels = _points_to_voxel_kernel(
            points, self.voxel_size, self.point_cloud_range,
            tuple(self.grid_size), self.max_num_points
        )
        
        # 截取有效的体素部分
        voxels = voxels[:num_valid_voxels]
        coordinates = coordinates[:num_valid_voxels]
        num_points_per_voxel = num_points_per_voxel[:num_valid_voxels]

        # 计算体素特征 (VFE)
        points_sum = voxels.sum(axis=1)
        num_points_for_feat = num_points_per_voxel[:, np.newaxis]
        safe_num_points = np.where(num_points_for_feat > 0, num_points_for_feat, 1)
        voxel_features = points_sum / safe_num_points
        
        # 为稠密化任务准备GT
        gt_points_padded = np.zeros((num_valid_voxels, self.max_num_points, 3), dtype=np.float32)
        gt_points_mask = np.zeros((num_valid_voxels, self.max_num_points), dtype=np.bool_)
        for i in range(num_valid_voxels):
            num_pts = num_points_per_voxel[i]
            gt_points_padded[i, :num_pts, :] = voxels[i, :num_pts, :3]
            gt_points_mask[i, :num_pts] = True
            
        return {
            'voxels': voxel_features,
            'coordinates': coordinates,
            'gt_points': gt_points_padded,
            'gt_points_mask': gt_points_mask
        }