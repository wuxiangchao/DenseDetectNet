# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-08-18 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-08-21 23:30:00

import torch
# 引入自定义的CUDA IOU计算模块（基于CPP实现）
from . import rotated_iou_cuda_kernel

def rotated_box_iou(boxes1, boxes2):
    """
    计算两组旋转框的IoU。
    
    Args:
        boxes1 (torch.Tensor): Rotated boxes of shape (N, 5) with format [cx, cy, w, h, angle_rad].
        boxes2 (torch.Tensor): Rotated boxes of shape (M, 5) with format [cx, cy, w, h, angle_rad].
        
    Returns:
        torch.Tensor: IoU matrix of shape (N, M).
    """
    # 将box参数转换为8个角点 (N, 4, 2)
    corners1 = _box_to_corners(boxes1)
    corners2 = _box_to_corners(boxes2)

    # 调用CUDA kernel计算交集面积
    # 确保输入是连续的内存布局
    intersection_areas = rotated_iou_cuda_kernel.calculate_intersection(corners1.float().contiguous(), corners2.float().contiguous())

    # 计算各自的面积
    areas1 = boxes1[:, 2] * boxes1[:, 3]
    areas2 = boxes2[:, 2] * boxes2[:, 3]

    # 计算并集面积
    union_areas = areas1.unsqueeze(1) + areas2.unsqueeze(0) - intersection_areas

    # 计算IoU
    iou = intersection_areas / union_areas.clamp(min=1e-6)
    
    return iou

def _box_to_corners(boxes):
    """将 [cx, cy, w, h, angle] 格式的box转换为4个角点"""
    cx, cy, w, h, a = boxes.unbind(dim=-1)
    
    # 在局部坐标系下创建角点
    corners_local = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                                 dtype=boxes.dtype, device=boxes.device)
    corners_local = corners_local * torch.stack([w, h], dim=-1).unsqueeze(1)

    # 旋转
    cos_a, sin_a = torch.cos(a), torch.sin(a)
    rot_mat = torch.stack([cos_a, -sin_a, sin_a, cos_a], dim=-1).view(-1, 2, 2)
    corners_rotated = torch.bmm(corners_local, rot_mat)

    # 平移
    corners = corners_rotated + torch.stack([cx, cy], dim=-1).unsqueeze(1)

    return corners