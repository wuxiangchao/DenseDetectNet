# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-08-21 17:30:11

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
# from torchvision.ops import box_iou  # 原版(轴向计算IoU，不包含旋转，学不到Yaw)
# fixed bug
from .iou_wrapper import rotated_box_iou


class FocalLoss(nn.Module):
    """
    Standard Focal Loss implementation.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = alpha_weight * (1 - pt).pow(self.gamma)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        loss = focal_weight * bce_loss
        
        # Normalize by the number of positive targets
        num_pos = target.sum().clamp(min=1.0)
        return loss.sum() / num_pos


class DetectionLoss(nn.Module):
    """
    Calculates the detection loss, which includes heatmap loss (Focal) and
    regression loss (SmoothL1 + Rotated IoU).
    """
    def __init__(self, config):
        super(DetectionLoss, self).__init__()
        self.focal_loss = FocalLoss(
            alpha=config['loss']['focal_loss_alpha'], 
            gamma=config['loss']['focal_loss_gamma']
        )
        
        # Use SmoothL1Loss, the industry standard for robust box regression.
        # The beta parameter is a common choice from literature (e.g., SSD).
        self.reg_loss_fn = nn.SmoothL1Loss(reduction='none', beta=1.0/9.0)
        
        self.config = config
        self.l1_weight = config['loss'].get('l1_loss_weight', 1.0)
        self.iou_weight = config['loss'].get('iou_loss_weight', 1.0)

    def decode_boxes_from_regression(self, regression, grid_indices, downsample_ratio):
        pc_range = torch.tensor(self.config['data']['point_cloud_range'], device=regression.device)
        voxel_size = torch.tensor(self.config['data']['voxel_size'], device=regression.device)

        # 将平均尺寸加载到GPU上
        # 主要处理Car
        mean_size_car = torch.tensor(self.config['data']['mean_size']['Car'], device=regression.device, dtype=torch.float32)
        # --------------------------------------
        
        # 获取均值和标准差
        reg_mean = torch.tensor(self.config['data']['reg_targets_mean'], device=regression.device, dtype=torch.float32)
        reg_std = torch.tensor(self.config['data']['reg_targets_std'], device=regression.device, dtype=torch.float32)
        
        # 将模型预测的标准化值，反向转换为真实的回归值
        regression = regression * reg_std + reg_mean
        
        cx = (grid_indices[:, 1].float() + regression[:, 0]) * downsample_ratio * voxel_size[0] + pc_range[0]
        cy = (grid_indices[:, 0].float() + regression[:, 1]) * downsample_ratio * voxel_size[1] + pc_range[1]
        cz = regression[:, 2]
        
        # 尺寸 = (ReLU(预测的残差) + 1) * 平均尺寸
        # ReLU(x)+1 确保乘数总是 >= 1
        # F.relu(regression[:, 3:6]) + 1 这种写法更安全
        size_residuals = regression[:, 3:6]
        size_scaling_factor = F.relu(size_residuals) + 1 
        l, w, h = (size_scaling_factor * mean_size_car).split(1, dim=-1)
        # ---------------------------------------------------------------
        
        yaw = torch.atan2(regression[:, 6], regression[:, 7])
        
        return torch.cat([cx.unsqueeze(-1), cy.unsqueeze(-1), cz.unsqueeze(-1), l, w, h, yaw.unsqueeze(-1)], dim=-1)

    def forward(self, pred_dict, target_dict):
        # Heatmap Loss (Focal Loss)
        heatmap_loss = 0
        pred_hms, target_hms = pred_dict['heatmaps'], target_dict['heatmaps']
        for pred_hm, target_hm in zip(pred_hms, target_hms):
            if pred_hm.shape[2:] != target_hm.shape[2:]:
                pred_hm = F.interpolate(pred_hm, size=target_hm.shape[2:], mode='bilinear', align_corners=False)
            heatmap_loss += self.focal_loss(pred_hm, target_hm)

        # Regression Loss (SmoothL1 + IoU)
        pred_boxes_raw = pred_dict['box_preds']
        pos_mask = target_dict['pos_mask'].bool()
        target_boxes_raw = target_dict['box_preds']

        if pred_boxes_raw.shape[1:3] != pos_mask.shape[1:]:
            pred_boxes_raw = F.interpolate(pred_boxes_raw.permute(0, 3, 1, 2), size=pos_mask.shape[1:], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        if not pos_mask.any():
            reg_loss = pred_boxes_raw.sum() * 0
            details = {
                'heatmap_loss': heatmap_loss.item(), 'reg_loss': 0.0, 'l1_loss': 0.0,
                'iou_loss': 0.0, 'z_h_loss': 0.0
            }
            return heatmap_loss, reg_loss, details

        pos_indices = torch.where(pos_mask)
        grid_yx = torch.stack([pos_indices[1], pos_indices[2]], dim=-1)
        pred_pos_regression = pred_boxes_raw[pos_mask]
        target_pos_regression = target_boxes_raw[pos_mask]
        
        num_pos = pos_mask.sum().clamp(min=1.0)

        # Calculate SmoothL1 loss on ALL 8 raw regression parameters.
        smooth_l1_loss_all_params = self.reg_loss_fn(pred_pos_regression, target_pos_regression).sum(dim=-1)
        smooth_l1_loss = smooth_l1_loss_all_params.sum() / num_pos
        
        # Decode boxes into physical space to calculate IoU loss
        downsample_ratio = self.config['data'].get('downsample_factor', 16)
        pred_physical_boxes = self.decode_boxes_from_regression(pred_pos_regression, grid_yx, downsample_ratio)
        target_physical_boxes = self.decode_boxes_from_regression(target_pos_regression, grid_yx, downsample_ratio)
        
        # Use the precise rotated IoU calculation
        pred_bev_boxes = pred_physical_boxes[:, [0, 1, 3, 4, 6]] # cx, cy, l, w, yaw
        target_bev_boxes = target_physical_boxes[:, [0, 1, 3, 4, 6]]
        
        iou_matrix = rotated_box_iou(pred_bev_boxes, target_bev_boxes)
        iou = iou_matrix.diag()
        iou_loss = (1 - iou).sum() / num_pos
        
        # Combine SmoothL1 and IoU losses for the final regression loss
        reg_loss = self.l1_weight * smooth_l1_loss + self.iou_weight * iou_loss
        
        # For logging purposes, calculate z and h L1 loss separately
        z_h_loss_val = torch.nn.functional.l1_loss(pred_pos_regression[:, [2,5]], target_pos_regression[:, [2,5]], reduction='sum') / num_pos
        
        details = {
            'heatmap_loss': heatmap_loss.item(), 
            'reg_loss': reg_loss.item(),
            'smooth_l1': smooth_l1_loss.item(), # Renamed for clarity
            'iou_loss': iou_loss.item(), 
            'z_h_loss': z_h_loss_val.item()
        }
        return heatmap_loss, reg_loss, details


class D2NetHybridLoss(nn.Module):
    """
    Main loss class that combines and weights losses from all tasks.
    """
    def __init__(self, config):
        super(D2NetHybridLoss, self).__init__()
        self.config = config
        self.detect_weight = config['loss']['detect_weight']
        self.completion_weight = config['loss']['completion_weight']
        self.dense_weight = config['loss']['dense_weight']
        
        self.heatmap_weight = config['loss'].get('heatmap_loss_weight', 1.0)
        
        self.detection_loss = DetectionLoss(config)
        
        self.completion_loss_fn = nn.L1Loss(reduction='sum')

    def forward(self, pred_dict, target_dict):
        # Calculate detection loss components
        heatmap_loss, reg_loss, detect_loss_details = self.detection_loss(pred_dict, target_dict)

        # Combine heatmap and reg loss with its own weight
        detect_loss = (self.heatmap_weight * heatmap_loss + reg_loss)

        # Calculate completion loss
        pred_shape_codes = pred_dict['predicted_shape_codes']
        gt_shape_codes = target_dict.get('gt_shape_codes')
        pos_mask_bev = target_dict.get('pos_mask', torch.zeros_like(pred_shape_codes[..., 0], dtype=torch.bool)).bool()
        
        completion_loss = torch.tensor(0.0, device=detect_loss.device)
        if gt_shape_codes is not None and gt_shape_codes.numel() > 0 and pos_mask_bev.any():
            if pred_shape_codes.shape[1:3] != pos_mask_bev.shape[1:]:
                pred_shape_codes = F.interpolate(
                    pred_shape_codes.permute(0, 3, 1, 2),
                    size=pos_mask_bev.shape[1:],
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            pred_pos_codes = pred_shape_codes[pos_mask_bev]
            gt_pos_codes = gt_shape_codes[pos_mask_bev]
            
            if pred_pos_codes.shape[0] > 0:
                completion_loss = self.completion_loss_fn(pred_pos_codes, gt_pos_codes) / pos_mask_bev.sum().clamp(min=1.0)
            
        # Calculate densification loss
        pred_dense_offsets = pred_dict.get('predicted_dense_offsets')
        voxel_pos_mask = target_dict.get('voxel_pos_mask')
        
        dense_loss = torch.tensor(0.0, device=detect_loss.device)
        if pred_dense_offsets is not None and voxel_pos_mask is not None:
            neg_voxel_mask = ~voxel_pos_mask.bool()
            
            if neg_voxel_mask.any():
                pred_offsets_in_empty_voxels = pred_dense_offsets[neg_voxel_mask]
                gt_points_in_empty_voxels = target_dict['gt_points'][neg_voxel_mask]
                gt_points_mask = target_dict['gt_points_mask'][neg_voxel_mask]
                gt_points_lengths = gt_points_mask.sum(dim=1)
                
                if pred_offsets_in_empty_voxels.shape[0] > 0 and gt_points_in_empty_voxels.shape[0] > 0:
                    dense_loss, _ = chamfer_distance(
                        pred_offsets_in_empty_voxels, 
                        gt_points_in_empty_voxels.to(pred_offsets_in_empty_voxels.dtype), 
                        y_lengths=gt_points_lengths,
                        batch_reduction='mean', point_reduction='mean'
                    )

        # Calculate final weighted total loss
        total_loss = (self.detect_weight * detect_loss + 
                      self.completion_weight * completion_loss + 
                      self.dense_weight * dense_loss)
        
        # Add the original heatmap loss to details for non-weighted logging
        detect_loss_details['raw_heatmap_loss'] = heatmap_loss.item()

        loss_details = {
            'total_loss': total_loss.item(),
            'detect_loss': detect_loss.item(),
            'completion_loss': completion_loss.item(),
            'dense_loss': dense_loss.item(),
            **detect_loss_details
        }
        
        return total_loss, loss_details