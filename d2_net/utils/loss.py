# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-07-29 16:30:00

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from torchvision.ops import box_iou

class FocalLoss(nn.Module):
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
        return loss.sum() / (target.sum().clamp(min=1.0))

class DetectionLoss(nn.Module):
    def __init__(self, config):
        super(DetectionLoss, self).__init__()
        self.focal_loss = FocalLoss(
            alpha=config['loss']['focal_loss_alpha'],
            gamma=config['loss']['focal_loss_gamma']
        )
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.config = config
        self.l1_weight = config['loss'].get('l1_loss_weight', 1.0)
        self.iou_weight = config['loss'].get('iou_loss_weight', 1.0)

    def decode_boxes_from_regression(self, regression, grid_indices, downsample_ratio):
        pc_range = torch.tensor(self.config['data']['point_cloud_range'], device=regression.device)
        voxel_size = torch.tensor(self.config['data']['voxel_size'], device=regression.device)
        cx = (grid_indices[:, 1].float() + regression[:, 0]) * downsample_ratio * voxel_size[0] + pc_range[0]
        cy = (grid_indices[:, 0].float() + regression[:, 1]) * downsample_ratio * voxel_size[1] + pc_range[1]
        cz = regression[:, 2]
        clamped_regression = regression[:, 3:6].clamp(max=5.0)
        l, w, h = torch.exp(clamped_regression).split(1, dim=-1)
        yaw = torch.atan2(regression[:, 6], regression[:, 7])
        return torch.cat([cx.unsqueeze(-1), cy.unsqueeze(-1), cz.unsqueeze(-1), l, w, h, yaw.unsqueeze(-1)], dim=-1)

    @staticmethod
    def get_bev_corners(boxes_7d):
        cx, cy, _, l, w, _, yaw = boxes_7d.unbind(dim=-1)
        corners_local = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=boxes_7d.dtype, device=boxes_7d.device)
        corners_local = corners_local * torch.stack([l, w], dim=-1).unsqueeze(1)
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        rot_mat = torch.stack([cos_yaw, -sin_yaw, sin_yaw, cos_yaw], dim=-1).view(-1, 2, 2)
        corners_rotated = torch.bmm(corners_local, rot_mat)
        corners = corners_rotated + torch.stack([cx, cy], dim=-1).unsqueeze(1)
        return corners
        
    def forward(self, pred_dict, target_dict):
        # 1. Heatmap Loss
        heatmap_loss = 0
        pred_hms = pred_dict['heatmaps']
        target_hms = target_dict['heatmaps']
        
        resized_pred_hms = []
        for i, pred_hm in enumerate(pred_hms):
            if pred_hm.shape[2:] != target_hms[i].shape[2:]:
                resized_pred_hms.append(F.interpolate(pred_hm, size=target_hms[i].shape[2:], mode='bilinear', align_corners=False))
            else:
                resized_pred_hms.append(pred_hm)

        for pred_hm, target_hm in zip(resized_pred_hms, target_hms):
            heatmap_loss += self.focal_loss(pred_hm, target_hm)

        # 2. Box Regression Loss
        pred_boxes_raw = pred_dict['box_preds']
        pos_mask = target_dict['pos_mask'].bool()
        target_boxes_raw = target_dict['box_preds']
        
        if pred_boxes_raw.shape[1:3] != pos_mask.shape[1:]:
            pred_boxes_raw = F.interpolate(
                pred_boxes_raw.permute(0, 3, 1, 2),
                size=pos_mask.shape[1:],
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        if not pos_mask.any():
            reg_loss = pred_boxes_raw.sum() * 0
            return heatmap_loss, {'heatmap_loss': heatmap_loss.item(), 'reg_loss': 0}

        pos_indices = torch.where(pos_mask)
        grid_yx = torch.stack([pos_indices[1], pos_indices[2]], dim=-1)
        pred_pos_regression = pred_boxes_raw[pos_mask]
        target_pos_regression = target_boxes_raw[pos_mask]
        
        downsample_ratio = 16
        pred_physical_boxes = self.decode_boxes_from_regression(pred_pos_regression, grid_yx, downsample_ratio)
        target_physical_boxes = self.decode_boxes_from_regression(target_pos_regression, grid_yx, downsample_ratio)

        num_pos = pos_mask.sum().clamp(min=1.0)
        
        z_loss = self.l1_loss(pred_physical_boxes[:, 2], target_physical_boxes[:, 2]) / num_pos
        h_loss = self.l1_loss(pred_physical_boxes[:, 5], target_physical_boxes[:, 5]) / num_pos
        
        pred_corners = self.get_bev_corners(pred_physical_boxes)
        target_corners = self.get_bev_corners(target_physical_boxes)
        pred_aa_boxes = torch.cat([pred_corners.min(dim=1)[0], pred_corners.max(dim=1)[0]], dim=1)
        target_aa_boxes = torch.cat([target_corners.min(dim=1)[0], target_corners.max(dim=1)[0]], dim=1)
        
        iou = box_iou(pred_aa_boxes, target_aa_boxes).diag()
        iou_loss = (1 - iou).sum() / num_pos
        
        reg_loss = self.l1_weight * (z_loss + h_loss) + self.iou_weight * iou_loss
        total_loss = heatmap_loss + reg_loss

        return total_loss, {
            'heatmap_loss': heatmap_loss.item(), 'reg_loss': reg_loss.item(),
            'iou_loss': iou_loss.item(), 'z_h_loss': (z_loss + h_loss).item()
        }

class D2NetLoss(nn.Module):
    def __init__(self, config):
        super(D2NetLoss, self).__init__()
        self.dense_weight = config['loss']['dense_weight']
        self.detect_weight = config['loss']['detect_weight']
        self.detection_loss = DetectionLoss(config)

    def forward(self, pred_dict, target_dict):
        # --- Densification Loss ---
        pred_patches = pred_dict['predicted_point_offsets']
        gt_patches = target_dict['gt_points']
        gt_mask = target_dict['gt_points_mask']
        gt_lengths = gt_mask.sum(dim=1)
        
        # [CORE FIX] Ensure both tensors have the same dtype before passing to chamfer_distance
        gt_patches = gt_patches.to(pred_patches.dtype)
        
        dense_loss, _ = chamfer_distance(
            pred_patches,
            gt_patches,
            y_lengths=gt_lengths,
            batch_reduction='mean',
            point_reduction='mean'
        )
        
        # --- Detection Loss ---
        detect_loss, detect_loss_details = self.detection_loss(pred_dict, target_dict)

        if torch.isnan(dense_loss) or torch.isinf(dense_loss) or torch.isnan(detect_loss) or torch.isinf(detect_loss):
            print(f"NaN/Inf detected! dense_loss: {dense_loss.item()}, detect_loss: {detect_loss.item()}")

        total_loss = self.dense_weight * dense_loss + self.detect_weight * detect_loss
        loss_details = {
            'total_loss': total_loss.item(), 'dense_loss': dense_loss.item(), 
            'detect_loss': detect_loss.item(), **detect_loss_details
        }
        return total_loss, loss_details