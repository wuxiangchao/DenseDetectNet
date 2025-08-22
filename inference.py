# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-08-11 16:30:00

import torch
import yaml
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from d2_net.models.d2net import D2Net
from d2_net.datasets.kitti_dataset import KittiDataset
from d2_net.utils.pointcloud_utils import VoxelGenerator # [核心] 导入 VoxelGenerator

def post_process_hybrid(outputs, config, score_thresh=0.1):
    """
    后处理函数，用于解码混合模型的输出，包括检测框和形状编码。
    """
    heatmap = outputs['heatmaps'][0][0].sigmoid()
    box_preds = outputs['box_preds'][0]
    shape_codes = outputs['predicted_shape_codes'][0]

    final_boxes, final_scores, final_labels, final_codes = [], [], [], []

    hmax = F.max_pool2d(heatmap, 3, stride=1, padding=1)
    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep

    for cls_id in range(heatmap.shape[0]):
        ys, xs = torch.where(heatmap[cls_id] > score_thresh)
        if ys.shape[0] == 0: continue
        
        task_class_names = [name for task in config['model']['detection_head']['tasks'] for name in task['class_names']]
        cls_name = task_class_names[cls_id]

        for y, x in zip(ys, xs):
            score = heatmap[cls_id, y, x].item()
            box_reg = box_preds[y, x].cpu().numpy()
            shape_code = shape_codes[y, x].cpu().numpy()

            pc_range = np.array(config['data']['point_cloud_range'])
            voxel_size = np.array(config['data']['voxel_size'])
            downsample_ratio = 16

            cx = (x.item() + box_reg[0]) * voxel_size[0] * downsample_ratio + pc_range[0]
            cy = (y.item() + box_reg[1]) * voxel_size[1] * downsample_ratio + pc_range[1]
            z_bottom = box_reg[2]
            
            dims_reg = box_reg[3:6]
            dims_reg = np.clip(dims_reg, a_min=None, a_max=5.0)
            l, w, h = np.exp(dims_reg)
            
            yaw = np.arctan2(box_reg[6], box_reg[7])
            z_center = z_bottom + h / 2.0
            
            final_boxes.append([cx, cy, z_center, l, w, h, yaw])
            final_scores.append(score)
            final_labels.append(cls_name)
            final_codes.append(shape_code)

    return np.array(final_boxes), np.array(final_scores), np.array(final_labels), np.array(final_codes)

def create_o3d_box(box_param, label):
    center, dims, yaw = box_param[0:3], box_param[3:6], box_param[6]
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
    box3d = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, dims)
    color_map = {'Car': [1, 0, 0], 'Pedestrian': [0, 1, 0], 'Cyclist': [0, 0, 1]}
    box3d.color = color_map.get(label, [0.5, 0.5, 0.5])
    return box3d

def main():
    config_path = 'configs/d2net_kitti.yaml'
    model_path = 'checkpoints/d2net_stage1_epoch_25.pth'
    sample_id = '000008' 
    print(f"推理样本: {sample_id}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 加载模型
    model = D2Net(config).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 加载数据集对象以获取原始数据
    dataset = KittiDataset(config, split='val') 
    
    try:
        sample_idx = dataset.sample_ids.index(sample_id)
    except ValueError:
        print(f"错误: 样本ID {sample_id} 不在 val.txt 文件中。")
        return
        
    raw_data = dataset[sample_idx]
    points = raw_data['points']
    
    # [核心修正] 在推理脚本中独立创建 VoxelGenerator
    voxel_generator = VoxelGenerator(
        voxel_size=config['data']['voxel_size'],
        point_cloud_range=config['data']['point_cloud_range'],
        max_num_points=config['data']['max_points_per_voxel'],
        max_voxels=config['data']['test_max_number_of_voxels'] # 使用测试时的体素上限
    )
    voxel_data = voxel_generator.generate(points)
    
    # 手动打包Batch
    batch_dict = {
        'voxels': torch.from_numpy(voxel_data['voxels']),
        'coordinates': torch.nn.functional.pad(torch.from_numpy(voxel_data['coordinates']), (1, 0), 'constant', 0),
        'batch_size': 1
    }
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor): batch_dict[key] = val.cuda()

    # 执行推理
    with torch.no_grad():
        outputs = model(batch_dict)

    # 后处理
    boxes, scores, labels, shape_codes = post_process_hybrid(outputs, config, score_thresh=0.4)
    print(f"检测到 {len(boxes)} 个物体。")

    # --- 可视化 ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="D2-Net Hybrid (Detection + Completion + Densification)")

    # 1. 可视化原始点云
    raw_points = dataset._load_velodyne(sample_id)
    pc_range = config['data']['point_cloud_range']
    mask = np.all((raw_points[:, 0:3] >= pc_range[0:3]) & (raw_points[:, 0:3] < pc_range[3:6]), axis=1)
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(raw_points[mask][:, :3])
    pcd_raw.paint_uniform_color([0.5, 0.5, 0.5])
    vis.add_geometry(pcd_raw)

    # 2. 可视化稠密化后的背景点云
    if 'predicted_dense_patches_world' in outputs:
        dense_patches = outputs['predicted_dense_patches_world']
        dense_points = dense_patches.view(-1, 3).cpu().numpy()
        pcd_dense = o3d.geometry.PointCloud()
        pcd_dense.points = o3d.utility.Vector3dVector(dense_points)
        pcd_dense.paint_uniform_color([0, 0.65, 0.9])
        vis.add_geometry(pcd_dense)
        
    # 3. 生成、变换并可视化补全后的物体点云
    shape_codes_tensor = torch.from_numpy(shape_codes).cuda()
    if shape_codes_tensor.shape[0] > 0:
        with torch.no_grad():
            completed_points_batch = model.shape_decoder(shape_codes_tensor)
        
        for i, box in enumerate(boxes):
            if labels[i] != 'Car': continue

            completed_points = completed_points_batch[i].cpu().numpy()
            center, dims, yaw = box[0:3], box[3:6], box[6]
            l, w, h = dims
            
            completed_points[:, 0] *= l
            completed_points[:, 1] *= w
            completed_points[:, 2] *= h

            rot_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            completed_points = completed_points @ rot_mat.T
            
            completed_points += center
            
            pcd_completed = o3d.geometry.PointCloud()
            pcd_completed.points = o3d.utility.Vector3dVector(completed_points)
            pcd_completed.paint_uniform_color([0, 0.9, 0.2])
            vis.add_geometry(pcd_completed)
            
            o3d_box = create_o3d_box(box, labels[i])
            vis.add_geometry(o3d_box)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main()
