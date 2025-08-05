# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-07-29 16:30:00

import torch
import yaml
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from d2_net.models.d2net import D2Net
from d2_net.datasets.kitti_dataset import KittiDataset

def post_process(outputs, config, score_thresh=0.1):
    """
    对模型输出进行后处理，将热力图和回归值解码为3D边界框。
    """
    heatmap = outputs['heatmaps'][0][0].sigmoid()
    box_preds = outputs['box_preds'][0]

    final_boxes, final_scores, final_labels = [], [], []

    hmax = F.max_pool2d(heatmap, 3, stride=1, padding=1)
    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep

    for cls_id in range(heatmap.shape[0]):
        ys, xs = torch.where(heatmap[cls_id] > score_thresh)
        if ys.shape[0] == 0:
            continue
        
        task_class_names = []
        for task in config['model']['detection_head']['tasks']:
            task_class_names.extend(task['class_names'])
        cls_name = task_class_names[cls_id]

        for y, x in zip(ys, xs):
            score = heatmap[cls_id, y, x].item()
            box_reg = box_preds[y, x].cpu().numpy()

            pc_range = np.array(config['data']['point_cloud_range'])
            voxel_size = np.array(config['data']['voxel_size'])
            
            downsample_ratio = 16

            cx = (x.item() + box_reg[0]) * voxel_size[0] * downsample_ratio + pc_range[0]
            cy = (y.item() + box_reg[1]) * voxel_size[1] * downsample_ratio + pc_range[1]
            z_bottom = box_reg[2]
            
            # --- [核心修正] ---
            # 在exp操作前，对尺寸的回归值进行范围限制 (clamping)
            # 这必须与 loss.py 中的解码逻辑保持完全一致，以防止回归值爆炸
            dims_reg = box_reg[3:6]
            dims_reg = np.clip(dims_reg, a_min=None, a_max=5.0)
            l, w, h = np.exp(dims_reg)
            # --------------------
            
            yaw = np.arctan2(box_reg[6], box_reg[7])
            z_center = z_bottom + h / 2.0
            
            final_boxes.append([cx, cy, z_center, l, w, h, yaw])
            final_scores.append(score)
            final_labels.append(cls_name)

    return np.array(final_boxes), np.array(final_scores), np.array(final_labels)


def create_o3d_box(box_param, label, score):
    center = box_param[0:3]
    dims = box_param[3:6]
    yaw = box_param[6]
    
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    box3d = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, dims)
    color_map = {'Car': [1, 0, 0], 'Pedestrian': [0, 1, 0], 'Cyclist': [0, 0, 1]}
    box3d.color = color_map.get(label, [0.5, 0.5, 0.5])
    return box3d


def main():
    config_path = 'configs/d2net_kitti.yaml'
    # 使用当前训练好的模型即可
    model_path = 'checkpoints/d2net_stage1_epoch_50.pth'
    sample_id = '000008' 
    print(f"正在对验证集的样本进行推理: {sample_id}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = D2Net(config).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dataset = KittiDataset(config, split='val')
    #------------------------------

    #------------------------------
    try:
        sample_idx = dataset.sample_ids.index(sample_id)
    except ValueError:
        print(f"错误: 样本ID {sample_id} 不在 {dataset.split}.txt 文件中。")
        return
        
    inputs, _ = dataset[sample_idx]

    batch_dict = {
        'voxels': torch.from_numpy(inputs['voxels']),
        'coordinates': torch.nn.functional.pad(torch.from_numpy(inputs['coordinates']), (1, 0), 'constant', 0),
        'batch_size': 1
    }

    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.cuda()

    with torch.no_grad():
        outputs = model(batch_dict)

    boxes, scores, labels = post_process(outputs, config, score_thresh=0.2)
    print(f"检测到： {len(boxes)} 个物体。")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="D2-Net Inference Result (Corrected)")

    # 可视化原始点云
    raw_points = dataset._load_velodyne(sample_id)
    pc_range = config['data']['point_cloud_range']
    mask = np.all((raw_points[:, 0:3] >= pc_range[0:3]) & (raw_points[:, 0:3] < pc_range[3:6]), axis=1)
    filtered_raw_points = raw_points[mask]
    
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(filtered_raw_points[:, :3])
    pcd_raw.paint_uniform_color([0.5, 0.5, 0.5])
    vis.add_geometry(pcd_raw)

    # 可视化稠密化点云
    if 'predicted_point_patches_world' in outputs:
        predicted_patches = outputs['predicted_point_patches_world']
        dense_points = predicted_patches.view(-1, 3).cpu().numpy()
        pcd_dense = o3d.geometry.PointCloud()
        pcd_dense.points = o3d.utility.Vector3dVector(dense_points)
        pcd_dense.paint_uniform_color([0, 0.65, 0.9])
        vis.add_geometry(pcd_dense)

    # 可视化检测框
    for i, box in enumerate(boxes):
        o3d_box = create_o3d_box(box, labels[i], scores[i])
        vis.add_geometry(o3d_box)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.5
    opt.line_width = 10
    opt.show_coordinate_frame = True

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
