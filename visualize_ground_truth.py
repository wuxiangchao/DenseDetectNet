# visualize_ground_truth.py (修正版)
import open3d as o3d
import numpy as np
import argparse
import yaml # [# FIX] 导入yaml库

# 确保可以从项目中导入您的数据集代码
from d2_net.datasets.kitti_dataset import KittiDataset

def create_o3d_box_from_annotation(ann, color):
    """从annotation字典创建一个Open3D的OrientedBoundingBox"""
    center = ann['location']
    dims = ann['dimensions'] # l, w, h
    yaw = ann['yaw']
    
    # Open3D的旋转矩阵需要围绕Z轴
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
    
    # Open3D需要 (l, w, h)
    box3d = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, dims)
    box3d.color = color
    return box3d

def main():
    parser = argparse.ArgumentParser(description="Visualize KITTI ground truth.")
    parser.add_argument('--sample_id', type=str, default='000008',
                        help="KITTI sample ID to visualize (e.g., 000008).")
    args = parser.parse_args()

    config_path = 'configs/d2net_kitti.yaml'

    # --- [核心修正] ---
    # 1. 首先，手动加载YAML配置文件，将其读入一个字典
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. 将加载好的 config 字典传递给 KittiDataset
    #    不再需要创建 D2Net 模型实例
    dataset = KittiDataset(config, split='val') 
    # --- [修正结束] ---
    
    try:
        sample_idx = dataset.sample_ids.index(args.sample_id)
    except ValueError:
        print(f"错误: 样本ID {args.sample_id} 不在 val.txt 文件中。")
        return
        
    data_dict = dataset[sample_idx]
    points = data_dict['points']
    annotations = data_dict['annotations']

    print(f"正在可视化样本: {args.sample_id}")
    print(f"找到 {len(annotations)} 个标注物体。")

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.paint_uniform_color([0.8, 0.8, 0.8]) # 灰色点云

    # 创建坐标系以供参考
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

    # 为每个标注物体创建3D框
    geometries_to_draw = [pcd, coord_frame]
    for ann in annotations:
        print(f"  - 物体: {ann['name']}, 位置: {np.round(ann['location'], 2)}, Yaw: {np.round(ann['yaw'], 2)}")
        box = create_o3d_box_from_annotation(ann, color=[1, 0, 0]) # 红色真值框
        geometries_to_draw.append(box)

    # 可视化
    o3d.visualization.draw_geometries(
        geometries_to_draw,
        window_name=f"Ground Truth Visualization for Sample {args.sample_id}"
    )

if __name__ == '__main__':
    main()