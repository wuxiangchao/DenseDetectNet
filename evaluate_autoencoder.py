import os
import torch
import yaml
import argparse
import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader

from d2_net.models.shape_autoencoder import ShapeAutoencoder
from d2_net.datasets.shapenet_dataset import ShapeNetDataset, shapenet_collate_fn

def visualize_reconstruction(original_points, reconstructed_points):
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(original_points)
    pcd_original.paint_uniform_color([0.7, 0.7, 0.7])

    reconstructed_points[:, 0] += 1.5 
    pcd_reconstructed = o3d.geometry.PointCloud()
    pcd_reconstructed.points = o3d.utility.Vector3dVector(reconstructed_points)
    pcd_reconstructed.paint_uniform_color([0, 0.65, 0.9])

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    print("\n" + "="*30)
    print("正在打开Open3D可视化窗口...")
    print("左侧 (灰色): 原始输入点云")
    print("右侧 (蓝色): 模型重建的点云")
    print("按 'Q' 关闭窗口以查看下一个样本。")
    print("="*30)

    o3d.visualization.draw_geometries([pcd_original, pcd_reconstructed, coord_frame],
                                      window_name="自编码器重建效果评估")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained Shape Autoencoder.")
    parser.add_argument('--config', type=str, default='configs/d2net_kitti.yaml',
                        help="Path to the main project config file.")
    parser.add_argument('--epoch', type=int, required=True,
                        help="Epoch number of the checkpoint to evaluate (e.g., 100).")
    parser.add_argument('--num_samples', type=int, default=5,
                        help="Number of random samples to visualize.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 更新点数和模型加载路径
    NUM_POINTS = 4096
    CHECKPOINT_DIR = './checkpoints_ae'
    
    encoder_path = os.path.join(CHECKPOINT_DIR, f"shapenet_encoder_epoch_{args.epoch}.pth")
    decoder_path = os.path.join(CHECKPOINT_DIR, f"shapenet_decoder_epoch_{args.epoch}.pth")

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Checkpoint for epoch {args.epoch} not found in {CHECKPOINT_DIR}")

    model = ShapeAutoencoder(
        latent_dim=config['model']['shape_autoencoder']['latent_dim'],
        num_points=NUM_POINTS
    ).cuda()
    
    model.encoder.load_state_dict(torch.load(encoder_path))
    model.decoder.load_state_dict(torch.load(decoder_path))
    model.eval()
    print("模型加载成功！")

    val_dataset = ShapeNetDataset(root_dir='./data/ShapeNetCore.v2', 
                                  num_points=NUM_POINTS, 
                                  split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=shapenet_collate_fn)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.num_samples:
                break
            if batch is None:
                continue

            points_original = batch.cuda()
            points_reconstructed, _ = model(points_original)
            
            visualize_reconstruction(
                points_original.squeeze(0).cpu().numpy(),
                points_reconstructed.squeeze(0).cpu().numpy()
            )

    print("评估完成。")

if __name__ == '__main__':
    main()