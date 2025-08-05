# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-30 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-08-04 16:30:00

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
import os

from d2_net.models.shape_autoencoder import ShapeAutoencoder
from d2_net.datasets.shapenet_dataset import ShapeNetDataset, shapenet_collate_fn

def main():
    # --- 配置 ---
    SHAPENET_ROOT = './data/ShapeNetCore.v2' # 修改为您的ShapeNet根目录
    LATENT_DIM = 256
    NUM_POINTS = 2048
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 0.001
    CHECKPOINT_DIR = './checkpoints_ae'

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # --- 数据集 ---
    train_dataset = ShapeNetDataset(root_dir=SHAPENET_ROOT, num_points=NUM_POINTS, split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        collate_fn=shapenet_collate_fn,
        pin_memory=True
    )

    # --- 模型 ---
    model = ShapeAutoencoder(latent_dim=LATENT_DIM, num_points=NUM_POINTS).cuda()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print("--- Starting Autoencoder Pre-training ---")

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0

        for batch in pbar:
            if batch is None: continue # 跳过空的批次
            
            points = batch.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            
            reconstructed_points, _ = model(points)
            
            # 使用Chamfer Distance作为损失函数
            loss, _ = chamfer_distance(reconstructed_points, points)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'chamfer_loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Chamfer Loss: {avg_loss:.6f}")
        scheduler.step()

        # 保存模型
        if (epoch + 1) % 10 == 0:
            # 我们只需要解码器部分用于后续任务
            torch.save(model.decoder.state_dict(), f'{CHECKPOINT_DIR}/shapenet_decoder_epoch_{epoch+1}.pth')
            print(f"Saved decoder checkpoint at epoch {epoch+1}")

    print("--- Autoencoder Pre-training Finished ---")

if __name__ == '__main__':
    main()
