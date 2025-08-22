# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-30 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-08-10 16:30:00

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
    SHAPENET_ROOT = './data/ShapeNetCore.v2'
    LATENT_DIM = 256
    NUM_POINTS = 4096 # [核心修改] 匹配模型中新的点数
    BATCH_SIZE = 32 # 适当减小batch size以适应更大的模型
    EPOCHS = 300 # 训练更长时间以学习更精细的特征
    LR = 0.001
    CHECKPOINT_DIR = './checkpoints_ae' # 使用新目录

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # --- 数据集 ---
    # 让数据集也加载4096个点
    train_dataset = ShapeNetDataset(root_dir=SHAPENET_ROOT, num_points=NUM_POINTS, split='train')
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, collate_fn=shapenet_collate_fn, pin_memory=True
    )

    # --- 模型 ---
    model = ShapeAutoencoder(latent_dim=LATENT_DIM, num_points=NUM_POINTS).cuda()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    
    scaler = torch.cuda.amp.GradScaler()

    print("--- Starting Autoencoder Pre-training (Enhanced Simple MLP Decoder) ---")

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0

        for batch in pbar:
            if batch is None: continue
            
            points = batch.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                reconstructed_points, _ = model(points)
            
            reconstructed_points = reconstructed_points.float()
            loss, _ = chamfer_distance(reconstructed_points, points)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'chamfer_loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Chamfer Loss: {avg_loss:.6f}")
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            encoder_path = f'{CHECKPOINT_DIR}/shapenet_encoder_epoch_{epoch+1}.pth'
            decoder_path = f'{CHECKPOINT_DIR}/shapenet_decoder_epoch_{epoch+1}.pth'
            torch.save(model.encoder.state_dict(), encoder_path)
            torch.save(model.decoder.state_dict(), decoder_path)
            print(f"Saved encoder to {encoder_path} and decoder to {decoder_path}")

    print("--- Autoencoder Pre-training Finished ---")

if __name__ == '__main__':
    main()