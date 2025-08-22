# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-08-22 10:30:00

import os
import yaml
import argparse
import time
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from d2_net.models.d2net import D2Net
from d2_net.datasets.kitti_dataset import KittiDataset, KittiCollator
from d2_net.utils.loss import D2NetHybridLoss

from utils.train_utils import (
    setup_optimizer_scheduler,
    train_one_epoch,
    clear_gpu_cache,
    save_checkpoint
)


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description="Multi-stage training script for D2-Net Hybrid Model")
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4],
                        help="Specify training stage: 1 (Detection), 2 (Completion), 3 (Densification), 4 (Fine-tuning)")
    parser.add_argument('--load_from', type=str, default=None,
                        help="Path to checkpoint to load weights from (required for stages > 1).")
    args = parser.parse_args()

    # 加载配置
    config_path = 'configs/d2net_kitti.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 初始化 TensorBoard
    log_name = f"stage_{args.stage}_batch_{config['training']['batch_size']}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(f'runs/{log_name}')
    print(f"TensorBoard logs will be saved to: runs/{log_name}")

    # 初始化模型
    model = D2Net(config).cuda()

    # 初始化优化器 & scheduler
    optimizer, scheduler, start_epoch = setup_optimizer_scheduler(model, config, args)
    criterion = D2NetHybridLoss(config)


    # 数据集
    train_dataset = KittiDataset(config, split='train')
    collator = KittiCollator(config, split='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collator,
        pin_memory=True
    )

    # 自动计算 steps_per_epoch
    config['training']['steps_per_epoch'] = len(train_loader)
    print(f"Steps per epoch: {config['training']['steps_per_epoch']}")

    scaler = torch.cuda.amp.GradScaler()
    global_step = 0

    # 训练循环
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"======== Stage {args.stage} | Epoch {epoch + 1}/{config['training']['epochs']} ========")

        global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, writer,
            epoch, global_step, config
        )

        # 清理缓存
        clear_gpu_cache(epoch)

        # 保存 checkpoint
        if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
        if (epoch + 1) % 5 == 0:
            save_path = f'checkpoints/d2net_stage{args.stage}_epoch_{epoch + 1}.pth'
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, save_path)
            print(f"Saved checkpoint to {save_path}")

    writer.close()
    print("Training finished.")


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()