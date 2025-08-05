# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-07-29 16:30:00

import os
import yaml
import argparse
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from d2_net.models.d2net import D2Net
from d2_net.datasets.kitti_dataset import KittiDataset
from d2_net.utils.loss import D2NetLoss

def main():
    # ... (Argument parsing and config loading remain the same) ...
    parser = argparse.ArgumentParser(description="Two-stage training script for D2-Net")
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2], help="Specify the training stage.")
    parser.add_argument('--load_from', type=str, default=None, help="Path to checkpoint to load from (for stage 2).")
    args = parser.parse_args()

    config_path = 'configs/d2net_kitti.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Initialize TensorBoard ---
    log_name = f"stage_{args.stage}_batch_{config['training']['batch_size']}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(f'runs/{log_name}')
    print(f"TensorBoard logs will be saved to: runs/{log_name}")

    # --- Initialize Model and Optimizer based on stage ---
    model = D2Net(config).cuda()
    
    if args.stage == 1:
        print("--- Running Training Stage 1: Detection Only ---")
        config['loss']['dense_weight'] = 0.0
        config['loss']['detect_weight'] = 1.0
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=0.01)
        start_epoch = 0
    elif args.stage == 2:
        print("--- Running Training Stage 2: Densification Only (frozen backbone) ---")
        if args.load_from is None:
            raise ValueError("A checkpoint path must be provided for stage 2 via --load_from.")
        
        print(f"Loading weights from: {args.load_from}")
        model.load_state_dict(torch.load(args.load_from))
        
        for name, param in model.named_parameters():
            if 'densification_head' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        config['loss']['detect_weight'] = 0.0
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=config['training']['lr'], weight_decay=0.01)
        start_epoch = 0

    # --- Prepare Dataset, Loss, Scheduler, Scaler ---
    train_dataset = KittiDataset(config, split='train')
    train_loader = DataLoader(
        train_dataset, batch_size=config['training']['batch_size'], shuffle=True,
        num_workers=config['training']['num_workers'], collate_fn=KittiDataset.collate_fn, pin_memory=True
    )
    criterion = D2NetLoss(config)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['training']['lr'],
        epochs=config['training']['epochs'], steps_per_epoch=len(train_loader)
    )
    scaler = torch.cuda.amp.GradScaler()

    # --- Training Loop ---
    global_step = 0
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"======== Stage {args.stage} | Epoch {epoch + 1}/{config['training']['epochs']} ========")
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        total_loss_epoch = 0

        for i, (inputs, targets) in enumerate(pbar):
            # ... (Data to GPU logic remains the same) ...
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor): inputs[key] = inputs[key].cuda(non_blocking=True)
            for key in targets:
                if isinstance(targets[key], torch.Tensor): targets[key] = targets[key].cuda(non_blocking=True)
                elif isinstance(targets[key], list): targets[key] = [t.cuda(non_blocking=True) for t in targets[key]]

            if inputs['voxels'].shape[0] == 0: continue

            # [CORE FIX] Perform forward pass inside autocast, but loss outside
            with torch.cuda.amp.autocast():
                outputs = model(inputs) # Model forward pass is in float16
            
            # Manually cast outputs to float32 before loss calculation
            for key, val in outputs.items():
                if isinstance(val, torch.Tensor):
                    outputs[key] = val.float()
                elif isinstance(val, list):
                    outputs[key] = [v.float() for v in val]
            
            # Calculate loss in full float32 precision
            total_loss, loss_details = criterion(outputs, targets)

            # --- Backward pass and optimization (unchanged) ---
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_norm_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss_epoch += total_loss.item()
            pbar.set_postfix({k: f'{v:.4f}' for k, v in loss_details.items()})

            writer.add_scalars('Loss_Details/Step', loss_details, global_step)
            writer.add_scalar('Learning_Rate/Step', optimizer.param_groups[0]['lr'], global_step)
            global_step += 1
            
            del outputs, total_loss, loss_details

        avg_loss = total_loss_epoch / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/Epoch_Average', avg_loss, epoch + 1)

        # ... (Checkpoint saving remains the same) ...
        if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'checkpoints/d2net_stage{args.stage}_epoch_{epoch + 1}.pth')

    writer.close()
    print("Training finished. TensorBoard logs saved.")

if __name__ == '__main__':
    main()