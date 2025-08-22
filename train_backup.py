# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-08-12 16:30:00

import os
import yaml
import argparse
import time
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
# import spconv.pytorch as spconv for clear cache 
import spconv.pytorch as spconv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from d2_net.models.d2net import D2Net
from d2_net.datasets.kitti_dataset import KittiDataset, KittiCollator
from d2_net.utils.loss import D2NetHybridLoss

def main():
    # 1. 设置和解析命令行参数
    parser = argparse.ArgumentParser(description="Multi-stage training script for D2-Net Hybrid Model")
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4],
                        help="Specify training stage: 1 (Detection), 2 (Completion), 3 (Densification), 4 (Fine-tuning)")
    parser.add_argument('--load_from', type=str, default=None,
                        help="Path to checkpoint to load weights from (required for stages > 1).")
    args = parser.parse_args()

    # 2. 加载配置
    config_path = 'configs/d2net_kitti.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 3. 初始化TensorBoard
    log_name = f"stage_{args.stage}_batch_{config['training']['batch_size']}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(f'runs/{log_name}')
    print(f"TensorBoard logs will be saved to: runs/{log_name}")

    # 4. 初始化模型
    model = D2Net(config).cuda()
    
    # 5. 根据训练阶段进行特定设置
    trainable_params = model.parameters() # 默认为全部参数
    start_epoch = 0
    optimizer_param_groups = None # 为stage 2准备
    
    if args.load_from:
        print(f"Loading weights from checkpoint: {args.load_from}")
        model.load_state_dict(torch.load(args.load_from), strict=False)

    if args.stage == 1:
        print("--- Running Training Stage 1: Detection Only ---")
        config['loss']['dense_weight'] = 0.0
        config['loss']['completion_weight'] = 0.0
        config['loss']['detect_weight'] = 1.0
    
    elif args.stage == 2:
        print("--- Running Training Stage 2: Fine-tuning for Completion ---")
        if not args.load_from:
            raise ValueError("Stage 2 requires a checkpoint from Stage 1.")

        
        # 设置差异化学习率
        base_lr = config['training']['lr']
        finetune_lr = base_lr * 0.1
        print(f"Setting up differential learning rates: completion_head LR = {base_lr}, fine-tune LR = {finetune_lr}")

        # 定义需要微调的层的关键字
        # 为补全任务提供特征的关键层
        FINETUNE_KEYWORDS = ['bev_backbone', 'unified_backbone.conv4', 'unified_backbone.conv3']

        completion_head_params = []
        finetune_params = []
        finetune_param_names = [] # 用于打印确认

        print("\n--- Analyzing model parameters for Stage 2 ---")
        for name, param in model.named_parameters():
            # 默认冻结所有层
            param.requires_grad = False
            
            if 'completion_head' in name:
                param.requires_grad = True
                completion_head_params.append(param)
            else:
                # 检查是否是需要微调的层
                for keyword in FINETUNE_KEYWORDS:
                    if keyword in name:
                        param.requires_grad = True
                        finetune_params.append(param)
                        finetune_param_names.append(name)
                        break # 匹配到一个关键字即可，避免重复添加
        
        # 3. 创建优化器参数组
        print(f"\n--- Parameters for Completion Head (LR: {base_lr}) ---")
        print("Total:", len(completion_head_params), "tensors.")
        
        print(f"\n--- Parameters to Fine-tune in Backbone (LR: {finetune_lr}) ---")
        if finetune_params:
            # 使用 set 去除重复的层名，然后排序打印
            for name in sorted(list(set(finetune_param_names))):
                print(name)
        else:
            print(f"Warning: No parameters found with keywords: {FINETUNE_KEYWORDS}. Backbone is fully frozen.")
        
        # 将参数组配置传递给优化器
        optimizer_param_groups = [
            {'params': completion_head_params, 'lr': base_lr},
            {'params': finetune_params, 'lr': finetune_lr}
        ]
        
        # 4. 设置损失权重
        config['loss']['completion_weight'] = 1.0
        config['loss']['detect_weight'] = 0.0
        config['loss']['dense_weight'] = 0.0


    elif args.stage == 3:
        print("--- Running Training Stage 3: Densification Only ---")
        if not args.load_from: raise ValueError("Stage 3 requires a checkpoint.")
        for name, param in model.named_parameters():
            if 'densification_head' not in name: 
                param.requires_grad = False
            else:
                param.requires_grad = True # 确保目标层是可训练的
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        config['loss']['dense_weight'] = 1.0
        config['loss']['completion_weight'] = 0.0
        config['loss']['detect_weight'] = 0.0

    elif args.stage == 4:
        print("--- Running Training Stage 4: End-to-end Fine-tuning ---")
        if not args.load_from: raise ValueError("Stage 4 requires a checkpoint.")
        # 解冻所有层进行微调
        for param in model.parameters():
            param.requires_grad = True
        # 使用配置文件中的权重进行联合微调
        config['training']['lr'] *= 0.1 # 使用一个更小的学习率
        print(f"Fine-tuning with a small learning rate: {config['training']['lr']}")

    # 6. 初始化优化器、数据集等
    # 根据不同阶段选择不同的优化器参数
    if args.stage == 2 and optimizer_param_groups is not None:
        print("\nInitializing optimizer with differential learning rates for Stage 2.")
        optimizer = torch.optim.AdamW(optimizer_param_groups, weight_decay=0.01)
    else:
        print("\nInitializing optimizer with a single learning rate.")
        optimizer = torch.optim.AdamW(trainable_params, lr=config['training']['lr'], weight_decay=0.01)

    train_dataset = KittiDataset(config, split='train')
    collator = KittiCollator(config, split='train')
    train_loader = DataLoader(
        train_dataset, batch_size=config['training']['batch_size'], shuffle=True,
        num_workers=config['training']['num_workers'], collate_fn=collator, pin_memory=True
    )
    criterion = D2NetHybridLoss(config)
    
    # 确保优化器里有参数
    if not optimizer.param_groups:
        raise ValueError("Optimizer has no parameters to optimize. Check requires_grad flags.")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[pg.get('lr', config['training']['lr']) for pg in optimizer.param_groups],
        epochs=config['training']['epochs'], steps_per_epoch=len(train_loader),
        pct_start=0.3 # OneCycleLR的常用设置
    )
    scaler = torch.cuda.amp.GradScaler()

    # 7. 训练循环
    global_step = 0
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"======== Stage {args.stage} | Epoch {epoch + 1}/{config['training']['epochs']} ========")
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        total_loss_epoch = 0
        for i, (inputs, targets) in enumerate(pbar):
            if not targets or 'voxels' not in inputs or inputs['voxels'].shape[0] == 0: 
                continue
            
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor): inputs[key] = inputs[key].cuda(non_blocking=True)
            for key in targets:
                if isinstance(targets[key], torch.Tensor): targets[key] = targets[key].cuda(non_blocking=True)
                elif isinstance(targets[key], list): targets[key] = [t.cuda(non_blocking=True) for t in targets[key]]
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
            
            for key, val in outputs.items():
                if isinstance(val, torch.Tensor): outputs[key] = val.float()
                elif isinstance(val, list): outputs[key] = [v.float() for v in val]
            
            total_loss, loss_details = criterion(outputs, targets)

            # 检查损失是否有效
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: Invalid loss detected (NaN or Inf) at step {global_step}. Skipping update.")
                continue
            
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer) # 在裁剪前unscale
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_norm_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss_epoch += total_loss.item()
            pbar.set_postfix({k: f'{v:.4f}' for k, v in loss_details.items()})
            
            writer.add_scalars('Loss_Details/Step', loss_details, global_step)
            writer.add_scalar('Learning_Rate/Step', optimizer.param_groups[0]['lr'], global_step)
            if len(optimizer.param_groups) > 1: # 记录微调学习率
                 writer.add_scalar('Learning_Rate/Step_finetune', optimizer.param_groups[1]['lr'], global_step)
            
            global_step += 1
            del outputs, total_loss, loss_details

        avg_loss = total_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/Epoch_Average', avg_loss, epoch + 1)

        # new fixed: for clear spconv buffer
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            spconv.ops.clear_cache()
        except:
            pass
        print(f"--- Cleared spconv buffer at the end of epoch: {epoch + 1} ---")
        
        if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
        if (epoch + 1) % 5 == 0:
            save_path = f'checkpoints/d2net_stage{args.stage}_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    # 建议为多进程数据加载设置 'spawn' 启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()