import torch
import spconv.pytorch as spconv
from tqdm import tqdm


def setup_optimizer_scheduler(model, config, args):
    """根据训练阶段设置优化器和调度器"""
    trainable_params = model.parameters()
    optimizer_param_groups = None
    start_epoch = 0

    # 加载 checkpoint
    if args.load_from:
        print(f"Loading checkpoint: {args.load_from}")
        ckpt = torch.load(args.load_from, map_location="cuda")

        if isinstance(ckpt, dict) and "model" in ckpt:
            # 新格式：完整 checkpoint
            model.load_state_dict(ckpt["model"], strict=False)
            if "optimizer" in ckpt:
                start_epoch = ckpt.get("epoch", 0) + 1
                print(f"Resuming from epoch {start_epoch}")
        else:
            # 旧格式：只保存了 model.state_dict()
            model.load_state_dict(ckpt, strict=False)
            print("Loaded model state_dict (old format, no optimizer/scheduler state).")

    # ===== Stage 配置 =====
    if args.stage == 1:
        print("--- Running Training Stage 1: Detection Only ---")
        config['loss'].update({'detect_weight': 1.0, 'completion_weight': 0.0, 'dense_weight': 0.0})


    elif args.stage == 2:
        print("--- Running Training Stage 2: Fine-tuning for Completion ---")
        if not args.load_from: raise ValueError("Stage 2 requires a checkpoint from Stage 1.")
        base_lr = config['training']['lr']
        finetune_lr = base_lr * 0.1
        FINETUNE_KEYWORDS = ['bev_backbone', 'unified_backbone.conv4', 'unified_backbone.conv3']

        completion_head_params, finetune_params = [], []
        for name, param in model.named_parameters():
            param.requires_grad = False
            if 'completion_head' in name:
                param.requires_grad = True
                completion_head_params.append(param)
            else:
                for keyword in FINETUNE_KEYWORDS:
                    if keyword in name:
                        param.requires_grad = True
                        finetune_params.append(param)
                        break

        optimizer_param_groups = [
            {'params': completion_head_params, 'lr': base_lr},
            {'params': finetune_params, 'lr': finetune_lr}
        ]
        config['loss'].update({'detect_weight': 0.0, 'completion_weight': 1.0, 'dense_weight': 0.0})


    elif args.stage == 3:
        print("--- Running Training Stage 3: Densification Only ---")
        if not args.load_from: raise ValueError("Stage 3 requires a checkpoint.")
        for name, param in model.named_parameters():
            param.requires_grad = 'densification_head' in name
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        config['loss'].update({'detect_weight': 0.0, 'completion_weight': 0.0, 'dense_weight': 1.0})


    elif args.stage == 4:
        print("--- Running Training Stage 4: End-to-end Fine-tuning ---")
        if not args.load_from: raise ValueError("Stage 4 requires a checkpoint.")
        for param in model.parameters(): param.requires_grad = True
        config['training']['lr'] *= 0.1
        print(f"Fine-tuning with a small lr: {config['training']['lr']}")

    # ===== 优化器 =====
    if args.stage == 2 and optimizer_param_groups is not None:
        optimizer = torch.optim.AdamW(optimizer_param_groups, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=config['training']['lr'], weight_decay=0.01)

    # ===== Scheduler =====
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[pg.get('lr', config['training']['lr']) for pg in optimizer.param_groups],
        epochs=config['training']['epochs'],
        steps_per_epoch=config['training']['steps_per_epoch'],
        pct_start=0.3
    )

    return optimizer, scheduler, start_epoch



def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, writer, epoch, global_step, config):
    """单个 epoch 训练"""
    model.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
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

        outputs = {k: (v.float() if isinstance(v, torch.Tensor) else [x.float() for x in v]) for k, v in outputs.items()}
        total_loss, loss_details = criterion(outputs, targets)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_norm_clip'])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss_epoch += total_loss.item()
        pbar.set_postfix({k: f'{v:.4f}' for k, v in loss_details.items()})

        writer.add_scalars('Loss_Details/Step', loss_details, global_step)
        writer.add_scalar('Learning_Rate/Step', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('Grad/Norm', grad_norm, global_step)
        writer.add_scalar('Memory/Allocated_MB', torch.cuda.memory_allocated() / 1024**2, global_step)

        global_step += 1
        del outputs, total_loss, loss_details

    avg_loss = total_loss_epoch / len(dataloader) if len(dataloader) > 0 else 0
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    writer.add_scalar('Loss/Epoch_Average', avg_loss, epoch + 1)
    return global_step


def clear_gpu_cache(epoch):
    """清理 GPU 和 spconv 缓存"""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    try:
        spconv.ops.clear_cache()
    except:
        pass
    print(f"--- Cleared GPU cache at the end of epoch: {epoch + 1} ---")


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, path):
    """保存完整 checkpoint，支持断点续训"""
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
    }, path)
