"""
改进版多支路拓扑网络训练脚本

特点:
1. 支持灵活的支路组合配置
2. 详细的训练日志和监控
3. 支持消融实验
"""

import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchvision import transforms
from torch.utils.data import DataLoader
from model import MoCoV3HybridTopo
from UnsupervisedContrastiveDataset import UnsupervisedContrastiveDataset
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import time
import argparse

from config import load_config


# ============================================================
# 默认配置参数
# ============================================================
def get_default_config():
    return load_config("train")


# ============================================================
# 日志函数
# ============================================================
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        print(formatted_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_msg + '\n')


# ============================================================
# 数据加载
# ============================================================
def get_data_loaders(config, image_preprocess, logger):
    """创建数据加载器"""
    logger.log("=" * 70)
    logger.log("Loading datasets...")
    logger.log("=" * 70)

    train_dataset = UnsupervisedContrastiveDataset(
        csv_file=config['train_csv'],
        transform=image_preprocess,
        use_augment=config['use_augment'],
        use_binarization=config['use_binarization'],
        binarization_threshold=config['binarization_threshold']
    )

    val_dataset = UnsupervisedContrastiveDataset(
        csv_file=config['val_csv'],
        transform=image_preprocess,
        use_augment=False,
        use_binarization=False
    )

    test_dataset = UnsupervisedContrastiveDataset(
        csv_file=config['test_csv'],
        transform=image_preprocess,
        use_augment=False,
        use_binarization=False
    )

    logger.log(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=config['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=config['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=config['pin_memory']
    )

    logger.log(f"Batches -> Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}\n")

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


# ============================================================
# 模型初始化
# ============================================================
def create_model(config, device, logger):
    """创建模型"""
    logger.log("=" * 70)
    logger.log("Creating Multi-Branch Topology Model...")
    logger.log("=" * 70)

    model = MoCoV3HybridTopo(
        feature_dim=config['feature_dim'],
        topo_dim=config['topo_dim'],
        K=config['K'],
        m=config['m'],
        T=config['T'],
        topo_weight=config['topo_weight'],
        use_legacy_branch=config['use_legacy_branch'],
        use_skeleton_branch=config['use_skeleton_branch'],
        use_edge_branch=config['use_edge_branch'],
        use_frequency_branch=config['use_frequency_branch'],
        device=device
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log(f"Total parameters: {total_params:,}")
    logger.log(f"Trainable parameters: {trainable_params:,}")

    # 打印支路信息
    enabled = []
    if config['use_legacy_branch']: enabled.append('Legacy')
    if config['use_skeleton_branch']: enabled.append('Skeleton')
    if config['use_edge_branch']: enabled.append('Edge')
    if config['use_frequency_branch']: enabled.append('Frequency')
    logger.log(f"Enabled branches: {', '.join(enabled)}\n")

    return model


# ============================================================
# 优化器和调度器
# ============================================================
def create_optimizer_and_scheduler(model, config, logger):
    """创建优化器和学习率调度器"""
    logger.log("=" * 70)
    logger.log("Creating optimizer and scheduler...")
    logger.log("=" * 70)

    # 分组学习率：backbone用较小学习率，拓扑模块用较大学习率
    backbone_params = []
    topo_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'encoder_q' in name or 'encoder_k' in name:
            backbone_params.append(param)
        elif 'topo_extractor' in name:
            topo_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': config['learning_rate'] * 0.1},
        {'params': topo_params, 'lr': config['learning_rate']},
        {'params': other_params, 'lr': config['learning_rate']}
    ]

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    logger.log(f"Optimizer: AdamW with layer-wise LR")
    logger.log(f"  Backbone LR: {config['learning_rate'] * 0.1}")
    logger.log(f"  Topology LR: {config['learning_rate']}")
    logger.log(f"  Other LR: {config['learning_rate']}")
    logger.log(f"Weight decay: {config['weight_decay']}")

    if config['lr_scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )
        logger.log(f"Scheduler: ReduceLROnPlateau\n")
    elif config['lr_scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['cosine_T_max'],
            eta_min=1e-7
        )
        logger.log(f"Scheduler: CosineAnnealingLR\n")
    elif config['lr_scheduler'] == 'cosine_warmup':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['warmup_epochs'],
            T_mult=2,
            eta_min=1e-7
        )
        logger.log(f"Scheduler: CosineAnnealingWarmRestarts\n")
    else:
        scheduler = None
        logger.log("No scheduler\n")

    return optimizer, scheduler


# ============================================================
# 验证函数
# ============================================================
def evaluate_loss(model, data_loader, device):
    """计算验证损失"""
    model.eval()
    total_visual_loss = 0.0
    total_topo_loss = 0.0
    total_loss = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for img1, img2 in data_loader:
            img1, img2 = img1.to(device), img2.to(device)
            visual_q, visual_k, topo_q, topo_k = model(img1, img2)
            loss, diagnostics = model.contrastive_loss(visual_q, visual_k, topo_q, topo_k)

            total_visual_loss += diagnostics['visual_loss']
            total_topo_loss += diagnostics['topo_loss']
            total_loss += diagnostics['total_loss']

    return (total_visual_loss / num_batches,
            total_topo_loss / num_batches,
            total_loss / num_batches)


# ============================================================
# 训练函数
# ============================================================
def train_epoch(model, train_loader, optimizer, device, config, epoch, logger):
    """训练一个epoch"""
    model.train()
    total_visual_loss = 0.0
    total_topo_loss = 0.0
    total_loss = 0.0
    num_batches = len(train_loader)

    epoch_start_time = time.time()

    for batch_idx, (img1, img2) in enumerate(train_loader):
        img1, img2 = img1.to(device), img2.to(device)

        # Forward pass
        optimizer.zero_grad()
        visual_q, visual_k, topo_q, topo_k = model(img1, img2)
        loss, diagnostics = model.contrastive_loss(visual_q, visual_k, topo_q, topo_k)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config['gradient_clip_max_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['gradient_clip_max_norm']
            )

        # Update parameters
        optimizer.step()

        # Update momentum encoder and queue
        with torch.no_grad():
            model.momentum_update_key_encoder()
            model.update_queue(visual_k, topo_k)

        # Accumulate losses
        total_visual_loss += diagnostics['visual_loss']
        total_topo_loss += diagnostics['topo_loss']
        total_loss += diagnostics['total_loss']

        # Print progress
        if (batch_idx + 1) % max(1, num_batches // 5) == 0:
            progress = (batch_idx + 1) / num_batches * 100
            avg_visual = total_visual_loss / (batch_idx + 1)
            avg_topo = total_topo_loss / (batch_idx + 1)
            avg_total = total_loss / (batch_idx + 1)
            elapsed = time.time() - epoch_start_time
            eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)

            logger.log(
                f"Epoch [{epoch + 1}/{config['num_epochs']}] "
                f"Batch [{batch_idx + 1}/{num_batches} ({progress:.1f}%)] "
                f"| V: {avg_visual:.4f} | T: {avg_topo:.4f} | Total: {avg_total:.4f} "
                f"| ETA: {eta/60:.1f}min"
            )

    # Calculate epoch averages
    avg_epoch_visual_loss = total_visual_loss / num_batches
    avg_epoch_topo_loss = total_topo_loss / num_batches
    avg_epoch_loss = total_loss / num_batches

    epoch_time = time.time() - epoch_start_time

    return avg_epoch_visual_loss, avg_epoch_topo_loss, avg_epoch_loss, epoch_time


# ============================================================
# 主训练函数
# ============================================================
def train_model(model, data_loaders, device, optimizer, scheduler, config, logger):
    """主训练循环"""
    logger.log("=" * 70)
    logger.log("Starting training with Multi-Branch Topology Network...")
    logger.log("=" * 70 + "\n")

    best_val_loss = float('inf')
    early_stop_counter = 0
    patience_limit = config['patience']

    best_model_path = os.path.join(config['checkpoint_dir'], "best_model.pth")
    last_model_path = os.path.join(config['checkpoint_dir'], "last_model.pth")
    best_checkpoint_path = os.path.join(config['checkpoint_dir'], "best_checkpoint.pth")

    history = {
        'train_visual_loss': [],
        'train_topo_loss': [],
        'train_total_loss': [],
        'val_visual_loss': [],
        'val_topo_loss': [],
        'val_total_loss': [],
        'learning_rates': [],
        'epoch_times': []
    }

    # Training loop
    for epoch in range(config['num_epochs']):
        logger.log("=" * 70)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # Learning rate warmup (手动warmup)
        if epoch < config['warmup_epochs'] and config['lr_scheduler'] != 'cosine_warmup':
            warmup_progress = (epoch + 1) / config['warmup_epochs']
            for i, param_group in enumerate(optimizer.param_groups):
                base_lr = config['learning_rate'] if i > 0 else config['learning_rate'] * 0.1
                param_group['lr'] = base_lr * warmup_progress
            current_lr = optimizer.param_groups[1]['lr']
            logger.log(f"Warmup epoch {epoch + 1}/{config['warmup_epochs']}, LR: {current_lr:.6f}\n")

        # Training phase
        train_visual_loss, train_topo_loss, train_total_loss, epoch_time = train_epoch(
            model, data_loaders['train'], optimizer, device, config, epoch, logger
        )

        logger.log(
            f"\nEpoch [{epoch + 1}/{config['num_epochs']}] Training completed ({epoch_time/60:.2f}min)"
        )
        logger.log(f"   Train Visual Loss: {train_visual_loss:.4f}")
        logger.log(f"   Train Topo Loss: {train_topo_loss:.4f}")
        logger.log(f"   Train Total Loss: {train_total_loss:.4f}")

        # Validation phase
        val_visual_loss, val_topo_loss, val_total_loss = evaluate_loss(
            model, data_loaders['val'], device
        )

        logger.log(f"\nValidation Results:")
        logger.log(f"   Val Visual Loss: {val_visual_loss:.4f}")
        logger.log(f"   Val Topo Loss: {val_topo_loss:.4f}")
        logger.log(f"   Val Total Loss: {val_total_loss:.4f}")
        logger.log(f"   Learning Rate: {current_lr:.6f}")

        # Save history
        history['train_visual_loss'].append(train_visual_loss)
        history['train_topo_loss'].append(train_topo_loss)
        history['train_total_loss'].append(train_total_loss)
        history['val_visual_loss'].append(val_visual_loss)
        history['val_topo_loss'].append(val_topo_loss)
        history['val_total_loss'].append(val_total_loss)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append(epoch_time)

        # Learning rate scheduling
        if scheduler is not None:
            if config['lr_scheduler'] == 'plateau':
                scheduler.step(val_total_loss)
            else:
                scheduler.step()

        # Save last model
        torch.save(model.state_dict(), last_model_path)

        # Periodic save
        if (epoch + 1) % config['save_every'] == 0:
            periodic_path = os.path.join(config['checkpoint_dir'], f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), periodic_path)
            logger.log(f"   Periodic save: {periodic_path}")

        # Early stopping check
        if val_total_loss < best_val_loss:
            logger.log(
                f"\nNEW BEST MODEL! Val Loss: {val_total_loss:.4f} "
                f"(Previous: {best_val_loss:.4f})"
            )

            best_val_loss = val_total_loss
            early_stop_counter = 0

            # Save best model
            torch.save(model.state_dict(), best_model_path)

            # Save best checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'config': config,
                'history': history,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, best_checkpoint_path)
            logger.log(f"   Best model and checkpoint saved!")
        else:
            early_stop_counter += 1
            logger.log(
                f"\nNo improvement. Early stopping counter: {early_stop_counter}/{patience_limit}"
            )
            logger.log(f"   Best val loss so far: {best_val_loss:.4f}")

        logger.log("=" * 70 + "\n")

        # Early stopping trigger
        if early_stop_counter >= patience_limit:
            logger.log(
                f"\nEARLY STOPPING TRIGGERED! "
                f"(No improvement for {patience_limit} epochs)"
            )
            logger.log(f"   Best validation loss: {best_val_loss:.4f}\n")
            break

    # Save training history
    history_path = os.path.join(config['checkpoint_dir'], 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    logger.log(f"Training history saved to {history_path}\n")

    return best_model_path, history


# ============================================================
# 测试函数
# ============================================================
def test_model(model, test_loader, device, logger):
    """Test the model on test set"""
    logger.log("=" * 70)
    logger.log("Testing on test set...")
    logger.log("=" * 70)

    test_visual_loss, test_topo_loss, test_total_loss = evaluate_loss(
        model, test_loader, device
    )

    logger.log(f"\nTest Results:")
    logger.log(f"   Test Visual Loss: {test_visual_loss:.4f}")
    logger.log(f"   Test Topo Loss: {test_topo_loss:.4f}")
    logger.log(f"   Test Total Loss: {test_total_loss:.4f}\n")

    return test_visual_loss, test_topo_loss, test_total_loss


# ============================================================
# 特征提取
# ============================================================
def extract_and_save_features(model, data_loaders, device, config, logger):
    """Extract and save features"""
    logger.log("=" * 70)
    logger.log("Extracting features from model...")
    logger.log("=" * 70)

    model.eval()

    features_dict = {
        'train_visual': [],
        'train_topo': [],
        'train_combined': [],
        'val_visual': [],
        'val_topo': [],
        'val_combined': [],
        'test_visual': [],
        'test_topo': [],
        'test_combined': []
    }

    with torch.no_grad():
        for split, loader in data_loaders.items():
            logger.log(f"\nExtracting features from {split} set...")

            visual_features = []
            topo_features = []
            combined_features = []

            for batch_idx, (img1, img2) in enumerate(loader):
                img1 = img1.to(device)

                # Extract visual features
                visual_feat = model.extract_features(
                    img1, use_query_encoder=True, feature_type='visual'
                )

                # Extract topo features
                topo_feat = model.extract_features(
                    img1, use_query_encoder=True, feature_type='topo'
                )

                # Extract combined features
                combined_feat = model.extract_features(
                    img1, use_query_encoder=True, feature_type='both'
                )

                visual_features.append(visual_feat.cpu().numpy())
                topo_features.append(topo_feat.cpu().numpy())
                combined_features.append(combined_feat.cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    logger.log(f"  Processed {batch_idx + 1}/{len(loader)} batches")

            # Concatenate features
            if len(visual_features) > 0:
                visual_feat_array = np.concatenate(visual_features, axis=0)
                topo_feat_array = np.concatenate(topo_features, axis=0)
                combined_feat_array = np.concatenate(combined_features, axis=0)

                features_dict[f'{split}_visual'] = visual_feat_array
                features_dict[f'{split}_topo'] = topo_feat_array
                features_dict[f'{split}_combined'] = combined_feat_array

                logger.log(f"  Visual features shape: {visual_feat_array.shape}")
                logger.log(f"  Topo features shape: {topo_feat_array.shape}")
                logger.log(f"  Combined features shape: {combined_feat_array.shape}")

    # Save features
    features_path = os.path.join(config['checkpoint_dir'], 'extracted_features.npz')
    np.savez(features_path, **features_dict)
    logger.log(f"\nFeatures saved to {features_path}\n")

    return features_dict


# ============================================================
# 绘制训练曲线
# ============================================================
def plot_training_curves(config, logger):
    """Plot training and validation loss curves"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        logger.log("Matplotlib not installed. Skipping visualization.")
        return

    history_path = os.path.join(config['checkpoint_dir'], 'training_history.json')
    if not os.path.exists(history_path):
        logger.log("Training history file not found.")
        return

    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = list(range(1, len(history['train_total_loss']) + 1))

    # Total loss
    axes[0, 0].plot(epochs, history['train_total_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_total_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Visual loss
    axes[0, 1].plot(epochs, history['train_visual_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_visual_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Visual Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Topology loss
    axes[1, 0].plot(epochs, history['train_topo_loss'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_topo_loss'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Topology Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plot_path = os.path.join(config['checkpoint_dir'], 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.log(f"Training curves saved to {plot_path}\n")
    plt.close()


# ============================================================
# 命令行参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Branch Topology Network')

    # 数据路径
    parser.add_argument('--train_csv', type=str, default='./csv/train.csv')
    parser.add_argument('--val_csv', type=str, default='./csv/val.csv')
    parser.add_argument('--test_csv', type=str, default='./csv/test.csv')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=15)

    # 支路配置
    parser.add_argument('--no_legacy', action='store_true', help='Disable legacy erosion branch')
    parser.add_argument('--no_skeleton', action='store_true', help='Disable skeleton branch')
    parser.add_argument('--no_edge', action='store_true', help='Disable edge branch')
    parser.add_argument('--no_frequency', action='store_true', help='Disable frequency branch')

    # 模型参数
    parser.add_argument('--topo_weight', type=float, default=0.35)
    parser.add_argument('--topo_dim', type=int, default=128)
    parser.add_argument('--feature_dim', type=int, default=256)

    # 其他
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/hybrid_v2_12/')
    parser.add_argument('--experiment_name', type=str, default='')

    return parser.parse_args()


# ============================================================
# 主函数
# ============================================================
def main():
    """Main training function"""

    # 解析命令行参数
    args = parse_args()

    # 获取默认配置并更新
    config = get_default_config()
    config['train_csv'] = args.train_csv
    config['val_csv'] = args.val_csv
    config['test_csv'] = args.test_csv
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['num_epochs'] = args.num_epochs
    config['patience'] = args.patience
    # 只有在命令行明确指定时才覆盖默认配置
    if args.no_legacy:
        config['use_legacy_branch'] = False
    if args.no_skeleton:
        config['use_skeleton_branch'] = False
    if args.no_edge:
        config['use_edge_branch'] = False
    if args.no_frequency:
        config['use_frequency_branch'] = False
    config['topo_weight'] = args.topo_weight
    config['topo_dim'] = args.topo_dim
    config['feature_dim'] = args.feature_dim

    # 设置checkpoint目录（只有在命令行指定时才覆盖）
    if args.experiment_name:
        config['checkpoint_dir'] = f"checkpoints/{args.experiment_name}/"
    elif args.checkpoint_dir != 'checkpoints/multi_topo/':
        # 只有当用户明确指定了不同的checkpoint_dir时才覆盖
        config['checkpoint_dir'] = args.checkpoint_dir

    # 创建目录
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(config['checkpoint_dir'], 'training_log.txt')

    # 初始化logger
    logger = Logger(log_file)

    # 图像预处理
    image_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 初始化日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write("=" * 70 + "\n\n")

    logger.log("=" * 70)
    logger.log("MULTI-BRANCH TOPOLOGY NETWORK TRAINING")
    logger.log("=" * 70 + "\n")

    # Set device
    device = torch.device("cuda" if config['use_gpu'] else "cpu")
    if config['use_gpu']:
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.log(f"   CUDA Version: {torch.version.cuda}")
        logger.log(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        logger.log("Using CPU (this may be slow)\n")

    # Save config
    config_path = os.path.join(config['checkpoint_dir'], 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    logger.log(f"Configuration saved to {config_path}")
    logger.log(f"   Checkpoint dir: {config['checkpoint_dir']}\n")

    # Load data
    data_loaders = get_data_loaders(config, image_preprocess, logger)

    # Create model
    model = create_model(config, device, logger)
    model = model.to(device)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, logger)

    # Training configuration summary
    logger.log("=" * 70)
    logger.log("TRAINING CONFIGURATION SUMMARY")
    logger.log("=" * 70)
    logger.log(f"Batch size: {config['batch_size']}")
    logger.log(f"Number of epochs: {config['num_epochs']}")
    logger.log(f"Warmup epochs: {config['warmup_epochs']}")
    logger.log(f"Learning rate: {config['learning_rate']}")
    logger.log(f"Early stopping patience: {config['patience']}")
    logger.log(f"MoCo queue size (K): {config['K']}")
    logger.log(f"Feature dimension: {config['feature_dim']}")
    logger.log(f"Topology dimension: {config['topo_dim']}")
    logger.log(f"Topology weight: {config['topo_weight']}")
    logger.log(f"Branches:")
    logger.log(f"  - Legacy: {config['use_legacy_branch']}")
    logger.log(f"  - Skeleton: {config['use_skeleton_branch']}")
    logger.log(f"  - Edge: {config['use_edge_branch']}")
    logger.log(f"  - Frequency: {config['use_frequency_branch']}")
    logger.log("=" * 70 + "\n")

    try:
        # Start training
        start_time = datetime.now()
        logger.log(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        best_model_path, history = train_model(
            model, data_loaders, device, optimizer, scheduler, config, logger
        )

        end_time = datetime.now()
        training_duration = end_time - start_time
        logger.log(f"\nTraining completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"   Total training time: {training_duration}\n")

    except KeyboardInterrupt:
        logger.log("\n\nTraining interrupted by user!")
        logger.log("   Saving current model...")
        last_model_path = os.path.join(config['checkpoint_dir'], "interrupted_model.pth")
        torch.save(model.state_dict(), last_model_path)
        logger.log(f"   Model saved to {last_model_path}")
        return
    except Exception as e:
        logger.log(f"\n\nError during training: {str(e)}")
        logger.log(f"   Exception type: {type(e).__name__}")
        import traceback
        logger.log(f"\nTraceback:\n{traceback.format_exc()}")
        return

    # Load best model
    logger.log("=" * 70)
    logger.log("LOADING BEST MODEL FOR EVALUATION")
    logger.log("=" * 70 + "\n")

    best_model_path = os.path.join(config['checkpoint_dir'], "best_model.pth")
    if os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            logger.log(f"Best model loaded from: {best_model_path}\n")
        except Exception as e:
            logger.log(f"Failed to load best model: {str(e)}")
            logger.log("   Using current model state instead\n")
    else:
        logger.log(f"Best model path not found: {best_model_path}")
        logger.log("   Using current model state\n")

    # Test on test set
    logger.log("\n")
    test_visual_loss, test_topo_loss, test_total_loss = test_model(
        model, data_loaders['test'], device, logger
    )

    # Extract features
    logger.log("\n")
    features_dict = extract_and_save_features(model, data_loaders, device, config, logger)

    # Plot training curves
    logger.log("\n")
    plot_training_curves(config, logger)

    # Final report
    logger.log("=" * 70)
    logger.log("FINAL TRAINING REPORT")
    logger.log("=" * 70 + "\n")

    best_train_loss = min(history['train_total_loss'])
    best_val_loss = min(history['val_total_loss'])
    best_epoch = history['val_total_loss'].index(best_val_loss) + 1

    logger.log(f"Best Training Results:")
    logger.log(f"   Best epoch: {best_epoch}")
    logger.log(f"   Best training loss: {best_train_loss:.4f}")
    logger.log(f"   Best validation loss: {best_val_loss:.4f}")
    logger.log(f"   Test visual loss: {test_visual_loss:.4f}")
    logger.log(f"   Test topo loss: {test_topo_loss:.4f}")
    logger.log(f"   Test total loss: {test_total_loss:.4f}\n")

    first_val_loss = history['val_total_loss'][0]
    improvement_ratio = (first_val_loss - best_val_loss) / first_val_loss * 100
    logger.log(f"Improvement:")
    logger.log(f"   Initial val loss: {first_val_loss:.4f}")
    logger.log(f"   Final val loss: {best_val_loss:.4f}")
    logger.log(f"   Improvement: {improvement_ratio:.2f}%\n")

    logger.log(f"Feature Statistics:")
    logger.log(f"   Visual feature dimension: {features_dict['train_visual'].shape[1]}")
    logger.log(f"   Topo feature dimension: {features_dict['train_topo'].shape[1]}")
    logger.log(f"   Combined feature dimension: {features_dict['train_combined'].shape[1]}")
    logger.log(f"   Train samples: {features_dict['train_visual'].shape[0]}")
    logger.log(f"   Val samples: {features_dict['val_visual'].shape[0]}")
    logger.log(f"   Test samples: {features_dict['test_visual'].shape[0]}\n")

    logger.log(f"Saved Files:")
    logger.log(f"   Config: {config_path}")
    logger.log(f"   Best model: {best_model_path}")
    logger.log(f"   Training history: {os.path.join(config['checkpoint_dir'], 'training_history.json')}")
    logger.log(f"   Features: {os.path.join(config['checkpoint_dir'], 'extracted_features.npz')}")
    logger.log(f"   Training log: {log_file}\n")

    logger.log("=" * 70)
    logger.log("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.log("=" * 70 + "\n")


# ============================================================
# Entry point
# ============================================================
if __name__ == '__main__':
    try:
        main()
        print("\nAll tasks completed!")
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user!")
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
