"""
训练脚本：多标签 ECG 分类（支持 Focal Loss、加权采样、自定义类别权重）
====================================================================
用法（在项目根目录 ECG_Model/ 下）：
    python src/train.py --config configs/config.yaml

交互式版本（含训练曲线可视化）：见 src/train.ipynb（两者逻辑一致）。

说明：
- 所有路径基于项目根目录（EGC_Model/）解析，可在任意位置运行。
- 验证集使用官方划分 data/processed/val（与 evaluate.py 阈值搜索一致），
  不再从训练集内再切分。
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("提示：未安装 tensorboard，将跳过 TensorBoard 日志（pip install tensorboard 可启用）")
import yaml
import argparse
import json
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, PROJECT_ROOT)
from models.model_zoo.resnet1d import resnet18_1d, resnet34_1d


def _resolve(path: str) -> str:
    """相对路径基于项目根解析；绝对路径原样返回"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


# ---------------------------------- Focal Loss ----------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 可以是 scalar, list, tensor
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C], targets: [B, C]
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        # ---------- 统一处理 alpha ----------
        if self.alpha is not None:
            # 将 alpha 转为 tensor，形状为 [C]（类别数）
            if isinstance(self.alpha, list):
                alpha_t = torch.tensor(self.alpha, dtype=inputs.dtype, device=inputs.device)
            elif isinstance(self.alpha, (float, int)):
                # 标量 -> 扩展成与类别数相同的 tensor
                num_classes = inputs.size(1)
                alpha_t = torch.full((num_classes,), self.alpha, dtype=inputs.dtype, device=inputs.device)
            else:
                alpha_t = self.alpha  # 已经是 tensor

            # 计算权重：正样本用 alpha_t，负样本用 1-alpha_t
            alpha_weight = alpha_t * targets + (1 - alpha_t) * (1 - targets)
            F_loss = alpha_weight * F_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# ---------------------------------- 数据加载 ----------------------------------
def load_data(data_dir):
    """加载官方 train / val / test 划分（相对项目根解析）"""
    def load_npy(rel):
        return np.load(os.path.join(data_dir, rel)).astype(np.float32)

    train_signals = load_npy(os.path.join('train', 'signals.npy'))
    train_labels = load_npy(os.path.join('labels', 'train_multilabel.npy'))
    val_signals = load_npy(os.path.join('val', 'signals.npy'))
    val_labels = load_npy(os.path.join('labels', 'val_multilabel.npy'))
    test_signals = load_npy(os.path.join('test', 'signals.npy'))
    test_labels = load_npy(os.path.join('labels', 'test_multilabel.npy'))
    return (train_signals, train_labels, val_signals, val_labels, test_signals, test_labels)


# ---------------------------------- 正样本权重 ----------------------------------
def get_pos_weights(labels, override=None):
    """
    计算每个类别的正样本权重。
    如果 override 列表不为空，直接使用 override 的值（长度必须等于类别数）。
    """
    if override is not None and len(override) > 0:
        print(f"使用手动设置的正样本权重: {override}")
        return torch.tensor(override, dtype=torch.float32)
    num_pos = labels.sum(axis=0)
    num_neg = len(labels) - num_pos
    pos_weight = num_neg / (num_pos + 1e-6)
    print(f"自动计算正样本权重: {pos_weight}")
    return torch.tensor(pos_weight, dtype=torch.float32)


# ---------------------------------- 采样器 ----------------------------------
def get_weighted_sampler(labels):
    """
    基于标签计算样本权重，用于 WeightedRandomSampler。
    每个样本的权重为正样本类别权重的总和，使得稀有类别样本更可能被采样。
    """
    # 每类正样本数
    pos_counts = labels.sum(axis=0) + 1e-6
    # 类别权重 = 1 / 正样本数
    class_weights = 1.0 / pos_counts
    # 样本权重 = 每个样本对应的类别权重之和
    sample_weights = (labels * class_weights).sum(axis=1)
    # 归一化
    sample_weights /= sample_weights.sum()
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ---------------------------------- 训练/验证 ----------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, label_smoothing=0.0):
    model.train()
    running_loss = 0.0
    for signals, labels in dataloader:
        signals, labels = signals.to(device), labels.to(device)

        if label_smoothing > 0:
            # 手动标签平滑：将 hard label 转换为 soft label
            labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * signals.size(0)
    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for signals, labels in dataloader:
            signals, labels = signals.to(device), labels.to(device)
            logits = model(signals)
            loss = criterion(logits, labels)
            running_loss += loss.item() * signals.size(0)
    return running_loss / len(dataloader.dataset)


# ---------------------------------- 主函数 ----------------------------------
def main(config_path):
    with open(_resolve(config_path), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train_cfg = config['training']
    model_cfg = config['model']
    num_classes = config['data']['num_classes']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据（官方 train / val / test 划分）
    data_dir = _resolve(config['data']['processed_dir'])
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir)
    print(f"训练集大小: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

    # 是否使用加权采样
    use_weighted_sampler = train_cfg.get('use_weighted_sampler', False)
    if use_weighted_sampler:
        print("启用 WeightedRandomSampler 以缓解类别不平衡")
        sampler = get_weighted_sampler(y_train)
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                                  batch_size=train_cfg['batch_size'], sampler=sampler)
    else:
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)

    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)

    # 构建模型
    model_name = model_cfg.get('name', 'resnet18')
    dropout = model_cfg.get('dropout', 0.3)
    if model_name == 'resnet34':
        model = resnet34_1d(in_channels=12, num_classes=num_classes, dropout=dropout)
    else:
        model = resnet18_1d(in_channels=12, num_classes=num_classes, dropout=dropout)
    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 标签平滑
    label_smoothing = float(train_cfg.get('label_smoothing', 0.0))
    if label_smoothing > 0:
        print(f"启用标签平滑，平滑系数: {label_smoothing}")

    # 选择损失函数
    loss_type = train_cfg.get('loss_type', 'bce').lower()
    if loss_type == 'focal':
        alpha = train_cfg.get('focal_alpha', 0.25)
        gamma = train_cfg.get('focal_gamma', 2.0)
        print(f"使用 Focal Loss (alpha={alpha}, gamma={gamma})")
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        # 带权重的 BCE
        pos_weight_override = train_cfg.get('pos_weight_override', None)
        pos_weight = get_pos_weights(y_train, override=pos_weight_override)
        if train_cfg.get('use_weighted_bce', True):
            print("使用加权 BCEWithLogitsLoss")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        else:
            print("使用不加权 BCEWithLogitsLoss")
            criterion = nn.BCEWithLogitsLoss()

    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg['learning_rate']),
        weight_decay=float(train_cfg.get('weight_decay', 5e-4))
    )

    # 学习率调度器
    scheduler_patience = train_cfg.get('lr_scheduler_patience', 10)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience
    )

    # 日志与保存路径（相对项目根）
    log_root = _resolve(config['paths'].get('log_dir', 'logs'))
    os.makedirs(log_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_root, f'{model_name}_{timestamp}')
    writer = SummaryWriter(log_dir) if HAS_TB else None
    checkpoint_dir = _resolve(config['paths']['checkpoints'])
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = train_cfg['early_stopping_patience']

    print(f"\n开始训练，共 {train_cfg['epochs']} 轮，早停耐心值 {patience}")
    for epoch in range(1, train_cfg['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device,
                                 label_smoothing=label_smoothing)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch:3d}/{train_cfg['epochs']} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发，在第 {epoch} 轮停止训练")
                break

    if writer is not None:
        writer.close()
    print(f"训练完成，最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存至: {os.path.join(checkpoint_dir, 'best_model.pth')}")

    train_info = {
        'config': config_path,
        'model_name': model_name,
        'best_val_loss': best_val_loss,
        'checkpoint': os.path.join(checkpoint_dir, 'best_model.pth'),
        'log_dir': log_dir
    }
    with open(os.path.join(log_dir, 'train_info.json'), 'w') as f:
        json.dump(train_info, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    main(args.config)
