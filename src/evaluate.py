#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全功能评估脚本：固定阈值 / 阈值搜索 / 加载优化阈值 / 基线对比 / 临床指标输出
用法：
  基础评估：    python evaluate.py --config configs/config.yaml --checkpoint models/checkpoints/best_model.pth
  阈值搜索：    python evaluate.py ... --optimize_threshold
  使用已有阈值： python evaluate.py ... --use_optimal
  基线对比：    python evaluate.py ... --baseline reports/baseline_results.json
"""
import os
import sys
import argparse
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, classification_report, hamming_loss,
    roc_auc_score, confusion_matrix, precision_score, recall_score
)
import matplotlib
matplotlib.use('Agg')          # 非交互式后端，便于服务器运行
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, List, Dict, Any

# 项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from models.model_zoo.resnet1d import resnet18_1d, resnet34_1d

# ------------------------- 辅助函数 -------------------------
def load_test_data(data_dir: str):
    """加载测试集"""
    signal_path = os.path.join(data_dir, 'test', 'signals.npy')
    label_path  = os.path.join(data_dir, 'labels', 'test_multilabel.npy')
    if not (os.path.exists(signal_path) and os.path.exists(label_path)):
        raise FileNotFoundError("测试数据缺失，请检查 processed_dir/test/ 目录")
    X = np.load(signal_path).astype(np.float32)
    y = np.load(label_path).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)

def load_train_val_data(data_dir: str, val_split: float, random_state: int = 42):
    """加载官方划分的训练集与验证集（阈值搜索用官方 val，避免与训练验证集不一致）"""
    train_signals = np.load(os.path.join(data_dir, 'train', 'signals.npy')).astype(np.float32)
    train_labels  = np.load(os.path.join(data_dir, 'labels', 'train_multilabel.npy')).astype(np.float32)
    val_signals = np.load(os.path.join(data_dir, 'val', 'signals.npy')).astype(np.float32)
    val_labels  = np.load(os.path.join(data_dir, 'labels', 'val_multilabel.npy')).astype(np.float32)
    return train_signals, val_signals, train_labels, val_labels

def load_optimal_thresholds(config: dict) -> np.ndarray:
    """
    尝试从 YAML 或 JSON 文件中加载优化阈值。
    优先级：1. configs/optimized_thresholds.yaml
           2. reports/optimal_thresholds.json
    """
    paths_to_try = [
        os.path.join(project_root, 'configs', 'optimized_thresholds.yaml'),
        os.path.join(project_root, 'configs', 'optimized_thresholds.yml'),
        os.path.join(project_root, 'reports', 'optimal_thresholds.json'),
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            if path.endswith(('.yaml', '.yml')):
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                # 格式可能是 {0: 0.2, 1: 0.3, ...} 或直接列表
                if isinstance(data, dict):
                    # 键为整数字符串或整数
                    thrs = [float(data[str(i)]) for i in range(len(data))]
                else:
                    thrs = list(data)
                return np.array(thrs)
            elif path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'thresholds' in data:
                    return np.array(data['thresholds'])
                else:
                    raise ValueError("JSON 文件中缺少 'thresholds' 字段")
    raise FileNotFoundError(
        "未找到优化阈值文件。请先运行 --optimize_threshold 生成阈值，"
        "或将 optimized_thresholds.yaml 放在 configs/ 目录下。"
    )

# ------------------------- 模型评估 -------------------------
def evaluate_model(model, dataloader, device, threshold):
    """返回概率、预测、真实标签"""
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            logits = model(signals)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            labels_list.append(labels.numpy())
    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    if np.isscalar(threshold):
        preds = (probs >= threshold).astype(int)
    else:
        preds = (probs >= threshold.reshape(1, -1)).astype(int)
    return probs, preds, labels

def threshold_search(probs_val, labels_val, start=0.05, end=0.95, step=0.05):
    """在验证集上每类搜索最佳 F1 阈值"""
    num_classes = labels_val.shape[1]
    thresholds = np.arange(start, end + step, step)
    best_thr = np.zeros(num_classes)
    best_f1  = np.zeros(num_classes)
    for c in range(num_classes):
        f1s = []
        for th in thresholds:
            pred = (probs_val[:, c] >= th).astype(int)
            f1 = f1_score(labels_val[:, c], pred, zero_division=0)
            f1s.append(f1)
        idx = np.argmax(f1s)
        best_thr[c] = thresholds[idx]
        best_f1[c]  = f1s[idx]
    return best_thr, best_f1

# ------------------------- 临床指标 -------------------------
def compute_clinical_metrics(labels, preds, class_names):
    """
    每个类别的灵敏度、特异度、PPV、NPV
    返回字典，方便打印和保存
    """
    metrics = {}
    for idx, name in enumerate(class_names):
        y_true = labels[:, idx]
        y_pred = preds[:, idx]
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        metrics[name] = {
            'sensitivity': round(sensitivity, 4),
            'specificity': round(specificity, 4),
            'ppv': round(ppv, 4),
            'npv': round(npv, 4),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
    return metrics

def print_clinical_table(clinical_metrics: dict):
    """美观打印临床指标表格"""
    print("\n" + "=" * 70)
    print("临床指标（各类别）")
    headers = ["Class", "Sensitivity", "Specificity", "PPV", "NPV"]
    print(f"{'':6s}  {'Sens':>8s}  {'Spec':>8s}  {'PPV':>8s}  {'NPV':>8s}")
    print("-" * 50)
    for cls, m in clinical_metrics.items():
        print(f"{cls:6s}  {m['sensitivity']:8.4f}  {m['specificity']:8.4f}  {m['ppv']:8.4f}  {m['npv']:8.4f}")
    print("=" * 70)

# ------------------------- 可视化 -------------------------
def plot_per_class_confusion_matrices(labels, preds, class_names, save_dir='reports'):
    """为每个类别绘制 2x2 混淆矩阵热力图"""
    n_classes = labels.shape[1]
    n_cols = 3
    n_rows = int(np.ceil(n_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*2.5))
    axes = axes.flatten()
    for idx, (name, ax) in enumerate(zip(class_names, axes)):
        if idx < n_classes:
            tp = ((labels[:, idx]==1) & (preds[:, idx]==1)).sum()
            tn = ((labels[:, idx]==0) & (preds[:, idx]==0)).sum()
            fp = ((labels[:, idx]==0) & (preds[:, idx]==1)).sum()
            fn = ((labels[:, idx]==1) & (preds[:, idx]==0)).sum()
            cm = np.array([[tn, fp],
                           [fn, tp]])
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False,
                        xticklabels=['Pred 0', 'Pred 1'],
                        yticklabels=['True 0', 'True 1'])
            ax.set_title(name)
        else:
            ax.axis('off')
    # 隐藏多余的子图
    for ax in axes[n_classes:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_confusion.png'), dpi=150)
    plt.close()
    print(f"各类别混淆矩阵已保存至 {save_dir}/per_class_confusion.png")

def plot_overall_confusion(labels, preds, class_names, save_dir='reports'):
    """绘制按多数类标签统计的总体混淆矩阵"""
    y_true = labels.argmax(axis=1)
    y_pred = preds.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Overall Confusion Matrix (Argmax)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_confusion.png'), dpi=150)
    plt.close()
    print(f"总体混淆矩阵已保存至 {save_dir}/overall_confusion.png")

# ------------------------- 基线对比 -------------------------
def load_baseline_metrics(baseline_path: str) -> dict:
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"基线结果文件未找到: {baseline_path}")
    with open(baseline_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_with_baseline(deep_metrics: dict, baseline_metrics: dict):
    """
    打印深度学习模型与基线的对比摘要。
    假设 baseline_metrics 中至少包含 'macro_f1', 'micro_f1', 'hamming_loss'
    """
    print("\n" + "=" * 60)
    print("模型性能对比 (深度学习 vs 基线)")
    print(f"{'指标':<20s} {'深度学习':>12s} {'基线':>12s}")
    print("-" * 45)
    keys = [
        ('Hamming Loss', 'hamming_loss'),
        ('Macro F1', 'f1_macro'),
        ('Micro F1', 'f1_micro'),
        ('Macro AUC', 'roc_auc_macro')   # 基线可能没有，适应
    ]
    for name, key in keys:
        deep_val = deep_metrics.get(key, float('nan'))
        base_val = baseline_metrics.get(key, float('nan'))
        if not np.isnan(deep_val) and not np.isnan(base_val):
            print(f"{name:<20s} {deep_val:>12.4f} {base_val:>12.4f}")
    print("=" * 60)

# ------------------------- 主流程 -------------------------
def main(args):
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = config['data']['processed_dir']
    class_names = config['data']['class_names']
    num_classes = len(class_names)

    # 准备模型
    model_cfg = config['model']
    model_name = model_cfg.get('name', 'resnet18')
    if model_name == 'resnet34':
        model = resnet34_1d(in_channels=12, num_classes=num_classes,
                            dropout=model_cfg.get('dropout', 0.2))
    else:
        model = resnet18_1d(in_channels=12, num_classes=num_classes,
                            dropout=model_cfg.get('dropout', 0.2))
    checkpoint = args.checkpoint
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    print(f"模型已加载: {model_name}")

    # --------- 确定阈值 ---------
    final_threshold = args.threshold
    if args.optimize_threshold:
        print("正在进行阈值搜索...")
        val_split = config['training'].get('val_split', 0.2)
        X_train, X_val, y_train, y_val = load_train_val_data(data_dir, val_split)
        val_set = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
        probs_val, _, labels_val = evaluate_model(model, val_loader, device, 0.5)
        best_thr, best_f1 = threshold_search(probs_val, labels_val)
        final_threshold = best_thr  # 数组形式
        print("\n各类别最优阈值及验证集 F1:")
        for i, name in enumerate(class_names):
            print(f"  {name:6s}: threshold={best_thr[i]:.2f}  F1={best_f1[i]:.4f}")
        # 保存阈值（YAML 为权威文件，JSON 供 app 读取）
        os.makedirs('reports', exist_ok=True)
        save_path = os.path.join('reports', 'optimal_thresholds.json')
        with open(save_path, 'w') as f:
            json.dump({'thresholds': best_thr.tolist(), 'val_f1': best_f1.tolist()}, f, indent=2)
        print(f"阈值已保存到 {save_path}")
        yaml_path = os.path.join('configs', 'optimized_thresholds.yaml')
        os.makedirs('configs', exist_ok=True)
        yaml_dict = {str(i): float(th) for i, th in enumerate(best_thr)}
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_dict, f, allow_unicode=True)
        print(f"阈值也保存到 {yaml_path} (权威文件，evaluate.py 与 app 统一读取)")
    elif args.use_optimal:
        print("从文件中加载优化阈值...")
        final_threshold = load_optimal_thresholds(config)
        print("已加载阈值:", final_threshold)
    else:
        print(f"使用固定阈值: {args.threshold}")

    # --------- 测试集评估 ---------
    X_test, y_test = load_test_data(data_dir)
    test_set = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    probs, preds, labels = evaluate_model(model, test_loader, device, final_threshold)

    # --------- 指标计算 ---------
    ham = hamming_loss(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, average='macro', multi_class='ovr')
    except ValueError:
        auc = float('nan')

    print("\n" + "=" * 60)
    print(f"{'测试集结果':^40s}")
    print("=" * 60)
    print(f"Hamming Loss   : {ham:.4f}")
    print(f"Macro F1       : {f1_macro:.4f}")
    print(f"Micro F1       : {f1_micro:.4f}")
    print(f"Macro AUC      : {auc:.4f}")
    print("\n各类别 F1:")
    for name, f1 in zip(class_names, f1_per_class):
        print(f"  {name:6s}: {f1:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    # 临床指标
    clinical = compute_clinical_metrics(labels, preds, class_names)
    print_clinical_table(clinical)

    # 汇总结果
    results = {
        'hamming_loss': float(ham),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'roc_auc_macro': float(auc) if not np.isnan(auc) else None,
        'f1_per_class': f1_per_class.tolist(),
        'clinical_metrics': clinical,
        'threshold_used': final_threshold if isinstance(final_threshold, float) else final_threshold.tolist()
    }

    # --------- 可视化 ---------
    os.makedirs('reports', exist_ok=True)
    plot_overall_confusion(labels, preds, class_names, 'reports')
    plot_per_class_confusion_matrices(labels, preds, class_names, 'reports')

    # --------- 基线对比（如果提供） ---------
    if args.baseline:
        try:
            baseline = load_baseline_metrics(args.baseline)
            compare_with_baseline(results, baseline)
        except Exception as e:
            print(f"⚠️ 基线对比失败: {e}")

    # --------- 保存结果 ---------
    with open('reports/eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已全部保存至 reports/eval_results.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ECG 多标签分类评估")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pth',
                        help='模型权重路径')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='全局阈值 (当不使用优化阈值时)')
    parser.add_argument('--optimize_threshold', action='store_true',
                        help='在验证集上搜索每类最佳阈值')
    parser.add_argument('--use_optimal', action='store_true',
                        help='从配置文件/报告加载已有优化阈值')
    parser.add_argument('--baseline', type=str, default=None,
                        help='基线模型评估结果 JSON 文件路径 (用于对比)')
    args = parser.parse_args()
    main(args)