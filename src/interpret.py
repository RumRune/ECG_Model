# src/interpret.py
# -*- coding: utf-8 -*-
# type: ignore
"""
模型可解释性模块：
- Grad‑CAM 1D：定位重要时间区域
- (可选) Integrated Gradients：全局特征归因
- 可视化与报告生成
用法：
  python src/interpret.py --config configs/config.yaml --checkpoint models/checkpoints/best_model.pth --sample 0
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

# 项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from models.model_zoo.resnet1d import resnet18_1d, resnet34_1d

# 尝试导入 captum（可选，用于 Integrated Gradients）
try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("提示：未安装 captum，Integrated Gradients 功能不可用。可执行 pip install captum 安装。")

# ------------------------- 辅助函数 -------------------------
def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """加载模型和配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    num_classes = config['data']['num_classes']
    model_name = config['model'].get('name', 'resnet18')
    dropout = config['model'].get('dropout', 0.2)
    if model_name == 'resnet34':
        model = resnet34_1d(in_channels=12, num_classes=num_classes, dropout=dropout)
    else:
        model = resnet18_1d(in_channels=12, num_classes=num_classes, dropout=dropout)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, config

def load_sample(data_dir: str, idx: int = 0):
    """加载单个样本的信号和标签"""
    test_signals = np.load(os.path.join(data_dir, 'test', 'signals.npy'))
    test_labels  = np.load(os.path.join(data_dir, 'labels', 'test_multilabel.npy'))
    signal = test_signals[idx]
    label = test_labels[idx]
    return torch.from_numpy(signal).float(), label

# ------------------------- Grad‑CAM 1D 实现 -------------------------
class GradCAM1D:
    """
    一维 Grad‑CAM 实现，适用于 Conv1d 网络。
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None, retain_graph=False):
        """
        生成 Grad‑CAM 热力图
        Args:
            input_tensor: (1, C, L) 输入
            class_idx: 指定类别索引，若为 None 则取预测概率最高的类
        Returns:
            heatmap: (L,) 一维热力权重
        """
        self.model.eval()
        input_tensor.requires_grad_()

        output = self.model(input_tensor)           # (1, num_classes)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=retain_graph)

        # 获取最后卷积层的激活图和梯度
        acts = self.activations.detach()           # (1, channels, length)
        grads = self.gradients.detach()            # (1, channels, length)

        # 全局平均池化梯度 -> 通道权重
        weights = grads.mean(dim=-1, keepdim=True) # (1, channels, 1)
        # 加权组合
        cam = (weights * acts).sum(dim=1).squeeze(0)  # (length,)
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()
        return cam.cpu().numpy()

def find_last_conv_layer(model):
    """
    自动查找最后一个卷积层（适用于 ResNet1D）。
    返回该层模块本身。
    """
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            last_conv = module
    if last_conv is None:
        raise ValueError("模型中未找到 Conv1d 层，无法使用 Grad‑CAM")
    return last_conv

# ------------------------- Integrated Gradients（可选）------------------
def integrated_gradients_attribution(model, input_tensor, class_idx, steps=50):
    """
    使用 Captum 计算 Integrated Gradients 归因。
    Args:
        input_tensor: (1, C, L) 输入
        class_idx: 目标类别
    Returns:
        attributions: (1, C, L) 每个位置的重要性
    """
    if not HAS_CAPTUM:
        print("Captum 未安装，无法使用 Integrated Gradients")
        return None
    model.eval()
    baseline = torch.zeros_like(input_tensor)
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(input_tensor, baseline, target=class_idx,
                                       n_steps=steps, return_convergence_delta=True)
    return attributions.squeeze(0).detach().cpu().numpy()  # (C, L)

# ------------------------- 可视化与报告 -------------------------
def plot_gradcam(signal: np.ndarray, heatmap: np.ndarray, class_name: str, save_path: str):
    """
    绘制原始信号（所有导联）并叠加热力图强度。
    Args:
        signal: (C, L) 原始信号
        heatmap: (L,) Grad‑CAM 热力权重
    """
    C, L = signal.shape
    fig, axes = plt.subplots(C, 1, figsize=(12, C*1.5), sharex=True)
    if C == 1:
        axes = [axes]
    time_axis = np.arange(L)
    for lead, ax in enumerate(axes):
        ax.plot(time_axis, signal[lead], color='blue', linewidth=0.8)
        ax2 = ax.twinx()
        ax2.fill_between(time_axis, 0, heatmap, alpha=0.3, color='red')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Importance', color='red')
        ax.set_ylabel(f'Lead {lead+1}')
    axes[-1].set_xlabel('Time (samples)')
    fig.suptitle(f'Grad‑CAM for {class_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_integrated_gradients(signal, attributions, class_name, save_path):
    """绘制 Integrated Gradients 归因结果（每导联）"""
    C, L = signal.shape
    fig, axes = plt.subplots(C, 1, figsize=(12, C*1.5), sharex=True)
    if C == 1:
        axes = [axes]
    time_axis = np.arange(L)
    for lead, ax in enumerate(axes):
        ax.plot(time_axis, signal[lead], color='black', linewidth=0.5)
        ax.fill_between(time_axis, 0, attributions[lead], alpha=0.5, color='green')
        ax.set_ylabel(f'Lead {lead+1}')
    axes[-1].set_xlabel('Time (samples)')
    fig.suptitle(f'Integrated Gradients for {class_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def generate_report(sample_idx, true_labels, pred_labels, probs, thresholds,
                    gradcam_heatmaps, ig_attributions=None, save_dir='reports'):
    """生成文本报告和汇总图"""
    report_path = os.path.join(save_dir, f'interpret_sample_{sample_idx}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'样本索引: {sample_idx}\n')
        f.write(f'真实标签: {true_labels}\n')
        f.write(f'预测概率: {probs}\n')
        f.write(f'使用阈值: {thresholds}\n')
        f.write(f'预测标签: {pred_labels}\n\n')
        f.write('各类别 Grad‑CAM 高重要性时间区域（top 3 最高峰）:\n')
        for cls, hm in gradcam_heatmaps.items():
            if hm is None:
                continue
            peaks = np.argsort(hm)[-3:][::-1]  # 前3 峰值
            f.write(f'  {cls}: 最高重要性位置 (样本索引): {peaks.tolist()}\n')

        if ig_attributions is not None:
            f.write('\nIntegrated Gradients 平均导联重要性:\n')
            for lead in range(ig_attributions.shape[0]):
                f.write(f'  导联 {lead+1}: 平均归因 {ig_attributions[lead].mean():.5f}\n')
    print(f'可解释性报告已保存至 {report_path}')

# ------------------------- 主功能 -------------------------
def run_interpretation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(args.config, args.checkpoint, device)
    class_names = config['data']['class_names']
    data_dir = config['data']['processed_dir']

    # 加载最优阈值（可选）
    threshold_path = os.path.join(project_root, 'reports', 'optimal_thresholds.json')
    thresholds = None
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            thresholds = np.array(json.load(f)['thresholds'])

    # 加载样本
    signal_tensor, true_label = load_sample(data_dir, args.sample)
    input_tensor = signal_tensor.unsqueeze(0).to(device)  # (1, C, L)

    # 模型预测
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().squeeze(0).numpy()
    if thresholds is not None:
        preds = (probs >= thresholds).astype(int)
    else:
        preds = (probs >= 0.5).astype(int)

    print(f'样本 {args.sample} 预测概率: {probs.round(4)}')
    print(f'预测标签: {preds}')
    print(f'真实标签: {true_label}')

    # Grad‑CAM 准备
    target_layer = find_last_conv_layer(model)
    gradcam = GradCAM1D(model, target_layer)

    os.makedirs(args.output_dir, exist_ok=True)
    gradcam_heatmaps = {}
    signal_np = signal_tensor.numpy()  # (C, L)

    # 对每个预测为阳性的类别生成 Grad‑CAM
    positive_classes = [i for i, p in enumerate(preds) if p == 1]
    if not positive_classes:
        print("无阳性预测，仅对概率最高类别生成 Grad‑CAM")
        positive_classes = [probs.argmax()]

    for cls_idx in positive_classes:
        class_name = class_names[cls_idx]
        heatmap = gradcam.generate(input_tensor, class_idx=cls_idx)
        gradcam_heatmaps[class_name] = heatmap
        save_path = os.path.join(args.output_dir, f'sample{args.sample}_class{cls_idx}_gradcam.png')
        plot_gradcam(signal_np, heatmap, class_name, save_path)
        print(f'Grad‑CAM 图已保存: {save_path}')

    # Integrated Gradients（如果启用且已安装）
    ig_attributions = None
    if args.integrated and HAS_CAPTUM:
        # 对概率最高类别计算 IG
        top_class = probs.argmax()
        ig_attr = integrated_gradients_attribution(model, input_tensor, top_class, steps=50)
        if ig_attr is not None:
            ig_attributions = ig_attr
            save_path = os.path.join(args.output_dir, f'sample{args.sample}_class{top_class}_ig.png')
            plot_integrated_gradients(signal_np, ig_attr, class_names[top_class], save_path)
            print(f'Integrated Gradients 图已保存: {save_path}')

    # 生成报告
    generate_report(
        args.sample, true_label, preds, probs, thresholds,
        gradcam_heatmaps, ig_attributions, args.output_dir
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECG 模型可解释性分析')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pth')
    parser.add_argument('--sample', type=int, default=0, help='测试样本索引')
    parser.add_argument('--output_dir', type=str, default='reports/interpretations')
    parser.add_argument('--integrated', action='store_true',
                        help='同时计算 Integrated Gradients（需 captum）')
    args = parser.parse_args()
    run_interpretation(args)