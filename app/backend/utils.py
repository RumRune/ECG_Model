# app/backend/utils.py
"""
后端工具函数：配置加载、模型构建、阈值加载、信号预处理。
所有文件路径均相对于项目根目录（ECG_Model/）解析，可在任意位置运行。
"""
import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

# 项目根目录 = app/backend/../../（EGC_Model/）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from models.model_zoo.resnet1d import resnet18_1d, resnet34_1d


def _resolve(path: str) -> str:
    """相对路径基于项目根解析；绝对路径原样返回"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_config(config_path: str) -> dict:
    with open(_resolve(config_path), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model_weights(model, checkpoint_path: str, device: torch.device):
    state = torch.load(_resolve(checkpoint_path), map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def build_model(config: dict, device: torch.device) -> nn.Module:
    num_classes = config['data']['num_classes']
    model_name = config['model'].get('name', 'resnet18')
    dropout = config['model'].get('dropout', 0.2)
    if model_name == 'resnet34':
        model = resnet34_1d(in_channels=12, num_classes=num_classes, dropout=dropout)
    else:
        model = resnet18_1d(in_channels=12, num_classes=num_classes, dropout=dropout)
    return model


def load_thresholds() -> np.ndarray:
    """
    加载每类最优阈值。
    优先级：1. configs/optimized_thresholds.yaml
            2. reports/optimal_thresholds.json
            3. 全 0.5 兜底
    """
    candidates = [
        os.path.join(PROJECT_ROOT, 'configs', 'optimized_thresholds.yaml'),
        os.path.join(PROJECT_ROOT, 'reports', 'optimal_thresholds.json'),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        if path.endswith(('.yaml', '.yml')):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return np.array([float(data[str(i)]) for i in range(len(data))])
            return np.array(list(data), dtype=float)
        elif path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'thresholds' in data:
                return np.array(data['thresholds'])
    return np.full(10, 0.5)


def preprocess_signal(raw_signals: list, config: dict) -> np.ndarray:
    """
    将前端传入的二维列表转换为模型输入 (num_leads, signal_length)。
    假设数据已经过前处理（带通、重采样等），本函数只做长度检查和归一化。
    """
    signals = np.array(raw_signals, dtype=np.float32)
    if signals.ndim != 2:
        raise ValueError("输入信号应为二维数组 (导联数, 采样点)")
    num_leads, length = signals.shape
    expected_leads = config['data'].get('num_leads', 12)
    if num_leads != expected_leads:
        raise ValueError(f"导联数错误：期望 {expected_leads}，实际 {num_leads}")

    # 可选：标准化（每个导联减去均值除以标准差）
    for i in range(num_leads):
        mean = signals[i].mean()
        std = signals[i].std() + 1e-8
        signals[i] = (signals[i] - mean) / std

    # 若长度不一致，这里简单截断或补零（生产环境应做更严格检查）
    expected_len = config['data'].get('signal_length', 1000)
    if length != expected_len:
        print(f"警告：信号长度 {length} 与配置不一致 {expected_len}，将进行裁剪或补零")
        if length > expected_len:
            signals = signals[:, :expected_len]
        else:
            pad = np.zeros((num_leads, expected_len - length), dtype=np.float32)
            signals = np.concatenate([signals, pad], axis=1)

    return signals  # (leads, length)
