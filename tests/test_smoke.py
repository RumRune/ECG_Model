"""
ECG 模型基础冒烟测试
用法：python -m pytest tests/ -v   或   python tests/test_smoke.py
覆盖：配置加载、模型构建/前向、标签加载、阈值加载、信号预处理
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch


def test_config_loads():
    from app.backend.utils import load_config
    config = load_config('configs/config.yaml')
    assert config['data']['num_classes'] == 10
    assert len(config['data']['class_names']) == 10
    assert 'NORM' in config['data']['class_names']
    assert 'SR' not in config['data']['class_names'], '旧类别 SR 不应再存在'


def test_model_forward():
    from models.model_zoo.resnet1d import resnet18_1d
    model = resnet18_1d(in_channels=12, num_classes=10)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 12, 1000))
    assert out.shape == (2, 10)


def test_labels_loaded():
    labels = np.load(os.path.join(PROJECT_ROOT, 'data', 'processed', 'labels', 'train_multilabel.npy'))
    assert labels.shape == (17181, 10)
    assert set(np.unique(labels)).issubset({0.0, 1.0})


def test_thresholds_fallback():
    from app.backend.utils import load_thresholds
    thr = load_thresholds()
    assert len(thr) == 10


def test_preprocess_signal():
    from app.backend.utils import load_config, preprocess_signal
    config = load_config('configs/config.yaml')
    sig = np.random.randn(12, 5000).astype(np.float32)
    proc = preprocess_signal(sig, config)
    assert proc.shape == (12, 1000)
    assert proc.dtype == np.float32


if __name__ == '__main__':
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'[PASS] {name}')
            except Exception:
                failed += 1
                print(f'[FAIL] {name}')
                traceback.print_exc()
    sys.exit(1 if failed else 0)
