"""
数据加载与预处理管线
封装 signals.npy + multilabel.npy → PyTorch Dataset
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ECGDataset(Dataset):
    def __init__(self, data_dir: str, split: str, use_multilabel: bool = True):
        split_dir = os.path.join(data_dir, split)
        signals_path = os.path.join(split_dir, 'signals.npy')
        if not os.path.exists(signals_path):
            raise FileNotFoundError(f"信号文件缺失: {signals_path}")
        self.signals = np.load(signals_path).astype(np.float32)

        if use_multilabel:
            multilabel_path = os.path.join(data_dir, 'labels', f'{split}_multilabel.npy')
            if not os.path.exists(multilabel_path):
                raise FileNotFoundError(f"多标签文件缺失: {multilabel_path}")
            self.labels = np.load(multilabel_path).astype(np.float32)
        else:
            raise NotImplementedError("暂仅支持多标签模式")

        assert len(self.signals) == len(self.labels), \
            f"信号({len(self.signals)})与标签({len(self.labels)})数量不一致"

        self.num_classes = self.labels.shape[1]
        self.use_multilabel = use_multilabel

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx])   # (12, 1000)
        label = torch.from_numpy(self.labels[idx])     # (10,)
        return signal, label


def create_dataloaders(data_dir: str, batch_size: int = 64,
                       num_workers: int = 4, use_multilabel: bool = True):
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        dataset = ECGDataset(data_dir, split, use_multilabel=use_multilabel)
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"{split}: {len(dataset)} 样本, {dataset.num_classes} 类")
    return dataloaders


if __name__ == "__main__":
    # 快速测试（路径相对项目根目录）
    import os
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataloaders = create_dataloaders(os.path.join(root, 'data', 'processed'), batch_size=32)
    for x, y in dataloaders['train']:
        print(f"训练批次: 信号 {x.shape}, 标签 {y.shape}")
        break
