"""
PTB-XL 数据预处理脚本（复现 notebook 02，独立可运行）
功能：
- 加载原始数据
- 清洗异常值
- 按官方 strat_fold 划分训练/验证/测试集
- 对每条记录应用信号处理流水线（带通→陷波→重采样→裁剪/补齐→Z-score）
- 保存处理后的信号并生成元数据

注意：
- 多标签矩阵的生成见 src/build_labels.py（本脚本只负责信号与原始标签列）。
- 所有路径基于项目根目录（ECG_Model/），可在任意位置运行。
用法：
  python notebooks/preprocess.py \
      --csv data/raw/ptb-xl_1.0.2/ptbxl_database.csv \
      --data_dir data/raw/ptb-xl_1.0.2 \
      --output_dir data/processed
"""
import os
import sys
import json
import ast
import argparse
import time
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# ---------------- 信号处理（与 notebook 02 一致） ----------------
def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=500, order=4):
    """零相位带通滤波（Butterworth）"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', output='ba')
    return filtfilt(b, a, data, axis=-1)


def notch_filter(data, freq=50.0, fs=500, quality=30):
    """陷波滤波器（去除工频干扰）"""
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data, axis=-1)


def resample(data, orig_fs, target_fs=100):
    """重采样到目标频率（等价于 resample_poly up=10/down=2 对 500→100）"""
    up, down = 10, int(10 / (orig_fs / target_fs))
    return resample_poly(data, up=up, down=down, axis=-1)


def zscore_normalize(data):
    """按导联进行 Z-score 标准化"""
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True) + 1e-8
    return (data - mean) / std


# ---------------- 数据处理 ----------------
def load_database(csv_path):
    """加载 ptbxl_database.csv 并执行初步清洗"""
    df = pd.read_csv(csv_path)
    df = df[(df['age'] >= 0) & (df['age'] <= 120)].copy()
    return df


def split_dataset(df):
    """按官方 strat_fold 划分数据集"""
    train = df[df['strat_fold'].isin(range(1, 9))]
    val = df[df['strat_fold'] == 9]
    test = df[df['strat_fold'] == 10]
    return train, val, test


def process_record(data_path, target_fs=100, duration=10):
    """
    加载原始 WFDB 记录 -> 滤波 -> 重采样 -> 裁剪/补齐 -> Z-score
    返回: shape (12, target_length)
    """
    record = wfdb.rdrecord(data_path)
    signals = record.p_signal.T  # (12, N)
    orig_fs = int(record.fs)

    # 滤波
    signals = bandpass_filter(signals, fs=orig_fs)
    signals = notch_filter(signals, fs=orig_fs)

    # 重采样
    signals = resample(signals, orig_fs, target_fs)

    # 截取或补齐到固定长度
    target_len = target_fs * duration
    if signals.shape[1] >= target_len:
        signals = signals[:, :target_len]
    else:
        pad_width = ((0, 0), (0, target_len - signals.shape[1]))
        signals = np.pad(signals, pad_width, mode='constant')

    # Z-score 归一化
    signals = zscore_normalize(signals)
    return signals.astype(np.float32)


def prepare_labels(df, label_col='scp_codes'):
    """
    将 SCP 编码转为多标签二值矩阵（仅示例，实际标签生成见 src/build_labels.py）
    """
    class_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
    labels = np.zeros((len(df), len(class_map)), dtype=int)

    for i, row in df.iterrows():
        try:
            scp_dict = ast.literal_eval(row[label_col])
        except (SyntaxError, ValueError):
            scp_dict = {}
        for code, prob in scp_dict.items():
            if code in class_map and prob > 0.5:  # 简单阈值
                labels[i, class_map[code]] = 1
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to ptbxl_database.csv (相对项目根或绝对路径)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to raw PTB-XL directory (containing records100/ etc.)')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory to save processed data (相对项目根或绝对路径)')
    parser.add_argument('--target_fs', type=int, default=100)
    parser.add_argument('--duration', type=int, default=10,
                        help='Fixed recording length in seconds')
    args = parser.parse_args()

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    csv_path = resolve(args.csv)
    data_dir = resolve(args.data_dir)
    output_dir = resolve(args.output_dir)

    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载与清洗
    print("Loading database...")
    df = load_database(csv_path)
    print(f"After cleaning: {len(df)} records")

    # 2. 数据集划分
    train_df, val_df, test_df = split_dataset(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. 处理并保存每个子集
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        signals_list = []
        labels_list = []

        print(f"\nProcessing {split_name} split...")
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df)):
            rel_path = row['filename_lr']  # 或 filename_hr
            record_path = os.path.join(data_dir, rel_path)
            if not os.path.exists(record_path + '.dat'):
                print(f"Warning: Record not found: {record_path}.dat, skipping (ecg_id={row['ecg_id']})")
                continue

            try:
                signal = process_record(record_path,
                                        target_fs=args.target_fs,
                                        duration=args.duration)
            except Exception as e:
                print(f"Error processing ECG {row['ecg_id']}: {e}")
                continue

            signals_list.append(signal)
            labels_list.append(row[['scp_codes', 'diagnostic_class']])

        signals_array = np.stack(signals_list)  # (N, 12, T)
        np.save(os.path.join(split_dir, 'signals.npy'), signals_array)

        labels_df = pd.DataFrame(labels_list)
        labels_df.to_csv(os.path.join(split_dir, 'labels.csv'), index=False)

        print(f"Saved {len(signals_array)} signals and labels for {split_name}")

    # 4. 生成元数据
    metadata = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source_csv': csv_path,
        'target_fs': args.target_fs,
        'duration_sec': args.duration,
        'filtering': {
            'bandpass': [0.5, 45],
            'notch': 50,
            'normalization': 'zscore_per_lead'
        },
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'age_cleaning': 'keep only 0-120'
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nPreprocessing complete in {elapsed:.1f}s. Metadata saved.")
    print("提示：多标签矩阵请运行 python src/build_labels.py 生成。")


if __name__ == '__main__':
    main()
