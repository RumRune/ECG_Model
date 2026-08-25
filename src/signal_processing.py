# src/signal_processing.py
# type: ignore
"""
ECG 信号处理模块
包含：带通滤波、陷波、重采样、标准化、数据增强等。
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample
from scipy import interpolate
import warnings

def butter_bandpass(lowcut, highcut, fs, order=4):
    """设计巴特沃斯带通滤波器"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_highpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a

def notch_filter(freq, q, fs):
    """设计陷波滤波器（去除工频干扰）"""
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, q)
    return b, a

def apply_filter(data, b, a):
    """对多导联信号逐导联应用零相位滤波器"""
    if data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        return np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=-1, arr=data)

def preprocess_ecg(raw_signal, fs_orig, target_fs=100, lowcut=0.5, highcut=45,
                   notch_freq=50, notch_q=30, normalize=True, clip_seconds=10):
    """
    完整预处理管线：
    1. 带通滤波 (0.5-45 Hz)
    2. 陷波 (50 Hz)
    3. 重采样到目标频率
    4. 截取固定长度
    5. Z-score 标准化（按导联）
    参数:
        raw_signal: (n_leads, n_samples) 原始ECG
        fs_orig: 原始采样率
        target_fs: 目标采样率
        lowcut, highcut: 带通截止频率
        notch_freq, notch_q: 陷波参数
        normalize: 是否标准化
        clip_seconds: 截取时长（秒）
    返回:
        (n_leads, target_length) 处理后的信号
    """
    # 1. 带通滤波
    b_bp, a_bp = butter_bandpass(lowcut, highcut, fs_orig, order=4)
    filtered = apply_filter(raw_signal, b_bp, a_bp)

    # 2. 陷波
    b_notch, a_notch = notch_filter(notch_freq, notch_q, fs_orig)
    filtered = apply_filter(filtered, b_notch, a_notch)

    # 3. 重采样
    n_samples_new = int(filtered.shape[-1] * target_fs / fs_orig)
    resampled = resample(filtered, n_samples_new, axis=-1)

    # 4. 截取固定长度
    target_len = int(target_fs * clip_seconds)
    if resampled.shape[-1] > target_len:
        resampled = resampled[..., :target_len]
    elif resampled.shape[-1] < target_len:
        # 补零到目标长度
        pad_width = target_len - resampled.shape[-1]
        resampled = np.pad(resampled, ((0, 0), (0, pad_width)), mode='constant')

    # 5. Z-score 标准化（每导联单独）
    if normalize:
        mean = np.mean(resampled, axis=-1, keepdims=True)
        std = np.std(resampled, axis=-1, keepdims=True)
        std[std == 0] = 1.0  # 避免除以零
        resampled = (resampled - mean) / std

    return resampled.astype(np.float32)

def augment_signal(signal, shift_max=50, stretch_range=(0.9, 1.1), noise_std=0.01):
    """
    数据增强：随机平移 + 时间拉伸 + 微小噪声
    参数:
        signal: (n_leads, n_samples)
    返回:
        增强后的信号 (n_leads, n_samples)
    """
    n_leads, n_samples = signal.shape
    augmented = signal.copy()

    # 随机平移
    shift = np.random.randint(-shift_max, shift_max)
    if shift > 0:
        augmented[:, shift:] = signal[:, :-shift]
        augmented[:, :shift] = 0
    elif shift < 0:
        augmented[:, :shift] = signal[:, -shift:]
        augmented[:, shift:] = 0

    # 时间拉伸
    scale = np.random.uniform(*stretch_range)
    new_time = np.linspace(0, 1, int(n_samples * scale))
    old_time = np.linspace(0, 1, n_samples)
    # 对每导联分别插值
    aug_stretched = np.zeros_like(augmented)
    for lead in range(n_leads):
        tck = interpolate.splrep(old_time, augmented[lead], s=0)
        aug_stretched[lead] = interpolate.splev(new_time, tck, der=0)
    # 重采样回原长度
    if aug_stretched.shape[1] > n_samples:
        aug_final = aug_stretched[:, :n_samples]
    else:
        aug_final = np.pad(aug_stretched, ((0,0),(0, n_samples - aug_stretched.shape[1])), mode='constant')
    augmented = aug_final

    # 加微小噪声
    noise = np.random.normal(0, noise_std, augmented.shape)
    augmented += noise

    return augmented.astype(np.float32)

def extract_stat_features(signal, fs):
    """
    提取统计特征，用于基线模型。
    返回 dict 包含常用时域、频域特征。
    """
    features = {}
    n_leads = signal.shape[0]
    for lead in range(n_leads):
        x = signal[lead]
        # 时域特征
        features[f'lead{lead}_mean'] = np.mean(x)
        features[f'lead{lead}_std'] = np.std(x)
        features[f'lead{lead}_min'] = np.min(x)
        features[f'lead{lead}_max'] = np.max(x)
        features[f'lead{lead}_pp'] = np.max(x) - np.min(x)
        # 频域特征 (PSD)
        fft_vals = np.abs(np.fft.rfft(x))
        fft_freq = np.fft.rfftfreq(len(x), 1/fs)
        features[f'lead{lead}_psd_mean'] = np.mean(fft_vals)
        features[f'lead{lead}_psd_max'] = np.max(fft_vals)
        features[f'lead{lead}_psd_median'] = np.median(fft_vals)
    # 导联间相关性
    corr_matrix = np.corrcoef(signal)
    for i in range(n_leads):
        for j in range(i+1, n_leads):
            features[f'corr_{i}_{j}'] = corr_matrix[i, j]
    return features