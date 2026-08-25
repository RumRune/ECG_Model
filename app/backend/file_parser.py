# app/backend/file_parser.py
"""
心电文件解析模块
================
解析 WFDB(.dat/.hea) 与 DICOM(.dcm) 心电图文件，
供 HTML 前端通过 /parse_wfdb、/parse_dicom 接口调用。

WFDB 解析为纯内存实现（numpy 直接解析 .dat 二进制），不依赖落盘临时文件，
因此无需 wfdb 库、无文件写入权限问题、更安全（上传文件不落盘）。

返回统一格式：
    {
        "signals":   [[...导联1...], [...导联2...], ...],   # (num_leads, n_samples)
        "lead_names":["I","II","V1",...],
        "fs":        float,                                 # 采样率
        "meta":      {key: value, ...}                      # 展示用元信息
    }
"""
import re
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------
# WFDB .hea 头文件解析
# ---------------------------------------------------------------
def _parse_hea(hea_text: str) -> Tuple[str, int, float, int, List[Dict]]:
    """
    解析 .hea 文本，返回:
        (record_name, n_sig, fs, n_samples, signal_specs)
    signal_specs: 每导联 [dat_file, fmt, gain, baseline, units, name]
    """
    lines = [l.strip() for l in hea_text.split("\n")
             if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("%")]
    if not lines:
        raise ValueError(".hea 文件内容为空，无法解析")

    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f".hea 首行格式错误: {lines[0]}")
    record_name = first[0]
    n_sig = int(first[1])
    fs = float(first[2])
    n_samples = int(first[3])

    signal_lines = lines[1:1 + n_sig]
    if len(signal_lines) < n_sig:
        raise ValueError(f".hea 声称 {n_sig} 个导联，但只有 {len(signal_lines)} 行信号描述")

    specs = []
    for line in signal_lines:
        tokens = line.split()
        if len(tokens) < 8:
            raise ValueError(f"信号行格式错误: {line}")
        dat_file = tokens[0]
        fmt = int(tokens[1])
        # 增益/基线/单位：形如 "1000.0(0)/mV" 或 "200.0" 或 "200/mV"
        gain_field = tokens[2]
        m = re.match(r'([\d.]+)(?:\(([-\d.]+)\))?(?:/(\w+))?', gain_field)
        gain = float(m.group(1)) if m else 200.0
        baseline = float(m.group(2)) if m and m.group(2) is not None else 0.0
        units = m.group(3) if m and m.group(3) else 'mV'
        # 导联名：WFDB 信号行标准字段位置（filename fmt gain baseline units adc_res adc_zero init checksum block_size sig_name）
        # PTB-XL 等常见格式省略部分字段，名字在最后一个 token；信号名不应是纯数字/已知字段名
        name = f"Lead_{len(specs)+1}"
        if len(tokens) >= 9:
            cand = tokens[-1]
            # 排除常见的非名字 token（数字、特殊值）
            if not re.fullmatch(r'[\d\-+.]+', cand) and cand not in ('mV', 'uV', 'm', '16', '0'):
                name = cand
        specs.append({
            "dat_file": dat_file, "fmt": fmt, "gain": gain,
            "baseline": baseline, "units": units, "name": name,
        })
    return record_name, n_sig, fs, n_samples, specs


# ---------------------------------------------------------------
# WFDB .dat 二进制解码（纯内存）
# ---------------------------------------------------------------
def _decode_16(raw: bytes, n_sig: int, n_samples: int) -> np.ndarray:
    """fmt 16: 16-bit 有符号整数，导联交错存储"""
    arr = np.frombuffer(raw, dtype='<i2')
    return arr.reshape(n_samples, n_sig).T.astype(np.float64)


def _decode_80(raw: bytes, n_sig: int, n_samples: int) -> np.ndarray:
    """fmt 80: 8-bit 无符号整数（偏移 128），导联交错存储"""
    arr = np.frombuffer(raw, dtype='u1').astype(np.float64) - 128.0
    return arr.reshape(n_samples, n_sig).T


def _decode_212(raw: bytes, n_sig: int, n_samples: int) -> np.ndarray:
    """fmt 212: 12-bit 有符号整数，3 字节存储 2 个样本，导联交错"""
    data = np.frombuffer(raw, dtype='u1')
    n_total = n_samples * n_sig
    # 每 3 字节存 2 个 12bit 值：byte0 | (byte1 & 0x0F)<<8 和 (byte1>>4) | byte2<<4
    needed = (n_total * 3 + 1) // 2
    if len(data) < needed:
        raise ValueError("fmt=212 数据长度不足")
    b0 = data[0::3].astype(np.int32)
    b1 = data[1::3].astype(np.int32)
    b2 = data[2::3].astype(np.int32)
    v0 = b0 | ((b1 & 0x0F) << 8)
    v1 = (b1 >> 4) | (b2 << 4)
    # 12-bit 符号扩展
    v0 = np.where(v0 & 0x800, v0 - 0x1000, v0)
    v1 = np.where(v1 & 0x800, v1 - 0x1000, v1)
    interleaved = np.empty(2 * len(v0), dtype=np.float64)
    interleaved[0::2] = v0
    interleaved[1::2] = v1
    interleaved = interleaved[:n_total]
    return interleaved.reshape(n_samples, n_sig).T


def _decode_24(raw: bytes, n_sig: int, n_samples: int) -> np.ndarray:
    """fmt 24: 24-bit 有符号整数，导联交错存储"""
    data = np.frombuffer(raw, dtype='u1')
    n_total = n_samples * n_sig
    needed = n_total * 3
    if len(data) < needed:
        raise ValueError("fmt=24 数据长度不足")
    b0 = data[0::3].astype(np.int32)
    b1 = data[1::3].astype(np.int32)
    b2 = data[2::3].astype(np.int32)
    vals = b0 | (b1 << 8) | (b2 << 16)
    vals = np.where(vals & 0x800000, vals - 0x1000000, vals)
    return vals[:n_total].reshape(n_samples, n_sig).T.astype(np.float64)


def _decode_32(raw: bytes, n_sig: int, n_samples: int) -> np.ndarray:
    """fmt 32: 32-bit 浮点，导联交错存储"""
    arr = np.frombuffer(raw, dtype='<f4')
    return arr.reshape(n_samples, n_sig).T.astype(np.float64)


def _decode_64(raw: bytes, n_sig: int, n_samples: int) -> np.ndarray:
    """fmt 64: 64-bit 浮点，导联交错存储"""
    arr = np.frombuffer(raw, dtype='<f8')
    return arr.reshape(n_samples, n_sig).T


_DECODERS = {
    16: _decode_16,
    80: _decode_80,
    212: _decode_212,
    24: _decode_24,
    32: _decode_32,
    64: _decode_64,
}


def _dat_to_signals(dat_bytes: bytes, specs: List[Dict], n_samples: int) -> np.ndarray:
    """按 .hea 描述把 .dat 二进制解码为 (n_sig, n_samples) 物理值"""
    n_sig = len(specs)
    fmt = specs[0]["fmt"]
    if any(s["fmt"] != fmt for s in specs):
        raise ValueError("多导联 fmt 不一致，暂不支持")

    if fmt not in _DECODERS:
        raise ValueError(f"不支持的 WFDB 数据格式 fmt={fmt}（支持 16/80/212/24/32/64）")

    digital = _DECODERS[fmt](dat_bytes, n_sig, n_samples)

    # 物理值 = (digital - baseline) / gain
    signals = np.empty_like(digital)
    for i, spec in enumerate(specs):
        signals[i] = (digital[i] - spec["baseline"]) / spec["gain"]
    return signals


# ---------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------
def parse_wfdb(dat_bytes: bytes, hea_bytes: bytes) -> Dict:
    """纯内存解析 WFDB：读取 .hea 描述 + 解码 .dat 二进制，无需落盘"""
    if not hea_bytes:
        raise ValueError(".hea 头文件为空")
    if not dat_bytes:
        raise ValueError(".dat 数据文件为空")

    hea_text = hea_bytes.decode("utf-8", errors="replace")
    record_name, n_sig, fs, n_samples, specs = _parse_hea(hea_text)

    # 校验 .dat 中引用的文件与本地上传一致（多文件场景给出明确提示）
    dat_files = [s["dat_file"] for s in specs]
    unique_dats = list(dict.fromkeys(dat_files))
    if len(unique_dats) > 1:
        raise ValueError(
            f"该 .hea 引用了多个数据文件 {unique_dats}，"
            "当前仅支持单个 .dat 文件，请分别上传。"
        )

    signals = _dat_to_signals(dat_bytes, specs, n_samples)

    lead_names = [s["name"] for s in specs]
    meta = {
        "采样率 (Hz)": fs,
        "导联数": n_sig,
        "采样点数": n_samples,
        "时长 (秒)": round(n_samples / fs, 2) if fs else "未知",
        "数据格式": f"fmt={specs[0]['fmt']}",
    }

    return {
        "signals": np.asarray(signals, dtype=float).tolist(),
        "lead_names": lead_names,
        "fs": float(fs),
        "meta": meta,
    }


def parse_dicom(file_bytes: bytes) -> Dict:
    """从上传的 .dcm 字节内容解析 DICOM 心电图（pydicom 从内存读取，无需落盘）"""
    import pydicom

    if not file_bytes:
        raise ValueError("DICOM 文件为空")

    ds = pydicom.dcmread(BytesIO(file_bytes))

    # ----- 1. 优先提取波形数据 -----
    if hasattr(ds, 'WaveformSequence') and ds.WaveformSequence:
        waveform = ds.WaveformSequence[0]
    else:
        sop_class = getattr(ds, 'SOPClassUID', '未知')
        modality = getattr(ds, 'Modality', '未知')
        if hasattr(ds, 'EncapsulatedDocument'):
            raise ValueError("该 DICOM 是封装的 PDF 报告，不包含原始波形数据。")
        if hasattr(ds, 'PixelData') or modality in ('SC', 'OT'):
            raise ValueError("该 DICOM 是图像截图（Secondary Capture），不含原始波形。")
        if sop_class == '1.2.840.10008.5.1.4.1.1.9.3.1':
            raise ValueError("该 DICOM 是结构化报告，不含原始波形。")
        if sop_class == '1.2.840.10008.5.1.4.1.1.88.33':
            raise ValueError("该 DICOM 是 Comprehensive SR（综合结构化报告），不含波形。")
        raise ValueError(
            f"该 DICOM 文件中未找到心电图波形数据（WaveformSequence 缺失）。"
            f"SOPClassUID={sop_class}, Modality={modality}"
        )

    # ----- 2. 波形数据解析 -----
    n_channels = int(waveform.NumberOfWaveformChannels)
    n_samples = int(waveform.NumberOfWaveformSamples)
    raw = np.array(waveform.WaveformData, dtype=np.float32)
    signals = raw.reshape(n_channels, n_samples)

    channel_defs = waveform.ChannelDefinitionSequence
    lead_names = []
    for ch in channel_defs:
        code = ch.get('ChannelSourceSequence', None)
        if code:
            lead_names.append(
                str(code[0].CodeMeaning) if code[0].CodeMeaning else f"通道{ch.ChannelNumber}"
            )
        else:
            lead_names.append(f"通道{ch.ChannelNumber}")

    fs = float(waveform.SamplingFrequency)

    meta = {
        "采样率 (Hz)": fs,
        "导联数": n_channels,
        "采样点数": n_samples,
        "时长 (秒)": round(n_samples / fs, 2) if fs else "未知",
        "患者ID": getattr(ds, 'PatientID', '未知'),
        "检查日期": getattr(ds, 'StudyDate', '未知'),
        "设备厂商": getattr(ds, 'Manufacturer', '未知'),
    }

    return {
        "signals": np.asarray(signals, dtype=float).tolist(),
        "lead_names": lead_names,
        "fs": fs,
        "meta": meta,
    }
