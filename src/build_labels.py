"""
src/build_labels.py — 多标签构建脚本（修复版）
==================================================
修复内容：
1. 编码 bug：PTB-XL 的 scp_codes 中 value<=0 表示"明确排除/不存在"，
   旧代码用 `code in scp_dict` 判断，导致 0 概率的码被误标为正例
   （SR 类正例率被虚高到 77%）。
2. 类别选择 bug：旧代码按"出现次数（含 0 概率）"选 top-10，
   导致 SR / ABQRS 这类从不以 >0 出现的节律码入选、成为全零类，
   而真正有意义的 IRBBB / PVC / IVCD 被挤出。

新规则：
- 类别：按 scp_codes 中 value>0 的出现次数取 top-10（诊断类），
  顺序与 configs/config.yaml 的 class_names 保持一致。
- 编码：仅当码存在且 value > 0 时计为正例。
- 划分：沿用官方 strat_fold（train=1-8, val=9, test=10），
  从 data/processed/*/labels.csv 读取原始 scp_codes 重新编码。

用法：
  python src/build_labels.py
输出：
  data/processed/labels/{split}_multilabel.npy
  data/processed/labels/{split}_labels_parsed.csv（含 scp_dict 列）
"""
import os
import ast
import json
import yaml
import numpy as np
import pandas as pd
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'ptb-xl_1.0.2')
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')


def parse_scp_codes(scp_str):
    """将 scp_codes 字符串安全解析为 dict"""
    if isinstance(scp_str, str):
        try:
            return ast.literal_eval(scp_str)
        except (SyntaxError, ValueError):
            return {}
    elif isinstance(scp_str, dict):
        return scp_str
    return {}


def load_all_scp_dicts():
    """加载所有划分的 scp_dict"""
    labels_dfs = {}
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(PROCESSED_DIR, split, 'labels.csv')
        df = pd.read_csv(csv_path)
        df['scp_dict'] = df['scp_codes'].apply(parse_scp_codes)
        labels_dfs[split] = df
    return labels_dfs


def select_classes(labels_dfs, top_n=10):
    """按 value>0 出现次数选择 top-N 类别（保持频率降序）"""
    counter = Counter()
    for df in labels_dfs.values():
        for d in df['scp_dict']:
            for code, prob in d.items():
                if prob > 0:
                    counter[code] += 1
    return [code for code, _ in counter.most_common(top_n)]


def encode_multilabel(scp_dict, code_list):
    """正确编码：仅当 value > 0 时计为正例"""
    return [1 if scp_dict.get(code, 0) > 0 else 0 for code in code_list]


def main():
    labels_dfs = load_all_scp_dicts()

    # 1. 选择类别（与 config 对齐）
    config = yaml.safe_load(open(CONFIG_PATH, encoding='utf-8'))
    cfg_names = config['data']['class_names']
    top_codes = select_classes(labels_dfs, top_n=len(cfg_names))
    print('按 value>0 统计的 top-%d 类别:' % len(cfg_names))
    print('  ', top_codes)

    if top_codes != cfg_names:
        print('⚠️  config.yaml 的 class_names 与统计结果不一致，以 config 为准：')
        print('  ', cfg_names)
        # 校验 config 中的类都确实有正例
        counter = Counter()
        for df in labels_dfs.values():
            for d in df['scp_dict']:
                for code, prob in d.items():
                    if prob > 0:
                        counter[code] += 1
        for c in cfg_names:
            n = counter.get(c, 0)
            flag = '⚠️ 零正例' if n == 0 else ''
            print('    %-6s %6d %s' % (c, n, flag))
    # 最终以 config 声明的类为准（避免模型/前端错位）
    final_codes = cfg_names

    # 2. 重新编码并保存
    out_dir = os.path.join(PROCESSED_DIR, 'labels')
    os.makedirs(out_dir, exist_ok=True)

    for split, df in labels_dfs.items():
        matrix = np.array([encode_multilabel(d, final_codes) for d in df['scp_dict']])
        np.save(os.path.join(out_dir, f'{split}_multilabel.npy'), matrix, allow_pickle=False)
        print(f'{split}: {matrix.shape} 正例数={matrix.sum(axis=0).tolist()}')

        cols_to_save = ['scp_codes', 'strat_fold', 'report', 'diagnostic_class']
        available = [c for c in cols_to_save if c in df.columns]
        df_save = df[available + ['scp_dict']]
        df_save.to_csv(os.path.join(out_dir, f'{split}_labels_parsed.csv'), index=False)

    print('\n✅ 标签已重新生成至', out_dir)


if __name__ == '__main__':
    main()
