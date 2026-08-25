"""
基线模型：传统机器学习方法作为深度学习性能下限
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, classification_report, hamming_loss
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Optional, List, Union


def extract_handcrafted_features(signals: np.ndarray, sampling_rate: int = 100):
    """
    从 ECG 信号中提取手工特征（时域 + 简单频域）
    Args:
        signals: (N, leads, time)
        sampling_rate: Hz
    Returns:
        features: (N, num_features)
    """
    N, num_leads, seq_len = signals.shape
    feature_list = []
    
    for i in range(N):
        lead_features = []
        for lead in range(num_leads):
            sig = signals[i, lead, :]
            
            # 时域特征
            mean_val = float(np.mean(sig))
            std_val = float(np.std(sig))
            max_val = float(np.max(sig))
            min_val = float(np.min(sig))
            range_val = max_val - min_val
            rms = float(np.sqrt(np.mean(sig ** 2)))
            
            # 差分统计（近似一阶导数）
            diff = np.diff(sig)
            diff_mean = float(np.mean(np.abs(diff)))
            diff_std = float(np.std(diff))
            
            # 过零点率
            zero_crossings = float(np.sum(np.diff(np.signbit(sig))) / seq_len)
            
            lead_features.extend([
                mean_val, std_val, max_val, min_val, range_val, rms,
                diff_mean, diff_std, zero_crossings
            ])
        
        # 导联间相关性（II 与 V5，索引 1 和 10）
        if num_leads >= 11:
            corr_val = np.corrcoef(signals[i, 1], signals[i, 10])[0, 1]
            lead_features.append(float(corr_val) if not np.isnan(corr_val) else 0.0)
        else:
            lead_features.append(0.0)
        
        feature_list.append(lead_features)
    
    return np.array(feature_list, dtype=np.float32)


class RandomForestBaseline:
    """随机森林基线（多标签）"""
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 20, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.scaler: StandardScaler = StandardScaler()
        # 使用 Any 类型避免静态检查器报错
        self.model: MultiOutputClassifier = None  # type: ignore[assignment]
    
    def fit(self, signals: np.ndarray, labels: np.ndarray, sampling_rate: int = 100):
        """
        Args:
            signals: (N, leads, time)
            labels:  (N, num_classes) 多标签二值矩阵
        """
        print("提取手工特征...")
        features = extract_handcrafted_features(signals, sampling_rate)
        print(f"特征维度: {features.shape}")
        
        print("标准化...")
        features = self.scaler.fit_transform(features)
        
        print("训练随机森林...")
        self.model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        )
        self.model.fit(features, labels)
        print("训练完成")
    
    def _extract_features(self, signals: np.ndarray, sampling_rate: int = 100) -> np.ndarray:
        """内部辅助方法：提取并标准化特征"""
        features = extract_handcrafted_features(signals, sampling_rate)
        return self.scaler.transform(features)
    
    def predict(self, signals: np.ndarray, sampling_rate: int = 100) -> np.ndarray:
        """返回二值预测 (N, num_classes)"""
        features = self._extract_features(signals, sampling_rate)
        # MultiOutputClassifier.predict 返回 numpy 数组
        return np.array(self.model.predict(features))  # type: ignore[union-attr]
    
    def predict_proba(self, signals: np.ndarray, sampling_rate: int = 100):
        """返回概率预测——MultiOutputClassifier 返回列表，每个元素是 (N, 2)"""
        features = self._extract_features(signals, sampling_rate)
        proba_list = self.model.predict_proba(features)  # type: ignore[union-attr]
        # 提取正类概率 → (N, num_classes)
        proba_pos = np.column_stack([p[:, 1] for p in proba_list])
        return proba_pos.astype(np.float64)
    
    def evaluate(self, signals: np.ndarray, labels: np.ndarray,
                 sampling_rate: int = 100) -> dict:
        """评估模型并返回指标字典"""
        preds = self.predict(signals, sampling_rate)
        
        # 计算指标
        ham_loss = float(hamming_loss(labels, preds))
        f1_macro = float(f1_score(labels, preds, average='macro', zero_division=0))
        f1_micro = float(f1_score(labels, preds, average='micro', zero_division=0))
        
        # 每个类别的 F1（确保返回 list）
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        if isinstance(per_class_f1, (float, np.floating)):
            per_class_f1 = [float(per_class_f1)]
        else:
            per_class_f1 = [float(v) for v in per_class_f1] # type: ignore
        
        results = {
            'hamming_loss': ham_loss,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'per_class_f1': dict(enumerate(per_class_f1)),
        }
        
        return results
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        print(f"模型已保存至: {path}")
    
    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        print(f"模型已从 {path} 加载")


if __name__ == "__main__":
    import os
    import sys

    # ---------- 自动定位项目根目录 ----------
    # 方法：从当前脚本所在目录向上找一层（src/ 的父目录即项目根）
    script_dir = os.path.dirname(os.path.abspath(__file__))   # .../src
    project_root = os.path.dirname(script_dir)                # .../ECG_Model
    os.chdir(project_root)                                    # 切换到项目根
    print(f"项目根目录: {project_root}")
    # ----------------------------------------

    data_dir = "data/processed"

    train_signals = np.load(os.path.join(data_dir, 'train', 'signals.npy'))
    train_labels  = np.load(os.path.join(data_dir, 'labels', 'train_multilabel.npy'))

    print(f"训练信号: {train_signals.shape}")
    print(f"训练标签: {train_labels.shape}")
    print(f"标签分布 (正样本数/总样本数):")
    for i in range(train_labels.shape[1]):
        pos_count = train_labels[:, i].sum()
        print(f"  类别 {i}: {int(pos_count)} / {len(train_labels)} ({pos_count/len(train_labels):.2%})")

    # 训练
    baseline = RandomForestBaseline(n_estimators=50, max_depth=10)
    baseline.fit(train_signals, train_labels)

    # 评估
    results = baseline.evaluate(train_signals, train_labels)
    print("\n===== 训练集评估结果 =====")
    print(f"  Hamming Loss: {results['hamming_loss']:.4f}")
    print(f"  F1 Macro:     {results['f1_macro']:.4f}")
    print(f"  F1 Micro:     {results['f1_micro']:.4f}")
    print(f"  各类别 F1:")
    for cls_idx, f1_val in results['per_class_f1'].items():
        print(f"    类别 {cls_idx}: {f1_val:.4f}")

    baseline.save("models/checkpoints/baseline_rf.pkl")