# app/backend/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union

class ECGInput(BaseModel):
    """单条原始心电信号的输入格式"""
    signals: List[List[float]] = Field(
        ...,
        description="原始心电信号，二维列表：第一维是导联，第二维是采样点数值（单位通常为mV）。例如12导联5000点数据应为 12×5000 的列表。",
        example=[
            [0.1, 0.2, 0.15, 0.3, 0.5],  # 示例用短信号，实际使用时替换为完整长度
            [0.0, 0.1, 0.05, 0.2, 0.4]
        ]
    )

class PredictionResult(BaseModel):
    """单条诊断结果"""
    probabilities: List[float] = Field(
        ...,
        description="每类疾病的预测概率，顺序与 class_names 对应"
    )
    pred_labels: List[int] = Field(
        ...,
        description="二值预测结果，1 表示阳性（有病），0 表示阴性（无病）"
    )
    class_names: List[str] = Field(
        ...,
        description="所有疾病类别名称（中文）"
    )
    threshold_used: List[float] = Field(
        ...,
        description="各类别判定阳性的阈值，概率超过该值即判为1"
    )
    gradcam_heatmaps: Optional[Dict[str, List[float]]] = Field(
        None,
        description="可选的 Grad‑CAM 热力图，键为疾病名称，值为一维列表（长度同信号长度），数值表示该位置对预测的贡献度"
    )
    message: Optional[str] = Field(
        None,
        description="附加信息，预处理出错时会显示错误原因"
    )

class BatchPredictionResponse(BaseModel):
    """批量诊断响应体"""
    results: List[PredictionResult] = Field(
        ...,
        description="每条样本的诊断结果列表"
    )
    summary: Dict[str, Union[int, Dict[str, float]]] = Field(
        ...,
        description="汇总统计，包含总样本数 (total_samples) 和各类别阳性比例 (positive_rate_per_class)"
    )