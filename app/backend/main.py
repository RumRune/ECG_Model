# app/backend/main.py
import os
import sys

# ===== 路径设置：必须放在所有项目模块导入之前！ =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
# 项目根目录（ECG_Model/）已在 sys.path 最前面，后续均用相对根目录的路径

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn

from app.backend.schemas import ECGInput, PredictionResult, BatchPredictionResponse
from app.backend.utils import (
    load_config, build_model, load_model_weights, load_thresholds, preprocess_signal
)
from app.backend.file_parser import parse_wfdb, parse_dicom
from src.interpret import GradCAM1D, find_last_conv_layer


# ---------- 静态前端目录 ----------
STATIC_DIR = os.path.join(PROJECT_ROOT, 'app', 'static')


# ---------- 自定义中文 Swagger UI ----------
ZH_TRANSLATION_SCRIPT = """
<script>
setTimeout(function() {
    const map = {
        // 界面按钮/标签
        "Server": "服务器",
        "Authorize": "授权",
        "Try it out": "试用",
        "Cancel": "取消",
        "Execute": "执行",
        "Responses": "响应",
        "Request body": "请求体",
        "Parameters": "参数",
        "No parameters": "无参数",
        "Example Value": "示例值",
        "Schema": "结构",
        "Model": "模型",
        "Send Request": "发送请求",
        "Clear": "清除",
        "Download": "下载",
        "Close": "关闭",
        "Show/Hide": "显示/隐藏",
        "Expand all": "展开全部",
        "Collapse all": "折叠全部",
        "Filter": "筛选",
        "Search": "搜索",
        "available": "可用",
        "required": "必填",
        "Deprecated": "已弃用",
        "Default": "默认",
        // 数据类型
        "string": "字符串",
        "integer": "整数",
        "number": "数字",
        "array": "数组",
        "object": "对象",
        "boolean": "布尔值",
        "null": "空值",
        // 参数位置
        "body": "请求体",
        "header": "请求头",
        "path": "路径",
        "query": "查询参数",
        "cookie": "Cookie",
        // 表单/操作
        "Description": "描述",
        "Type": "类型",
        "Value": "值",
        "Submit": "提交",
        "Remove": "移除",
        "Add item": "添加一项",
        // 授权相关
        "Authorization": "授权",
        "Logout": "退出",
        "Scopes": "权限范围",
        "Bearer": "Bearer 令牌",
        "API key": "API 密钥",
        "Basic HTTP": "基本 HTTP 认证",
        "OAuth2": "OAuth2 认证",
        // 其他常见文本
        "Select a definition": "选择一个定义",
        "Media type": "媒体类型",
        "Example": "示例",
    };

    function translate(node) {
        if (!node) return;
        if (node.nodeType === 3) {
            const t = node.textContent.trim();
            if (map[t]) node.textContent = node.textContent.replace(t, map[t]);
        } else if (node.nodeType === 1) {
            if (node.placeholder && map[node.placeholder]) node.placeholder = map[node.placeholder];
            // 翻译按钮内的文本
            if (node.tagName === 'BUTTON' && node.textContent.trim() in map) {
                node.textContent = map[node.textContent.trim()];
            }
            node.childNodes.forEach(translate);
        }
    }

    const el = document.getElementById('swagger-ui');
    translate(el);
    // 持续翻译动态插入的元素（如响应区）
    setInterval(() => translate(el), 1500);
}, 1200);
</script>
"""

app = FastAPI(
    title="心电图多标签诊断 API",
    version="1.0",
    docs_url=None,   # 禁用默认 /docs，使用自定义中文版
    redoc_url=None
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="心电图多标签诊断 API（中文文档）",
        swagger_favicon_url=""
    )
    content = html.body.decode()
    content = content.replace("</body>", ZH_TRANSLATION_SCRIPT + "</body>")
    return HTMLResponse(content=content)


# ---------- 全局变量：模型、配置、阈值 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = load_config('configs/config.yaml')
model = build_model(config, device)
model = load_model_weights(model, 'models/checkpoints/best_model.pth', device)
thresholds = load_thresholds()
class_names = config['data']['class_names']

# 初始化 Grad‑CAM（可选的可视化解释）
target_layer = find_last_conv_layer(model)
gradcam = GradCAM1D(model, target_layer)


# ---------- API 端点 ----------
@app.get("/", include_in_schema=False)
async def index():
    """返回 HTML 单页前端"""
    index_path = os.path.join(STATIC_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(
        "<html><body><h2>未找到前端页面</h2>"
        "<p>请确认 app/static/index.html 存在，或改用 /docs 查看 API 文档。</p></body></html>",
        status_code=404,
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health", summary="健康检查")
def health_check():
    """检查服务是否正常运行，并返回当前使用的计算设备（CPU/GPU）"""
    return {"status": "ok", "device": str(device)}


@app.post("/parse_wfdb", summary="解析 WFDB 心电图文件")
async def parse_wfdb_file(dat: UploadFile = File(...), hea: UploadFile = File(...)):
    """解析上传的 WFDB 文件（.dat + .hea），返回信号、导联名与元信息"""
    try:
        dat_bytes = await dat.read()
        hea_bytes = await hea.read()
        result = parse_wfdb(dat_bytes, hea_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"WFDB 解析失败：{e}")


@app.post("/parse_dicom", summary="解析 DICOM 心电图文件")
async def parse_dicom_file(dcm: UploadFile = File(...)):
    """解析上传的 DICOM 波形文件（.dcm），返回信号、导联名与元信息"""
    try:
        dcm_bytes = await dcm.read()
        result = parse_dicom(dcm_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DICOM 解析失败：{e}")


@app.post("/predict", response_model=PredictionResult, summary="单条心电图诊断")
def predict_single(ecg: ECGInput, explain: bool = False):
    """
    对一条心电图（多导联原始信号）进行多标签自动诊断。

    **输入说明：**
    - `ecg`：包含 `signals` 字段，它是一个二维列表，形状为 `[导联数, 采样点数]`，数值为原始电压（mV）。
      - 示例：12导联、5000个采样点的数据应为 `[ [0.1, 0.2, ...], [0.0, 0.1, ...], ... ]`，共12个子列表，每个子列表长度5000。
    - `explain`：是否返回模型判断依据的可视化热力图（Grad‑CAM），默认关闭以加快速度。

    **输出说明：**
    - `probabilities`：每一类疾病的预测概率（0~1之间的小数）
    - `pred_labels`：根据最优阈值得到的二值预测结果（0=无病，1=有病）
    - `class_names`：所有可能疾病的中文名称列表
    - `threshold_used`：每种疾病对应的判断阈值
    - `gradcam_heatmaps`：（仅在 `explain=true` 时出现）阳性类别的热力图，展示模型关注的心电信号片段
    """
    try:
        signals = preprocess_signal(ecg.signals, config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"信号预处理失败：{e}")

    input_tensor = torch.from_numpy(signals).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().squeeze(0).numpy()
    preds = (probs >= thresholds).astype(int)

    result = {
        "probabilities": probs.tolist(),
        "pred_labels": preds.tolist(),
        "class_names": class_names,
        "threshold_used": thresholds.tolist(),
    }

    if explain:
        heatmaps = {}
        for idx, val in enumerate(preds):
            if val == 1:
                hm = gradcam.generate(input_tensor, class_idx=idx)
                heatmaps[class_names[idx]] = hm.tolist()
        if not heatmaps:
            top_class = np.argmax(probs)
            hm = gradcam.generate(input_tensor, class_idx=top_class)
            heatmaps[class_names[top_class]] = hm.tolist()
        result["gradcam_heatmaps"] = heatmaps

    return PredictionResult(**result)


@app.post("/predict_batch", response_model=BatchPredictionResponse, summary="批量心电图诊断")
def predict_batch(ecgs: List[ECGInput], explain: bool = False):
    """
    一次传入多条心电图数据进行批量诊断，适合大批量自动筛查。

    **输入说明：**
    - `ecgs`：是一个列表，列表中每个元素是一条心电图的输入（格式同单条诊断的 `ECGInput`）。
    - `explain`：是否生成热力图，开启后每条记录都会包含解释，处理时间会相应增加。

    **输出说明：**
    - `results`：每条心电图的详细诊断结果（同单条诊断的输出）。
    - `summary`：汇总统计，包括总样本数和每种疾病的阳性比例。
    """
    results = []
    for ecg in ecgs:
        try:
            signals = preprocess_signal(ecg.signals, config)
        except Exception as e:
            results.append(PredictionResult(
                probabilities=[], pred_labels=[], class_names=class_names,
                threshold_used=thresholds.tolist(), message=f"预处理错误: {e}"
            ))
            continue

        input_tensor = torch.from_numpy(signals).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().squeeze(0).numpy()
        preds = (probs >= thresholds).astype(int)

        item = {
            "probabilities": probs.tolist(),
            "pred_labels": preds.tolist(),
            "class_names": class_names,
            "threshold_used": thresholds.tolist(),
        }
        if explain:
            heatmaps = {}
            for idx, val in enumerate(preds):
                if val == 1:
                    hm = gradcam.generate(input_tensor, class_idx=idx)
                    heatmaps[class_names[idx]] = hm.tolist()
            if not heatmaps:
                top_class = np.argmax(probs)
                hm = gradcam.generate(input_tensor, class_idx=top_class)
                heatmaps[class_names[top_class]] = hm.tolist()
            item["gradcam_heatmaps"] = heatmaps

        results.append(PredictionResult(**item))

    # 汇总统计
    total = len(results)
    positive_rate = {name: 0.0 for name in class_names}
    for res in results:
        for i, label in enumerate(res.pred_labels):
            if label == 1:
                positive_rate[class_names[i]] += 1
    for k in positive_rate:
        positive_rate[k] = round(positive_rate[k] / total, 4) if total > 0 else 0.0

    summary = {
        "total_samples": total,
        "positive_rate_per_class": positive_rate
    }
    return BatchPredictionResponse(results=results, summary=summary)


if __name__ == "__main__":
    # 默认仅本机访问（127.0.0.1）。如需局域网共享，改为 host="0.0.0.0"
    uvicorn.run("app.backend.main:app", host="127.0.0.1", port=8000, reload=True)