# 心电图 AI 辅助诊断平台 (ECG_Model)

基于 **PTB-XL** 公开数据集的 **12 导联心电图多标签智能诊断系统**，采用一维深度神经网络（ResNet-1D）同时识别 10 种心血管疾病，并提供 Web 网页端交互式诊断与 **Grad-CAM 可解释性热力图**。

- **Web 前端**：纯 HTML + Plotly.js + FastAPI（单服务部署，浏览器即用）
- **模型**：ResNet-18 一维变体（含 Squeeze-and-Excitation 注意力）
- **诊断**：10 类多标签分类 + 优化阈值判定 + Grad-CAM 波形热力图

> ⚠️ **医学免责声明**：本系统输出的诊断结果仅供专业医师参考，不作为自动诊断或临床决策的唯一依据。

---

## 演示视频

<video src="demo_video.mp4" controls width="100%"></video>

> 🎬 点击上方 ▶ 即可在网页端直接播放操作演示（视频已随仓库分发，也可下载 `demo_video.mp4` 本地查看）。

---

## 功能特性

| 功能 | 说明 |
|---|---|
| 📥 **双格式上传** | 支持 WFDB（`.dat` + `.hea`）与 DICOM（`.dcm`）心电图文件 |
| 📈 **交互式波形** | Plotly.js 多导联堆叠图，支持缩放、悬停读取、导联多选 |
| 🧠 **多标签诊断** | 同时输出 10 类心血管疾病的概率与阳性/阴性判定 |
| 🔥 **可解释性热力图** | Grad-CAM 热力图叠加到波形，展示模型关注的心电片段 |
| 🩺 **结果可视化** | 概率进度条、阳性异常高亮、异常汇总警示框 |
| 🚀 **单服务部署** | FastAPI 同时托管后端 API 与 HTML 前端，一条命令启动 |

---

## 支持的诊断类别（10 类多标签）

| SCP 编码 | 疾病名称 |
|----------|----------|
| NORM | 正常心电图 |
| IMI | 下壁心肌梗死 |
| ASMI | 前间壁心肌梗死 |
| LVH | 左心室肥厚 |
| NDT | 非特异性 ST-T 改变 |
| LAFB | 左前分支传导阻滞 |
| ISC_ | 心肌缺血 |
| IRBBB | 不完全性右束支传导阻滞 |
| PVC | 室性早搏 |
| IVCD | 室内传导异常 |

---

## 安装

### 1. 环境要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | **3.9+**（已在 3.14 上验证） | 建议 3.10 ~ 3.14 |
| PyTorch | 2.2+（已在 2.10 上验证） | 无 GPU 时安装 CPU 版即可 |
| numpy / pandas / scipy | 2.x / 2.x+ / 1.13+ | 兼容最新版本 |
| 硬件 | 无硬性要求 | 推理 CPU 即可；训练推荐 GPU 加速 |

> 本项目**无需 GPU 也能运行**：模型推理在 CPU 上约 0.5 秒/条；训练在 CPU 上较慢（约数小时），建议有 GPU 时训练。

### 2. 获取代码

```bash
git clone https://github.com/RumRune/ECG_Model.git
cd ECG_Model
```

### 3. 创建虚拟环境（推荐）

**方式 A：venv（官方原生）**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

**方式 B：conda**

```bash
conda create -n ecg_model python=3.11
conda activate ecg_model
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

安装后验证：

```bash
python -c "import torch, numpy, pandas, fastapi, uvicorn; print('✅ 依赖安装成功')"
```

**核心依赖清单**：`torch`、`numpy`、`scipy`、`pandas`、`scikit-learn`、`matplotlib`、`seaborn`、`plotly`、`fastapi`、`uvicorn[standard]`、`pydantic`、`pyyaml`、`tqdm`、`wfdb`、`pydicom`、`joblib`。

> 可选：`pip install captum` 可启用 Integrated Gradients 可解释性分析（`interpret.py --integrated`）；`pip install tensorboard` 可记录训练曲线（未安装时 train.py 自动跳过日志）。

---

## 运行（使用已训练模型）

> ✅ 训练好的 10 类模型权重 `models/checkpoints/best_model.pth`（约 16MB）。
> ⚠️ 权重**不随仓库分发**（`.gitignore` 默认排除，避免仓库过大）；发布时已作为 **GitHub Release 附件**提供，请从 Release 页下载后放入 `models/checkpoints/`，或按下方「复现流程」自行训练。

### 方式 1：启动 Web 页面（推荐）

在项目根目录执行：

```bash
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

浏览器访问： **http://localhost:8000**

> 等效的另一种启动方式：`python app/backend/main.py`（main.py 内置 uvicorn 启动）。

**局域网共享**：把 `--host` 改为 `0.0.0.0`，其他设备访问 `http://<本机局域网IP>:8000`（如 `http://192.168.1.100:8000`）。
> ⚠️ `0.0.0.0` 是服务器监听地址，**不能**作为浏览器访问目标。

**页面使用步骤**：
1. 在左侧选择 **WFDB**（`.dat` + `.hea`）或 **DICOM**（`.dcm`）格式
2. 点击上传框或**直接拖拽**文件进入（WFDB 需两个文件都选齐）
3. 自动解析并显示 12 导联波形（默认显示第一个导联，可在下拉框**多选**）
4. 点击「🔍 开始 AI 分析」→ 显示诊断结果表格 + 异常汇总
5. 波形上会叠加 **Grad-CAM 热力图**（红色半透明色带 = 模型关注的片段）

**试运行示例文件**（仓库自带 PTB-XL 数据时可直接用）：

```
data/raw/ptb-xl_1.0.2/records100/00000/00001_lr.dat
data/raw/ptb-xl_1.0.2/records100/00000/00001_lr.hea
```

### 方式 2：调用 API（命令行 / 程序）

启动服务后，可用 curl 或任意 HTTP 客户端调用：

```bash
# 健康检查
curl http://localhost:8000/health

# 单条诊断（signals 为 12 导联信号，形状 (12, N)）
curl -X POST "http://localhost:8000/predict?explain=true" \
  -H "Content-Type: application/json" \
  -d '{"signals": [[0.1,0.2,0.15,...], [0.0,0.1,0.2,...], ...]}'

# 批量诊断
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{"ecgs": [{"signals": [...]}, {"signals": [...]}]}'
```

Python 调用示例：

```python
import requests

resp = requests.post(
    "http://localhost:8000/predict?explain=true",
    json={"signals": signals_2d_list},   # 12 导联 × N 采样点
    timeout=60,
)
result = resp.json()
print(result["class_names"])        # 10 类名称
print(result["probabilities"])      # 各类概率
print(result["pred_labels"])        # 0/1 预测
print(result["gradcam_heatmaps"])   # 热力图（explain=true 时）
```

**API 文档**：启动服务后访问 **http://localhost:8000/docs** 查看中文 Swagger 交互式文档，可直接在页面调试所有接口。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | HTML 前端页面 |
| `/health` | GET | 健康检查，返回当前设备（CPU/GPU） |
| `/parse_wfdb` | POST | 上传 `.dat` + `.hea`，解析返回信号与导联 |
| `/parse_dicom` | POST | 上传 `.dcm`，解析返回信号与导联 |
| `/predict` | POST | 单条心电信号诊断（`?explain=true` 返回热力图） |
| `/predict_batch` | POST | 批量诊断 + 汇总统计 |

---

## 项目结构

```
ECG_Model/
├── app/                    # Web 应用
│   ├── backend/            # FastAPI 后端
│   │   ├── main.py         #   后端入口 + API 服务 + HTML 托管
│   │   ├── utils.py        #   配置加载 / 模型构建 / 阈值 / 预处理
│   │   ├── file_parser.py  #   WFDB / DICOM 纯内存解析
│   │   └── schemas.py      #   Pydantic 数据模型
│   └── static/             # HTML 前端
│       ├── index.html      #   单页前端（上传/波形/诊断/热力图）
│       └── vendor/plotly.min.js  # 本地 Plotly（无外网依赖）
├── src/                    # 数据处理与模型
│   ├── train.py            # 训练脚本（命令行）
│   ├── train.ipynb         # 训练 notebook（含曲线可视化）
│   ├── inference.ipynb     # 推理 notebook
│   ├── threshold_optimization.ipynb # 阈值优化 notebook
│   ├── build_labels.py     # 多标签矩阵构建（修复版）
│   ├── evaluate.py         # 评估与阈值优化 CLI
│   ├── interpret.py        # Grad-CAM 可解释性分析
│   ├── baseline.py         # 随机森林基线
│   ├── data_loader.py      # 数据加载器
│   └── signal_processing.py# 信号处理工具
├── models/                 # 模型
│   ├── model_zoo/resnet1d.py  # ResNet-1D 网络定义
│   └── checkpoints/            # 训练好的权重（.gitignore 默认排除）
│       ├── best_model.pth      #    深度学习模型（当前）
│       └── baseline_rf.pkl     #    随机森林基线
├── data/                   # 数据（.gitignore 默认排除）
│   ├── raw/                #   原始 PTB-XL（physionet）
│   └── processed/          #   预处理后的信号 + 多标签
├── configs/                # 配置文件
│   ├── config.yaml         #   主配置（类别/超参/阈值）
│   └── label_mapping.json  #   类别映射
├── reports/                # 评估结果、混淆矩阵、曲线
├── notebooks/              # 数据探索与预处理 notebook
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_label_parsing.ipynb
│   ├── data_checking.ipynb
│   └── preprocess.py       #   预处理命令行脚本
├── logs/                   # TensorBoard 训练日志（.gitignore 排除）
├── tests/                  # 冒烟测试
├── requirements.txt        # 依赖清单
└── README.md
```

---

## 数据说明与准备

### 数据集

使用 **PTB-XL**（PhysioNet 公开心电图数据集，[官方页面](https://physionet.org/content/ptb-xl/1.0.2/)），包含约 2.2 万条 12 导联心电记录。

> ⚠️ 原始数据较大（约 5GB），`data/raw/` 与 `data/processed/` 已在 `.gitignore` 中排除，**不随仓库分发**。需复现训练请先获取数据。

### 下载原始数据

```bash
# 从 PhysioNet 下载 PTB-XL 1.0.2（需按页面说明获取）
# 解压后目录结构应为：
# data/raw/ptb-xl_1.0.2/
# ├── ptbxl_database.csv
# ├── scp_statements.csv
# ├── records100/     (100Hz 信号)
# └── records500/     (500Hz 信号)
```

### 预处理（信号 → npy）

**方式 A：运行 notebook**（推荐，可视化逐步执行）

用 Jupyter 打开 `notebooks/02_data_preprocessing.ipynb`，按顺序执行：
1. 加载 `ptbxl_database.csv` 并清洗（保留年龄 0–120）
2. 按官方 `strat_fold` 划分 train(1–8) / val(9) / test(10)
3. 每条记录：带通滤波(0.5–45Hz) → 陷波(50Hz) → 重采样至 100Hz → 截取 10 秒(1000点) → Z-score
4. 输出到 `data/processed/{train,val,test}/signals.npy` + `labels.csv`

**方式 B：命令行脚本**

```bash
python notebooks/preprocess.py \
  --csv data/raw/ptb-xl_1.0.2/ptbxl_database.csv \
  --data_dir data/raw/ptb-xl_1.0.2 \
  --output_dir data/processed \
  --target_fs 100 --duration 10
```

### 构建多标签矩阵

```bash
python src/build_labels.py
```

该脚本按 SCP 编码 `value>0` 选取 10 个诊断类并生成多标签矩阵（修复了旧版把 0 概率码误标为正例的 bug）：

```
data/processed/labels/{train,val,test}_multilabel.npy
data/processed/labels/{train,val,test}_labels_parsed.csv
```

---

## 复现流程（训练 → 评估 → 解释）

> 训练好的 `best_model.pth` 通过 **GitHub Release 附件**提供（不在 git 仓库内），若仅想使用可跳过训练，从 Release 下载放入 `models/checkpoints/` 即可。以下为完整复现。

### 步骤 1：构建数据与标签（见上文）

### 步骤 2：训练模型

**方式 A：命令行**

```bash
python src/train.py --config configs/config.yaml
```

**方式 B：Jupyter notebook**

```bash
jupyter notebook src/train.ipynb
```
（含训练曲线可视化，分步执行）

**训练说明**：
- 默认配置 50 epochs，Focal Loss（gamma=3.0）+ 加权采样，早停 patience=10
- 输出：`models/checkpoints/best_model.pth`（验证损失最优）+ `logs/resnet18_<时间戳>/` 日志
- 无 GPU 时自动使用 CPU；有 GPU 自动使用 CUDA
- 训练完成提示：`训练完成，最佳验证损失: x.xxxx`

**配置调整**（`configs/config.yaml`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training.epochs` | 50 | 最大训练轮数 |
| `training.batch_size` | 64 | 批大小 |
| `training.learning_rate` | 0.0005 | 初始学习率 |
| `training.focal_gamma` | 3.0 | Focal Loss 聚焦参数 |
| `training.early_stopping_patience` | 10 | 早停耐心值 |
| `training.use_weighted_sampler` | true | 类别不平衡加权采样 |
| `model.name` | resnet18 | 可选 resnet18 / resnet34 |
| `data.class_names` | 10 类 | 诊断类别列表 |

### 步骤 3：阈值优化 + 评估

```bash
# 在验证集上为每类搜索最优 F1 阈值，并输出测试集评估
python src/evaluate.py \
  --config configs/config.yaml \
  --checkpoint models/checkpoints/best_model.pth \
  --optimize_threshold
```

**evaluate.py 参数说明**：

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径 |
| `--checkpoint` | 模型权重路径 |
| `--threshold` | 固定全局阈值（默认 0.5，不使用优化阈值时） |
| `--optimize_threshold` | 在验证集搜索每类最优阈值并评估测试集 |
| `--use_optimal` | 加载已有的优化阈值（`configs/optimized_thresholds.yaml`） |
| `--baseline` | 基线模型结果 JSON（用于对比打印） |

**输出**：
- 控制台：Macro/Micro F1、AUC、各类别 F1、临床指标（敏感度/特异度/PPV/NPV）
- `reports/eval_results.json`：评估结果
- `reports/optimal_thresholds.json` + `configs/optimized_thresholds.yaml`：优化阈值
- `reports/overall_confusion.png`、`reports/per_class_confusion.png`：混淆矩阵图

```bash
# 仅评估（使用已有阈值）
python src/evaluate.py --config configs/config.yaml \
  --checkpoint models/checkpoints/best_model.pth --use_optimal
```

### 步骤 4：可解释性分析（Grad-CAM）

```bash
# 分析测试集第 0 个样本
python src/interpret.py --config configs/config.yaml \
  --checkpoint models/checkpoints/best_model.pth --sample 0

# 同时计算 Integrated Gradients（需 pip install captum）
python src/interpret.py --config configs/config.yaml \
  --checkpoint models/checkpoints/best_model.pth --sample 0 --integrated
```

**interpret.py 参数说明**：

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径 |
| `--checkpoint` | 模型权重路径 |
| `--sample` | 测试集样本索引（默认 0） |
| `--output_dir` | 输出目录（默认 `reports/interpretations/`） |
| `--integrated` | 同时计算 Integrated Gradients（需 captum） |

**输出**：Grad-CAM 热力图叠加波形图 + 文本报告（各类别重要时间区域）。

### 步骤 5（可选）：随机森林基线

```bash
python src/baseline.py
```
训练传统机器学习基线（手工特征 + 随机森林），输出 `models/checkpoints/baseline_rf.pkl`，用于与深度学习模型对比。

---

## 测试

```bash
# 冒烟测试（无需数据/模型）
python tests/test_smoke.py

# 或使用 pytest
python -m pytest tests/ -v
```

覆盖：配置加载、模型前向、标签加载、阈值加载、信号预处理。

---

## 模型与评估

### 模型结构
- **ResNet-18 一维变体**：`Conv1d` 替换 `Conv2d`，残差块内嵌入 **SE（Squeeze-and-Excitation）注意力**
- 输入 `(batch, 12, 1000)`，输出 `(batch, 10)` logits（无 sigmoid，训练用 Focal Loss）
- 支持 `resnet18_1d` 与 `resnet34_1d` 两种配置

### 评估指标（测试集 2161 例）

| 指标 | 数值 |
|------|------|
| Macro F1 | **0.586** |
| Micro F1 | 0.674 |
| Macro AUC | **0.918** |
| Hamming Loss | 0.079 |

各类别表现较好：NORM (F1 0.84)、ASMI (0.74)、LAFB (0.72)。受类别不平衡影响，稀有类（IVCD F1 0.20）偏弱。

---

## 已知限制与后续改进

- 稀有类别（如 IVCD）样本少，识别偏弱，可尝试数据增强 / 更多数据 / 类别重加权
- 模型当前为 CPU 训练，训练时间较长，训练脚本已支持 GPU 加速
- WFDB 当前支持单 `.dat` 文件记录（对应 PTB-XL 等常规格式）

---

## 许可证与致谢

- 数据集：PTB-XL（© 2020 Physionet，遵循其数据集许可协议）
- 本项目暂时仅供练习、测试用途

**版本**：v0.2.0 | **日期**：2026-08
