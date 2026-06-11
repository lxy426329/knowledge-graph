# 外眼多模态识别系统（EfficientNet-B3 + BERT）

> 针对外眼病（白内障 / 结膜炎 / 眼睑疾病 / 葡萄膜炎 / 正常）的细分类识别，同时输出图像特征、文本特征、融合特征与风险等级，方便对接知识图谱 / RAG 咨询模块。

## 一、项目结构

```
.
├── config.py                         # 全局配置：路径 / 训练参数 / 类别体系 / 高风险名单
├── main.py                           # 命令行入口（训练 / 推理 / 评估 / API）
├── train.py                          # 训练脚本：Focal Loss + 余弦退火 + 早停
├── predict_module.py                 # 推理接口 + 手机实拍图预处理 + DEMO 模式
│
├── models/
│   ├── dataset.py                    # 外眼病多模态数据集 + train/val/test 划分 + 增广
│   ├── efficientnet.py               # EfficientNet-B3 图像编码器
│   ├── bert_encoder.py               # BERT（bert-base-chinese）文本编码器
│   └── multimodal.py                 # 门控融合 + 5 类细分类头
│
├── api/
│   └── app.py                        # FastAPI 服务（单图 / 批量 / 类别 / 健康检查）
│
├── evaluation/
│   └── multimodal_eval.py            # 评估脚本 + 消融实验（仅图像 / 随机基线）
│
├── data/
│   └── archive_external/             # 训练图像：按类别分子目录放图
│       ├── Cataract/                 #   白内障
│       ├── Conjunctivitis/           #   结膜炎
│       ├── Eyelid/                   #   眼睑疾病
│       ├── Uveitis/                  #   葡萄膜炎
│       └── Normal/                   #   正常
│
├── _self_check.py                    # 一键自检（导入 / 数据 / 模型 / 推理 / API）
├── requirements.txt                  # 依赖清单
└── README.md
```

## 二、类别体系

| 索引 | 英文 key        | 中文名称 | 高风险 | 典型数据量 |
|-----|----------------|----------|-------|-----------|
| 0   | Cataract       | 白内障   | 是    | ~544 张   |
| 1   | Conjunctivitis | 结膜炎   | 是    | ~357 张   |
| 2   | Eyelid         | 眼睑疾病 | 是    | ~525 张   |
| 3   | Uveitis        | 葡萄膜炎 | 是    | ~223 张   |
| 4   | Normal         | 正常     | 否    | ~649 张   |

- `EXTERNAL_EYE_CLASSES` 与 `CLASS_MAP` 定义于 `config.py`，训练 / 推理 / 评估全局一致。
- 高风险类别保存在 `HIGH_RISK_EXTERNAL`，训练与评估都会单独统计高风险召回率。

## 三、运行环境

推荐 Python 3.12（已在 3.12 验证通过），CPU 与 CUDA 均可：

```powershell
# 创建并激活 conda 环境（推荐）
conda create -n mingshijie python=3.12 -y
conda activate mingshijie

# 安装项目依赖
pip install -r requirements.txt
```

> 首次运行会联网下载 `bert-base-chinese`（约 400 MB，HuggingFace 已镜像加速）与 EfficientNet-B3 预训练权重，之后即可离线使用。

## 四、数据准备

1. 按上面的目录结构在 `data/archive_external/` 下放置训练图像；子目录名必须是 `Cataract / Conjunctivitis / Eyelid / Uveitis / Normal` 之一。
2. （可选）在 `data/annotations/symptoms.json` 中提供症状文本：

   ```json
   { "img_001.jpg": "左眼红肿3天，畏光流泪", "img_002.jpg": "" }
   ```

3. 查看数据分布：
   ```powershell
   python main.py     # 选择 4
   ```

## 五、训练

```powershell
python main.py     # 选择 2，或直接：
python train.py
```

- 主干：EfficientNet-B3（ImageNet 预训练）+ bert-base-chinese（HuggingFace 预训练）
- 融合：门控机制（sigmoid gate + 线性投影 + LayerNorm）
- 损失：Focal Loss（γ=2.0）+ label smoothing=0.05
- 优化：AdamW（lr=5e-4, weight_decay=1e-4）+ 余弦退火 + 梯度裁剪 norm=5.0
- 早停：在验证集 loss 上监测，连续 5 轮未下降则停止

产物：
- `checkpoints/best.pt` — 最佳权重 + 类别映射
- `checkpoints/class_map.json` — 类别元信息
- `runs/train_*.log` — 训练日志（每轮 acc / macro_f1 / 高风险召回率）

## 六、推理

### 6.1 单张图像推理（本地）

```python
from predict_module import ExternalEyePredictor

p = ExternalEyePredictor("checkpoints/best.pt")
r = p.predict("data/archive_external/Conjunctivitis/xx.jpg",
              symptom_text="左眼红肿3天")

print(r.disease_name)         # 结膜炎
print(r.confidence)           # 0.82
print(r.risk_level)           # high / medium / low
print(r.risk_reason)          # 风险说明
print(r.fused_feature)        # 融合特征（512 维，L2 归一化，可直接向量检索）
```

若 `checkpoints/best.pt` 不存在，`ExternalEyePredictor()` 会自动进入 **DEMO 模式**（随机权重，便于联调）。

### 6.2 FastAPI 服务

```powershell
uvicorn api.app:app --host 0.0.0.0 --port 8000
# 或：python main.py  # 选择 1
```

| 方法 | 路径                          | 说明                              |
|------|-------------------------------|-----------------------------------|
| POST | `/external_eye/predict`       | 单图 + 可选症状文本 → 细分类+风险  |
| POST | `/external_eye/predict/batch` | 批量推理                          |
| GET  | `/classes`                    | 类别体系 + 高风险索引             |
| GET  | `/health`                     | 健康检查                          |
| GET  | `/`                           | API 信息                          |

`/external_eye/predict` 响应示例：

```json
{
  "disease_key": "Conjunctivitis",
  "disease_name": "结膜炎",
  "confidence": 0.82,
  "all_probabilities": [0.03, 0.82, 0.05, 0.07, 0.03],
  "all_class_keys": ["Cataract","Conjunctivitis","Eyelid","Uveitis","Normal"],
  "all_class_names": ["白内障","结膜炎","眼睑疾病","葡萄膜炎","正常"],
  "risk_level": "high",
  "risk_reason": "高风险外眼病（结膜炎），置信度较高，建议尽快就诊",
  "image_feature": [...],
  "text_feature": [...],
  "fused_feature": [...]
}
```

## 七、评估与消融实验

```powershell
python evaluation/multimodal_eval.py
# 或：python main.py  # 选择 5
```

会输出：
- `train / val / test` 的 accuracy / macro_f1 / high_risk_recall
- 混淆矩阵 + 每类 P / R / F1
- `ablation_image_only`（症状置空，仅靠图像）
- `ablation_random_baseline`（随机猜测基线）

## 八、一键自检

```powershell
python _self_check.py
```

覆盖 5 项检查：模块导入、配置、数据集加载、模型 forward、推理接口。若全部通过则打印 `整体结果: OK 全部通过`。

## 九、对接下游模块

- **知识图谱**：读 `fused_feature` 做向量检索，读 `disease_key / confidence / risk_level` 触发图谱路径。
- **RAG 咨询**：把 `disease_name / risk_reason / confidence` 拼接成提示词，交给大模型生成就诊建议。

## 十、手机实拍图适配

`predict_module.mobile_preprocess` 在推理前自动执行：
1. 锐化（UnsharpMask radius=1.5），抑制手机模糊
2. 对比度 / 亮度 / 色彩轻微增强（1.15 / 1.05 / 1.05）
3. 中心裁剪 + resize 到 300×300，减弱角度偏差
4. 再走 ImageNet 归一化 → 模型

若后续需要更激进的增强（直方图均衡 / 去噪），修改 `mobile_preprocess` 即可，不影响模型推理主流程。

## 十一、已知限制与说明

- 类别 0–3 均被视作高风险；类别 4（Normal）为正常。
- 图像输入尺寸默认 300×300；若需其他尺寸，修改 `TRAIN_CONFIG["image_size"]` 即可，训练推理需保持一致。
- 症状文本在缺失时会被替换为固定字符串 "无明显症状"，避免 BERT 空序列导致的不稳定。
- 默认情况下 PyCharm Terminal 的 conda 环境名 `mingshijie` 不会出现在 `conda env list` 中，只需在 PyCharm 的 Python 解释器设置里手动选 `C:/Users/<你>/.conda/envs/mingshijie/python.exe` 或 `D:/anaconda/envs/mingshijie/python.exe` 即可。
