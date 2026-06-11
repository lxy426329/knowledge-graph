"""
外眼识别 REST API — 基于 EfficientNet-B3 + BERT 多模态模型。

Endpoints:
    POST   /external_eye/predict         单张图像 + 可选症状 → 细分类 + 风险
    POST   /external_eye/predict/batch   批量推理
    GET    /classes                      外眼病类别列表
    GET    /health                       健康检查
    GET    /                             API 信息
"""
from __future__ import annotations

# 标准库
import io
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

# FastAPI 依赖
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image  # 用于解析上传的图像字节

# 项目内部：类别体系 / 高风险名单 / 标准化推理器
from config import (
    API_CONFIG,
    EXTERNAL_EYE_SUBCLASS,
    HIGH_RISK_EXTERNAL,
)
from predict_module import ExternalEyePredictor


# ── 全局状态：外眼多模态推理器 ──────────────────────────
# 在 lifespan 中创建一次，之后所有接口共享。避免每次请求都加载权重。
state: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：启动时加载模型，停止时释放资源。"""
    print("[API] 正在加载外眼多模态推理模块 ...")
    state["eye_predictor"] = ExternalEyePredictor()
    print("[API] 模型加载完成，可以接受请求。")
    yield  # 服务运行期
    state.clear()
    print("[API] 服务已停止。")


app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    lifespan=lifespan,
)


async def _read_image(file: UploadFile) -> Image.Image:
    """
    从上传文件读取 bytes 并解析为 PIL RGB 图像。
    解析失败直接抛出 HTTP 400，避免上层 try/except 散落到各处。
    """
    try:
        contents = await file.read()
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法解析图像文件: {exc}")


@app.get("/")
async def root():
    """API 基本信息（便于监控 / 浏览器查看）。"""
    return {
        "title": API_CONFIG["title"],
        "description": API_CONFIG["description"],
        "version": API_CONFIG["version"],
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """健康检查：返回模型是否已成功加载。"""
    return {
        "status": "healthy",
        "model_loaded": "eye_predictor" in state and state["eye_predictor"] is not None,
    }


@app.get("/classes")
async def get_classes():
    """返回类别体系 & 高风险索引，供下游模块（知识图谱 / RAG）使用。"""
    predictor: Optional[ExternalEyePredictor] = state.get("eye_predictor")
    if predictor is None:
        # 尚未加载时退回到 config 里的默认列表，避免空响应
        return {
            "class_keys": list(EXTERNAL_EYE_SUBCLASS.keys()) + ["Normal"],
            "class_names": list(EXTERNAL_EYE_SUBCLASS.values()) + ["正常"],
            "high_risk_keys": HIGH_RISK_EXTERNAL,
        }
    return {
        "class_keys": predictor.class_keys,
        "class_names": predictor.class_names,
        "high_risk_indices": predictor.high_risk_indices,
        "high_risk_keys": [
            k for i, k in enumerate(predictor.class_keys) if i in predictor.high_risk_indices
        ],
    }


@app.post("/external_eye/predict")
async def external_eye_predict(
    file: UploadFile = File(...),
    symptom_text: Optional[str] = Form(default=""),
):
    """
    单张图像推理：
        - file:          上传的图像文件（jpg/png/bmp 等）
        - symptom_text:  可选症状文本（form-data），为空时模型内部退化为 "无明显症状"
    返回：与 PredictResult.to_dict() 一致
    """
    image = await _read_image(file)
    predictor: ExternalEyePredictor = state["eye_predictor"]
    result = predictor.predict(image, symptom_text=symptom_text or "")
    return result.to_dict()


@app.post("/external_eye/predict/batch")
async def external_eye_predict_batch(
    files: List[UploadFile] = File(...),
    symptom_texts: Optional[str] = Form(default=""),
):
    """
    批量推理：
        - files:         多文件上传
        - symptom_texts: 可选症状文本，用英文逗号分隔与 files 对齐；不足自动补空
    返回：
        - bad_files:     无法解析的文件文件名列表
        - results:       每条图像一个 PredictResult dict
    """
    images: List[Image.Image] = []
    bad_files: List[str] = []
    for f in files:
        try:
            images.append(await _read_image(f))
        except HTTPException:
            bad_files.append(f.filename or "unknown")

    # 症状文本按 "," 拆分，并与 images 长度对齐
    texts: List[str] = []
    if symptom_texts:
        texts = [t.strip() for t in symptom_texts.split(",")]
    while len(texts) < len(images):
        texts.append("")
    texts = texts[: len(images)]

    predictor: ExternalEyePredictor = state["eye_predictor"]
    results = predictor.predict_batch(images, texts)
    return {
        "bad_files": bad_files,
        "results": [r.to_dict() for r in results],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host=API_CONFIG["host"], port=API_CONFIG["port"], reload=False)
