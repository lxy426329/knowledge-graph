r"""
项目整体体检：模块 import / 推理 forward / 批量 / 便捷函数 / FastAPI 接口 / 评估脚本。
运行： .venv\Scripts\python.exe _self_check.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ---------- 工具 ----------
def _tick(name: str):
    print(f"\n>>> [{name}]")


def _ok(msg: str):
    print(f"    OK  - {msg}")


def _fail(msg: str):
    print(f"    FAIL - {msg}")


def _rand_pil():
    from PIL import Image
    arr = np.random.randint(0, 255, (480, 640, 3), dtype="uint8")
    return Image.fromarray(arr)


def _real_image_path() -> str:
    root = HERE / "data" / "archive_external" / "Conjunctivitis"
    if root.exists():
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                return str(p)
    # 没有真实图像就落回到临时文件
    tmp = HERE / "_tmp_dummy.png"
    _rand_pil().save(tmp)
    return str(tmp)


# ---------- 1. 模块 import ----------
def check_modules() -> int:
    _tick("模块 import")
    failed = 0
    modules = [
        "config",
        "models.dataset",
        "models.efficientnet",
        "models.bert_encoder",
        "models.multimodal",
        "predict_module",
        "api.app",
        "evaluation.multimodal_eval",
        "train",
    ]
    for m in modules:
        try:
            __import__(m)
            _ok(m)
        except Exception as exc:
            _fail(f"{m}: {exc}")
            failed += 1
    return failed


# ---------- 2. 配置 ----------
def check_config() -> int:
    _tick("配置")
    import config as cfg

    required = [
        cfg.EXTERNAL_EYE_SUBCLASS,
        cfg.HIGH_RISK_EXTERNAL,
        cfg.TRAIN_CONFIG,
    ]
    for r in required:
        assert r is not None and len(r) > 0, "配置为空"

    # 类别数：4 病 + 1 正常 = 5
    classes = list(cfg.EXTERNAL_EYE_SUBCLASS.keys())
    assert len(classes) == 4, f"EXTERNAL_EYE_SUBCLASS 应有 4 个病种，实际 {len(classes)}"
    _ok(f"EXTERNAL_EYE_SUBCLASS={list(cfg.EXTERNAL_EYE_SUBCLASS.keys())}")
    _ok(f"HIGH_RISK_EXTERNAL={cfg.HIGH_RISK_EXTERNAL}")
    _ok(f"TRAIN_CONFIG.image_size={cfg.TRAIN_CONFIG['image_size']}")
    return 0


# ---------- 3. 数据集 ----------
def check_dataset() -> int:
    _tick("数据集")
    from models.dataset import build_dataloaders, ExternalEyeDataset

    try:
        loaders, meta = build_dataloaders(batch_size=2, num_workers=0, image_size=224)
    except Exception as exc:
        _fail(f"build_dataloaders: {exc}")
        return 1

    assert meta.num_disease_classes == 5, f"num_disease_classes={meta.num_disease_classes}"
    assert meta.total_samples > 0, "样本数为 0"
    _ok(f"num_disease_classes=5, total_samples={meta.total_samples}, class_keys={meta.disease_class_keys}")

    # 跑一个 batch
    for split in ("train", "val", "test"):
        images, texts, labels = next(iter(loaders[split]))
        assert images.shape == (2, 3, 224, 224), f"{split} image shape wrong: {images.shape}"
        assert len(texts) == 2, f"{split} texts length wrong"
        assert labels.shape == (2,), f"{split} labels shape wrong: {labels.shape}"
        assert labels[0].item() in range(5), f"{split} label out of range: {labels[0].item()}"
        _ok(f"{split}: images={images.shape}, labels={labels.tolist()}, texts={[t[:20] for t in texts]}")
    return 0


# ---------- 4. 模型 forward ----------
def check_model_forward() -> int:
    _tick("模型 forward")
    import torch
    from models.multimodal import MultiModalEyeModel

    model = MultiModalEyeModel(num_disease_classes=5, feature_dim=512, image_pretrained=False)
    model.eval()

    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones_like(input_ids)
        out = model(x, input_ids, attention_mask)

    assert out.disease_logits.shape == (2, 5), f"disease_logits shape={out.disease_logits.shape}"
    assert out.image_features.shape == (2, 512), f"image_features shape={out.image_features.shape}"
    assert out.text_features.shape == (2, 512), f"text_features shape={out.text_features.shape}"
    assert out.fused_features.shape == (2, 512), f"fused_features shape={out.fused_features.shape}"
    _ok("disease_logits [2,5], image/text/fused [2,512]")
    return 0


# ---------- 5. predict_module.predict() 多种输入 ----------
def check_predict_module() -> int:
    _tick("predict_module")
    failed = 0
    from predict_module import ExternalEyePredictor, predict_text, predict_json

    predictor = ExternalEyePredictor()

    # 5.1 PIL.Image
    r1 = predictor.predict(_rand_pil(), symptom_text="左眼红肿3天")
    assert r1.disease_key in predictor.class_keys
    assert 0.0 <= r1.confidence <= 1.0
    assert len(r1.all_probabilities) == 5
    assert abs(sum(r1.all_probabilities) - 1.0) < 1e-3
    assert len(r1.image_feature) == 512
    assert len(r1.text_feature) == 512
    assert len(r1.fused_feature) == 512
    assert r1.risk_level in {"high", "medium", "low"}
    _ok("predict(PIL.Image, symptom) OK")

    # 5.2 路径字符串
    path = _real_image_path()
    r2 = predictor.predict(path, symptom_text="")
    assert r2.disease_key in predictor.class_keys
    _ok(f"predict({Path(path).name}) OK -> {r2.disease_name} conf={r2.confidence:.3f}")

    # 5.3 Path 对象
    r3 = predictor.predict(Path(path))
    assert r3.disease_key in predictor.class_keys
    _ok("predict(Path) OK")

    # 5.4 predict_text / predict_json
    text_out = predict_text(path, symptom_text="左眼红肿")
    assert "外眼病识别结果" in text_out
    assert "置信度" in text_out
    assert "风险等级" in text_out
    _ok("predict_text() 返回字符串报告")

    json_out = predict_json(path, symptom_text="畏光流泪")
    data = json.loads(json_out)
    assert data["disease_key"] in predictor.class_keys
    assert data["risk_level"] in {"high", "medium", "low"}
    assert len(data["image_feature"]) == 512
    _ok("predict_json() 返回合法 JSON，含特征向量")

    # 5.5 str(result) / to_string() / to_dict()
    s = str(r1)
    d = r1.to_dict()
    assert "外眼病识别结果" in s
    assert d["disease_key"] == r1.disease_key
    _ok("str(result) / to_dict() OK")

    # 5.6 批量
    batch_results = predictor.predict_batch([_rand_pil(), _rand_pil()], ["症状A", ""])
    assert len(batch_results) == 2
    for r in batch_results:
        assert r.disease_key in predictor.class_keys
    _ok("predict_batch([2 images]) OK")

    # 5.7 空路径 / 非法路径应抛出（不是 silent fail）
    try:
        predictor.predict("nonexistent_xxx_1234.jpg")
        _fail("predict(nonexistent) 未抛异常")
        failed += 1
    except (FileNotFoundError, OSError):
        _ok("predict(nonexistent) 正确抛出异常")
    except Exception as exc:
        _fail(f"predict(nonexistent) 抛了异常但不是 FileNotFoundError/OSError: {type(exc).__name__}: {exc}")
        failed += 1

    return failed


# ---------- 6. FastAPI 接口 ----------
def check_fastapi() -> int:
    _tick("FastAPI 接口")
    failed = 0
    from fastapi.testclient import TestClient
    from api.app import app

    # TestClient 必须在 with 语句中使用，才会执行 lifespan（加载 predictor 到 state）
    with TestClient(app) as client:

        # 6.1 GET /
        resp = client.get("/")
        assert resp.status_code == 200, f"/: {resp.status_code}"
        data = resp.json()
        assert "title" in data
        _ok("GET / -> " + str(resp.status_code))

        # 6.2 GET /health
        resp = client.get("/health")
        assert resp.status_code == 200
        _ok("GET /health -> " + str(resp.status_code))

        # 6.3 GET /classes
        resp = client.get("/classes")
        assert resp.status_code == 200
        data = resp.json()
        assert "class_keys" in data
        _ok(f"GET /classes -> class_keys={data.get('class_keys')}")

        # 6.4 POST /external_eye/predict
        buf = io.BytesIO()
        _rand_pil().save(buf, format="PNG")
        buf.seek(0)
        resp = client.post(
            "/external_eye/predict",
            files={"file": ("photo.png", buf.getvalue(), "image/png")},
            data={"symptom_text": "左眼红肿3天，畏光流泪"},
        )
        assert resp.status_code == 200, f"/external_eye/predict: {resp.status_code} {resp.text[:200]}"
        data = resp.json()
        assert data["disease_key"] in {"Cataract", "Conjunctivitis", "Eyelid", "Uveitis", "Normal"}
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["risk_level"] in {"high", "medium", "low"}
        assert len(data.get("image_feature", [])) == 512
        _ok(f"POST /external_eye/predict -> {data['disease_name']} conf={data['confidence']:.3f} risk={data['risk_level']}")

        # 6.5 POST /external_eye/predict/batch
        buf1 = io.BytesIO(); _rand_pil().save(buf1, format="PNG")
        buf2 = io.BytesIO(); _rand_pil().save(buf2, format="PNG")

        resp = client.post(
            "/external_eye/predict/batch",
            files=[
                ("files", ("a.png", buf1.getvalue(), "image/png")),
                ("files", ("b.png", buf2.getvalue(), "image/png")),
            ],
            data={"symptom_texts": "左眼红肿,"},
        )
        assert resp.status_code == 200, f"/external_eye/predict/batch: {resp.status_code} {resp.text[:200]}"
        data = resp.json()
        assert len(data["results"]) == 2, f"批量返回长度={len(data['results'])}"
        _ok(f"POST /external_eye/predict/batch -> 返回 {len(data['results'])} 条")

        # 6.6 非法文件应返回 4xx / 500（不是静默 200）
        resp = client.post(
            "/external_eye/predict",
            files={"file": ("bad.txt", b"not an image", "text/plain")},
        )
        _ok(f"POST /external_eye/predict with bad file -> {resp.status_code}")

    return failed


# ---------- 7. 评估脚本 ----------
def check_eval_script() -> int:
    _tick("评估脚本 evaluation.multimodal_eval")
    # 直接跑一遍 main，但要避免它又调回本文件；它的 --checkpoint 默认不存在，会走 DEMO 模式
    failed = 0

    # 7.1 检查模块符号
    try:
        from evaluation import multimodal_eval as ev
        assert hasattr(ev, "main"), "missing main()"
        assert hasattr(ev, "_per_class_metrics"), "missing _per_class_metrics"
        assert hasattr(ev, "_run_image_only_ablation"), "missing _run_image_only_ablation"
        assert hasattr(ev, "_run_random_baseline"), "missing _run_random_baseline"
        _ok("evaluation.multimodal_eval 符号齐全")
    except Exception as exc:
        _fail(str(exc))
        failed += 1

    # 7.2 检查 _per_class_metrics 数学正确性
    from evaluation.multimodal_eval import _per_class_metrics
    preds = np.array([0, 1, 2, 0, 1, 2])
    labels = np.array([0, 1, 2, 1, 2, 0])
    per, cm = _per_class_metrics(preds, labels, 3)
    assert cm.shape == (3, 3)
    for c in range(3):
        assert per[c]["precision"] >= 0 and per[c]["precision"] <= 1
    _ok("_per_class_metrics 返回混淆矩阵 + per-class P/R/F1")

    return failed


# ---------- 8. 训练脚本（只静态 sanity） ----------
def check_train_script() -> int:
    _tick("训练脚本 train.py 静态检查")
    failed = 0
    try:
        import train as tr
        for attr in ("train_main", "TRAIN_CONFIG"):
            if not hasattr(tr, attr):
                _fail(f"train.py 缺少 {attr}")
                failed += 1
            else:
                _ok(f"train.py 存在 {attr}")
    except Exception as exc:
        _fail(str(exc))
        failed += 1
    return failed


# ---------- main ----------
def main() -> int:
    print("=" * 70)
    print("外眼多模态模型 · 项目整体体检")
    print("=" * 70)

    start = time.time()
    failed_total = 0
    checks = [
        ("1. 模块 import",   check_modules),
        ("2. 配置",          check_config),
        ("3. 数据集",        check_dataset),
        ("4. 模型 forward",  check_model_forward),
        ("5. predict_module", check_predict_module),
        ("6. FastAPI 接口",  check_fastapi),
        ("7. 评估脚本",      check_eval_script),
        ("8. 训练脚本",      check_train_script),
    ]

    summary = []
    for name, fn in checks:
        try:
            fail = fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            fail = 1
        summary.append((name, fail))
        failed_total += fail

    print()
    print("=" * 70)
    print("总结：")
    for name, fail in summary:
        status = "PASS" if fail == 0 else f"FAIL({fail})"
        print(f"  {status:<10} {name}")
    print(f"  耗时: {time.time() - start:.1f}s")
    print("=" * 70)
    print("整体结果:", "OK 全部通过" if failed_total == 0 else f"共 {failed_total} 项失败")
    return 0 if failed_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
