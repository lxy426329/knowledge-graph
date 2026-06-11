"""
外眼多模态模型 — 命令行入口。

菜单:
    1. 启动 API 服务器
    2. 训练 EfficientNet-B3 + BERT 多模态模型
    3. 对本地图片做一次推理
    4. 查看数据集统计
    5. 跑完整评估（含消融实验）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 在所有导入之前设置 HuggingFace 镜像源（无网络或网络受限环境）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    print("=" * 60)
    print("外眼识别系统 — EfficientNet-B3 + BERT 多模态模型")
    print("=" * 60)
    print()
    print("请选择运行模式:")
    print("  1. 启动 API 服务器")
    print("  2. 训练多模态模型")
    print("  3. 本地图片推理")
    print("  4. 查看数据集统计")
    print("  5. 跑完整评估（含消融实验）")

    choice = input("\n请输入选项 (1-5): ").strip()

    if choice == "1":
        run_api_server()
    elif choice == "2":
        run_train()
    elif choice == "3":
        run_predict()
    elif choice == "4":
        show_dataset_stats()
    elif choice == "5":
        run_evaluation()
    else:
        print("无效选项")


def run_api_server():
    from api.app import app
    import uvicorn
    from config import API_CONFIG

    print(f"启动 API 服务器: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print("文档: /docs")
    uvicorn.run(app, host=API_CONFIG["host"], port=API_CONFIG["port"])


def run_train():
    print("\n训练 EfficientNet-B3 + BERT 多模态模型 ...")
    print("产物: checkpoints/best.pt, checkpoints/class_map.json, runs/train_*.log")
    print()
    from train import train_main
    train_main()


def run_predict():
    image_path = input("图片路径 (jpg/png): ").strip().strip('"')
    symptom = input("症状文本（可留空）: ").strip()

    if not Path(image_path).exists():
        print(f"图片不存在: {image_path}")
        return

    from predict_module import ExternalEyePredictor
    predictor = ExternalEyePredictor()
    r = predictor.predict(image_path, symptom_text=symptom)

    print("\n" + "-" * 50)
    print(f"类别: {r.disease_name} ({r.disease_key})  置信度={r.confidence:.3f}")
    print(f"风险: {r.risk_level}  ({r.risk_reason})")
    print(f"融合特征维度: {len(r.fused_feature)}")
    print("各分类概率:")
    for name, p in zip(r.all_class_names, r.all_probabilities):
        bar = "█" * int(round(p * 30))
        print(f"  {name:<10} {p:6.3f}  {bar}")
    print("-" * 50)


def show_dataset_stats():
    from config import EXTERNAL_EYE_DATA_DIR, IMAGE_EXTS, CLASS_MAP

    root = EXTERNAL_EYE_DATA_DIR
    if not root.exists():
        print(f"数据集目录不存在: {root}")
        print(f"请在 {root}/<类别名>/ 下放图片，可选类别: {list(CLASS_MAP.keys())}")
        return

    print("\n数据集统计")
    print("-" * 40)
    total = 0
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        count = sum(1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        total += count
        cn = CLASS_MAP.get(class_dir.name, class_dir.name)
        print(f"  {class_dir.name:<16} ({cn:<8}) {count} 张")
    print(f"\n总计: {total} 张图像")


def run_evaluation():
    from evaluation.multimodal_eval import main as eval_main
    # 走模块默认参数，避免重复解析 sys.argv
    sys.argv = [sys.argv[0]]
    eval_main()


if __name__ == "__main__":
    main()
