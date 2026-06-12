import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny

# ---- 基础配置 ----
IMAGE_SIZE = 384    
NUM_CLASSES = 8     
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 8个类别的中文名称
CLASS_CHINESE = [
    "正常眼底 (N)", "糖尿病视网膜病变 (D)", "青光眼 (G)", "白内障 (C)",
    "年龄相关性黄斑变性 (A)", "高血压眼底病变 (H)", "近视黄斑病变 (M)", "其他眼病 (O)"
]


# ---- 模型结构 ----
class ODIRModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.model = convnext_tiny(weights=None) 
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# 封装类
class ODIRPredictor:
    """
    眼底图像预测类，支持直接 import 调用
    """
    def __init__(self, model_path="./models/convnext_odir/odir_convnext_best.pth"):
        """
        初始化：加载训练好的权重文件，并开启测试模式
        """
        self.device = DEVICE
        self.model = ODIRModel(num_classes=NUM_CLASSES)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到权重文件: {model_path}，请检查路径。")
            
        # 加载最优权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # 关闭 Dropout

        # 图片预处理流（缩放到384，转张量，做归一化）
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
            transforms.ToTensor(),                       
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _apply_clahe(self, img_pil: Image.Image) -> Image.Image:
        """ 眼底图自适应直方图均衡化（提升血管细节对比度） """
        img = np.array(img_pil)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    def predict(self, image_input) -> dict:
        """
        单张图片预测方法
        :param image_input: 可以是本地图片路径字符串，也可以是 PIL Image 对象
        :return: 返回包含预测疾病和全类概率的字典
        """
        try:
            # 兼容路径输入或直接图片对象输入
            if isinstance(image_input, str):
                img = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                img = image_input.convert('RGB')
            else:
                return {"status": "error", "message": "不支持的图片格式"}

            # 图片预处理与模型推理
            img_enhanced = self._apply_clahe(img)
            img_tensor = self.transform(img_enhanced).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(img_tensor)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # 解析结果
            pred_idx = int(probabilities.argmax())
            all_distribution = {CLASS_CHINESE[i]: round(float(probabilities[i]), 4) for i in range(NUM_CLASSES)}
            
            return {
                "status": "success",
                "prediction": {
                    "class_index": pred_idx,
                    "disease_name": CLASS_CHINESE[pred_idx],
                    "confidence": round(float(probabilities[pred_idx]), 4)
                },
                "probability_distribution": all_distribution
            }
        except Exception as e:
            return {"status": "error", "message": f"推理失败: {str(e)}"}
