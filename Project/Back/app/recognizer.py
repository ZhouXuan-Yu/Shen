"""
手语识别推理模块
集成 ResNet 迁移学习模型到 FastAPI 后端
"""
import os
import sys
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class InferenceConfig:
    """推理配置"""
    model_path: str = "models/sign_language/sign_language_model.pth"
    label_encoder_path: str = "models/sign_language/label_encoder.pkl"
    label_map_path: str = "models/sign_language/label_map.json"
    num_classes: int = 1000
    hidden_dim: int = 512
    dropout: float = 0.5
    pretrained: bool = True
    freeze_backbone: bool = True


class SignLanguageClassifier(nn.Module):
    """手语识别分类器"""
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        
        # 加载预训练的 ResNet18
        if config.pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # 获取骨干网络输出维度
        backbone_dim = self.backbone.fc.in_features
        
        # 冻结骨干网络
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 替换分类头
        self.backbone.fc = nn.Identity()
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class SignLanguageRecognizer:
    """
    手语识别器
    
    功能:
    1. 加载训练好的模型
    2. 预处理输入图像
    3. 执行推理
    4. 返回识别结果
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """初始化识别器"""
        self.config = config or InferenceConfig()
        self.model = None
        self.label_encoder = None
        self.label_map = None
        self.transform = None
        
    def load_model(self) -> bool:
        """加载模型"""
        model_path = PROJECT_ROOT / self.config.model_path
        
        if not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return False
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        # 更新配置
        self.config.num_classes = checkpoint['config']['num_classes']
        self.config.hidden_dim = checkpoint['config']['hidden_dim']
        self.config.dropout = checkpoint['config']['dropout']
        
        # 创建模型
        self.model = SignLanguageClassifier(self.config).to(DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 加载标签编码器
        encoder_path = PROJECT_ROOT / self.config.label_encoder_path
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        
        # 加载标签映射
        label_map_path = PROJECT_ROOT / self.config.label_map_path
        if label_map_path.exists():
            with open(label_map_path, 'r', encoding='utf-8') as f:
                self.label_map = json.load(f)
        
        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"模型加载成功！")
        print(f"  类别数量: {self.config.num_classes}")
        print(f"  模型路径: {model_path}")
        
        return True
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """预处理图像"""
        if self.transform is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # 转换为 RGB（如果是灰度图）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 预处理
        tensor = self.transform(image)
        # 添加批次维度
        tensor = tensor.unsqueeze(0).to(DEVICE)
        return tensor
    
    def recognize(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """
        识别手语图像
        
        Args:
            image: 输入图像 (PIL Image)
            top_k: 返回前 k 个结果
        
        Returns:
            识别结果列表
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # 预处理
        input_tensor = self.preprocess_image(image)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # 获取 top-k 结果
            top_probs, top_indices = probs.topk(top_k)
        
        results = []
        for i in range(top_k):
            prob = top_probs[0][i].item()
            idx = top_indices[0][i].item()
            
            # 获取标签
            if self.label_map and str(idx) in self.label_map:
                label = self.label_map[str(idx)]
            elif self.label_encoder:
                label = self.label_encoder.inverse_transform([idx])[0]
            else:
                label = f"未知类别_{idx}"
            
            results.append({
                "text": label,
                "confidence": round(prob * 100, 2),
                "class_id": idx
            })
        
        return results
    
    def recognize_video_frame(self, frame: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        识别视频帧
        
        Args:
            frame: 输入帧 (numpy array, HWC)
            top_k: 返回前 k 个结果
        
        Returns:
            识别结果列表
        """
        # 转换为 PIL Image
        image = Image.fromarray(frame)
        return self.recognize(image, top_k)


# 全局识别器实例
recognizer = None


def get_recognizer() -> SignLanguageRecognizer:
    """获取全局识别器实例"""
    global recognizer
    if recognizer is None:
        recognizer = SignLanguageRecognizer()
        recognizer.load_model()
    return recognizer


if __name__ == "__main__":
    # 测试推理模块
    print("=" * 60)
    print("手语识别推理模块测试")
    print("=" * 60)
    
    # 创建识别器
    recognizer = SignLanguageRecognizer()
    
    # 加载模型
    if recognizer.load_model():
        # 创建一个测试图像
        test_image = Image.new('RGB', (224, 224), color=(73, 109, 137))
        
        # 识别
        results = recognizer.recognize(test_image, top_k=5)
        
        print("\n识别结果:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['text']} - 置信度: {result['confidence']}%")

