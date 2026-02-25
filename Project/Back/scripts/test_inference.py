"""
手语识别推理测试脚本
用法:
  python test_inference.py --checkpoint 模型路径.pth --test-image 图片路径.jpg
  python test_inference.py --checkpoint 模型路径.pth --test-video 视频路径.mp4
  python test_inference.py --checkpoint 模型路径.pth --test-dir 图片文件夹
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import cv2
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("  Sign Language Recognition - Inference Test")
print("="*70)
print(f"\nDevice: {DEVICE}")


class SignLanguageVideoClassifier(nn.Module):
    """视频分类器"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        self.backbone.fc = nn.Identity()
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        features = self.backbone(x)
        features = features.reshape(b, t, -1)
        features = features.mean(dim=1)
        return self.fc(features)


def load_model(checkpoint_path: str):
    """加载模型"""
    print(f"\n加载模型: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # 尝试获取类别数
    if 'label_encoder_classes' in checkpoint:
        num_classes = len(checkpoint['label_encoder_classes'])
        print(f"  类别数 (from checkpoint): {num_classes}")
    else:
        num_classes = 500
        print(f"  使用默认类别数: {num_classes}")
    
    # 创建模型
    model = SignLanguageVideoClassifier(num_classes).to(DEVICE)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print("  模型加载成功!")
    
    return model, checkpoint


def load_label_map(checkpoint_path: str) -> Dict[int, str]:
    """加载标签映射"""
    label_map_path = str(Path(checkpoint_path).parent / "label_map.json")
    
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        print(f"  从 {label_map_path} 加载标签映射")
        return {int(k): v for k, v in label_map.items()}
    
    # 尝试从检查点加载
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'label_encoder_classes' in checkpoint:
        classes = checkpoint['label_encoder_classes']
        return {i: c for i, c in enumerate(classes)}
    
    print("  未找到标签映射，使用默认索引")
    return {}


def preprocess_image(image_path: str, frame_num: int = 8) -> torch.Tensor:
    """预处理图片"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # 转换为 tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)
    
    return img_tensor.to(DEVICE)


def preprocess_video(video_path: str, frame_num: int = 8) -> torch.Tensor:
    """预处理视频"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"视频为空: {video_path}")
    
    # 均匀采样帧
    frame_indices = np.linspace(0, total_frames - 1, frame_num, dtype=int)
    frames = []
    
    # 归一化参数
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            
            # 手动归一化
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - mean) / std
            
            frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"无法读取视频帧: {video_path}")
    
    # 转换为 tensor: (T, H, W, C) -> (C, T, H, W)
    frames_array = np.array(frames)
    frames_tensor = torch.from_numpy(frames_array).float()
    frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
    frames_tensor = frames_tensor.unsqueeze(0)  # (1, C, T, H, W)
    
    return frames_tensor.to(DEVICE)


def inference(model, input_tensor: torch.Tensor, label_map: Dict[int, str] = None) -> Tuple[str, float]:
    """推理"""
    model.eval()
    
    with torch.no_grad():
        if DEVICE.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                output = model(input_tensor)
        else:
            output = model(input_tensor)
        
        probs = torch.softmax(output, dim=1)
        top_prob, top_idx = probs[0].max(0)
        
        idx = top_idx.item()
        prob = top_prob.item()
        
        if label_map and idx in label_map:
            gloss = label_map[idx]
        else:
            gloss = f"Class_{idx}"
    
    return gloss, prob


def test_image(model, image_path: str, label_map: Dict[int, str] = None):
    """测试单张图片"""
    print(f"\n{'='*70}")
    print(f"测试图片: {image_path}")
    print("="*70)
    
    try:
        img_tensor = preprocess_image(image_path)
        gloss, prob = inference(model, img_tensor, label_map)
        
        print(f"\n  预测结果:")
        print(f"    手语词汇: {gloss}")
        print(f"    置信度: {prob*100:.2f}%")
        
        return gloss, prob
        
    except Exception as e:
        print(f"  错误: {e}")
        return None, None


def test_video(model, video_path: str, label_map: Dict[int, str] = None):
    """测试视频"""
    print(f"\n{'='*70}")
    print(f"测试视频: {video_path}")
    print("="*70)
    
    try:
        video_tensor = preprocess_video(video_path)
        gloss, prob = inference(model, video_tensor, label_map)
        
        print(f"\n  预测结果:")
        print(f"    手语词汇: {gloss}")
        print(f"    置信度: {prob*100:.2f}%")
        
        return gloss, prob
        
    except Exception as e:
        print(f"  错误: {e}")
        return None, None


def test_directory(model, dir_path: str, label_map: Dict[int, str] = None):
    """测试目录中的所有图片"""
    print(f"\n{'='*70}")
    print(f"测试目录: {dir_path}")
    print("="*70)
    
    # 支持的图片格式
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # 获取所有图片文件
    image_files = []
    for f in os.listdir(dir_path):
        ext = Path(f).suffix.lower()
        if ext in extensions:
            image_files.append(os.path.join(dir_path, f))
    
    image_files.sort()
    
    if not image_files:
        print(f"  目录中没有找到图片文件")
        return
    
    print(f"  找到 {len(image_files)} 张图片\n")
    
    results = []
    for img_path in tqdm(image_files, desc="识别中"):
        try:
            img_tensor = preprocess_image(img_path)
            gloss, prob = inference(model, img_tensor, label_map)
            results.append({
                'file': os.path.basename(img_path),
                'gloss': gloss,
                'confidence': prob
            })
            print(f"  {os.path.basename(img_path)}: {gloss} ({prob*100:.2f}%)")
        except Exception as e:
            print(f"  {os.path.basename(img_path)}: 错误 - {e}")
            results.append({
                'file': os.path.basename(img_path),
                'gloss': 'ERROR',
                'confidence': 0
            })
    
    # 保存结果
    result_path = os.path.join(dir_path, "inference_results.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n  结果已保存到: {result_path}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='手语识别推理测试')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--test-image', type=str, default='', help='测试图片路径')
    parser.add_argument('--test-video', type=str, default='', help='测试视频路径')
    parser.add_argument('--test-dir', type=str, default='', help='测试图片目录')
    args = parser.parse_args()
    
    if not any([args.test_image, args.test_video, args.test_dir]):
        parser.print_help()
        print("\n示例用法:")
        print("  python test_inference.py --checkpoint model.pth --test-image test.jpg")
        print("  python test_inference.py --checkpoint model.pth --test-video test.mp4")
        print("  python test_inference.py --checkpoint model.pth --test-dir ./test_images")
        return
    
    # 加载模型
    model, checkpoint = load_model(args.checkpoint)
    
    # 加载标签映射
    label_map = load_label_map(args.checkpoint)
    
    # 测试
    if args.test_image:
        test_image(model, args.test_image, label_map)
    
    if args.test_video:
        test_video(model, args.test_video, label_map)
    
    if args.test_dir:
        test_directory(model, args.test_dir, label_map)
    
    print(f"\n{'='*70}")
    print("  测试完成!")
    print("="*70)


if __name__ == "__main__":
    main()

