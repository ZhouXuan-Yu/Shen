"""
CE-CSL 手语识别训练脚本 - 优化版
特性：
1. 多进程数据预加载，提高 GPU 利用率
2. 支持 Ctrl+C 中断，自动保存最佳模型
3. 支持测试模式（图片/视频）
4. 支持断点续训
"""
import os
import sys
import json
import random
import time
import gc
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score
)
import cv2

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from visualizer import PaperVisualizer, reset_visualizer

# 全局变量用于处理中断
INTERRUPTED = False

def signal_handler(sig, frame):
    """处理 Ctrl+C 中断"""
    global INTERRUPTED
    INTERRUPTED = True
    print("\n\n检测到中断信号，正在保存模型...")
    
# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("  Sign Language Recognition - Optimized Training")
print("="*70)
print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "resnet50"
    num_samples: int = 1000
    test_size: float = 0.2
    num_classes: int = 500
    hidden_dim: int = 512
    dropout: float = 0.4
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 50
    eval_interval: int = 1
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_epoch: int = 15
    label_smoothing: float = 0.1
    use_amp: bool = True
    gradient_clip: float = 1.0
    frame_num: int = 8
    num_workers: int = 4  # 多进程数据加载
    prefetch_factor: int = 2  # 预加载因子
    test_only: bool = False  # 仅测试模式
    test_image: str = ""  # 测试图片路径
    test_video: str = ""  # 测试视频路径
    checkpoint: str = ""  # 断点续训模型路径


class VideoFrameDataset(Dataset):
    """视频帧数据集 - 优化版"""
    
    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        frame_num: int = 8,
        cache_frames: bool = False  # 缓存帧到内存
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.frame_num = frame_num
        self.cache_frames = cache_frames
        self.frame_cache = {} if cache_frames else None
        
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def _extract_frames(self, video_path: str) -> torch.Tensor:
        """从视频中提取帧 - 优化版"""
        if not os.path.exists(video_path):
            return torch.randn(3, self.frame_num, 224, 224)
        
        # 检查缓存
        if self.cache_frames and video_path in self.frame_cache:
            return self.frame_cache[video_path]
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return torch.randn(3, self.frame_num, 224, 224)
        
        # 均匀采样帧
        frame_indices = np.linspace(0, total_frames - 1, self.frame_num, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return torch.randn(3, self.frame_num, 224, 224)
        
        # 转换为 tensor
        frames_tensor = torch.from_numpy(np.array(frames)).float()
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        
        # 缓存
        if self.cache_frames:
            self.frame_cache[video_path] = frames_tensor
        
        return frames_tensor
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self._extract_frames(video_path)
        
        return frames, label


class ImageDataset(Dataset):
    """图片数据集 - 用于测试和推理"""
    
    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
        
        if self.transform:
            img = self.transform(img)
        
        return img, img_path


class SignLanguageVideoClassifier(nn.Module):
    """视频分类器"""
    
    def __init__(self, config: ModelConfig, num_classes: int):
        super().__init__()
        self.config = config
        
        if config.pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        for param in self.backbone.parameters():
            param.requires_grad = not config.freeze_backbone
        
        self.backbone.fc = nn.Identity()
        
        self.fc = nn.Sequential(
            nn.Linear(2048, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        
        x = x.view(b * t, c, h, w)
        features = self.backbone(x)
        features = features.view(b, t, -1)
        
        features = features.mean(dim=1)
        
        return self.fc(features)


def prepare_video_data(config: ModelConfig) -> Tuple[VideoFrameDataset, VideoFrameDataset, LabelEncoder]:
    """准备视频数据"""
    print(f"\n[1/5] 准备视频数据...")
    
    base_path = 'D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL'
    video_base = f'{base_path}/video'
    label_path = f'{base_path}/label/train.csv'
    
    import pandas as pd
    df = pd.read_csv(label_path)
    
    print(f"  CSV 标签数: {len(df)}")
    
    all_glosses = []
    for gloss_str in df['Gloss'].dropna():
        if isinstance(gloss_str, str) and gloss_str.strip():
            glosses = [g.strip() for g in gloss_str.split('/') if g.strip()]
            all_glosses.extend(glosses)
    
    unique_glosses = list(set(all_glosses))
    print(f"  唯一词汇数: {len(unique_glosses)}")
    
    from collections import Counter
    gloss_counts = Counter(all_glosses)
    top_glosses = [g for g, _ in gloss_counts.most_common(config.num_classes)]
    
    gloss_to_idx = {g: i for i, g in enumerate(top_glosses)}
    
    print(f"  使用类别数: {len(top_glosses)}")
    
    video_paths = []
    labels = []
    
    for idx, row in df.iterrows():
        gloss_str = row['Gloss']
        if not isinstance(gloss_str, str) or not gloss_str.strip():
            continue
        
        glosses = [g.strip() for g in gloss_str.split('/') if g.strip()]
        main_gloss = glosses[0] if glosses else None
        
        if main_gloss not in gloss_to_idx:
            continue
        
        video_num = row['Number'].replace('train-', '')
        translator = row['Translator']
        video_path = f'{video_base}/train/{translator}/train-{video_num}.mp4'
        
        if os.path.exists(video_path):
            video_paths.append(video_path)
            labels.append(gloss_to_idx[main_gloss])
    
    print(f"  有效视频数: {len(video_paths)}")
    
    if len(video_paths) < 100:
        raise ValueError(f"有效视频太少: {len(video_paths)}")
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(top_glosses)
    
    n_test = int(len(video_paths) * config.test_size)
    indices = list(range(len(video_paths)))
    random.shuffle(indices)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    train_videos = [video_paths[i] for i in train_idx]
    test_videos = [video_paths[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]
    
    print(f"  训练集: {len(train_videos)} 视频")
    print(f"  测试集: {len(test_videos)} 视频")
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = VideoFrameDataset(train_videos, train_labels, transform=train_transform, frame_num=config.frame_num)
    test_dataset = VideoFrameDataset(test_videos, test_labels, transform=test_transform, frame_num=config.frame_num)
    
    return train_dataset, test_dataset, label_encoder


def train_epoch(model, dataloader, criterion, optimizer, scaler, epoch, config, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"  Epoch {epoch}")
    
    batch_count = 0
    for batch_idx, (frames, labels) in enumerate(progress_bar):
        if INTERRUPTED:
            break
            
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if config.use_amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(frames)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        batch_count += 1
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'acc': f'{100.*correct/total:.2f}%'
        })
        
        if device.type == 'cuda' and batch_idx % 20 == 0:
            torch.cuda.synchronize()
            gc.collect()
    
    return total_loss / batch_count, 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    batch_count = 0
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="  Evaluating"):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            batch_count += 1
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100. * accuracy_score(all_labels, all_preds)
    
    return total_loss / batch_count, accuracy, all_preds, all_labels


def calculate_metrics(all_labels, all_preds):
    """计算评估指标"""
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'confusion_matrix': cm,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, output_dir, label_encoder):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
        'label_encoder_classes': label_encoder.classes_.tolist() if hasattr(label_encoder, 'classes_') else []
    }
    torch.save(checkpoint, output_dir / "latest_checkpoint.pth")
    print(f"  已保存检查点: {output_dir / 'latest_checkpoint.pth'}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_acc']


def test_image(model, image_path: str, label_encoder, device):
    """测试单张图片"""
    print(f"\n测试图片: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"  文件不存在!")
        return
    
    # 加载图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"  无法读取图片!")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)  # (1, C, T, H, W)
    
    # 如果没有时间维度，添加一个
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.unsqueeze(2)  # (1, C, 1, H, W)
    
    model.eval()
    with torch.no_grad():
        if DEVICE.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                output = model(img_tensor)
        else:
            output = model(img_tensor)
        
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = probs.topk(5, dim=1)
    
    print(f"\n  预测结果 (Top 5):")
    for i in range(5):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        gloss = label_encoder.classes_[idx] if idx < len(label_encoder.classes_) else f"Class_{idx}"
        print(f"    {i+1}. {gloss}: {prob*100:.2f}%")


def test_video(model, video_path: str, label_encoder, device, frame_num: int = 8):
    """测试视频"""
    print(f"\n测试视频: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"  文件不存在!")
        return
    
    # 提取帧
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print("  视频为空!")
        cap.release()
        return
    
    # 均匀采样
    frame_indices = np.linspace(0, total_frames - 1, frame_num, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        print("  无法读取视频帧!")
        return
    
    # 转换为 tensor
    frames_tensor = torch.from_numpy(np.array(frames)).float()
    frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
    frames_tensor = frames_tensor.unsqueeze(0).to(device)  # (1, C, T, H, W)
    
    transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    frames_tensor = transform(frames_tensor)
    
    model.eval()
    with torch.no_grad():
        if DEVICE.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                output = model(frames_tensor)
        else:
            output = model(frames_tensor)
        
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = probs.topk(5, dim=1)
    
    print(f"\n  视频预测结果 (Top 5):")
    for i in range(5):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        gloss = label_encoder.classes_[idx] if idx < len(label_encoder.classes_) else f"Class_{idx}"
        print(f"    {i+1}. {gloss}: {prob*100:.2f}%")


def main():
    """主函数"""
    print("="*70)
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--test-only', action='store_true', help='仅测试模式')
    parser.add_argument('--test-image', type=str, default='', help='测试图片路径')
    parser.add_argument('--test-video', type=str, default='', help='测试视频路径')
    parser.add_argument('--checkpoint', type=str, default='', help='断点续训模型路径')
    parser.add_argument('--classes', type=int, default=500, help='类别数量')
    args = parser.parse_args()
    
    # 输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = PROJECT_ROOT / "models" / "sign_language" / f"optimized_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results" / "plots" / f"optimized_{timestamp}"
    
    # 初始化可视化器
    reset_visualizer()
    visualizer = PaperVisualizer(str(results_dir))
    
    # 配置
    config = ModelConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_classes=args.classes,
        num_workers=4 if DEVICE.type == 'cuda' else 0,
        test_only=args.test_only,
        test_image=args.test_image,
        test_video=args.test_video,
        checkpoint=args.checkpoint
    )
    
    print(f"\n训练配置:")
    print(f"  - 模型: {config.model_name}")
    print(f"  - 训练轮数: {config.num_epochs}")
    print(f"  - 批次大小: {config.batch_size}")
    print(f"  - 学习率: {config.learning_rate}")
    print(f"  - 数据加载进程: {config.num_workers}")
    
    # 准备数据
    train_dataset, test_dataset, label_encoder = prepare_video_data(config)
    config.num_classes = len(label_encoder.classes_)
    
    # 数据加载器 - 优化版
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0
    )
    
    # 创建模型
    print(f"\n[2/5] 创建 {config.model_name} 模型...")
    model = SignLanguageVideoClassifier(config, config.num_classes).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    # 优化器
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.learning_rate * 0.1},
        {'params': classifier_params, 'lr': config.learning_rate}
    ], weight_decay=config.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if config.use_amp and DEVICE.type == 'cuda' else None
    
    # 断点续训
    start_epoch = 1
    best_acc = 0.0
    
    if config.checkpoint and os.path.exists(config.checkpoint):
        print(f"\n加载检查点: {config.checkpoint}")
        start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, config.checkpoint)
        print(f"  从 Epoch {start_epoch} 继续训练")
        print(f"  当前最佳准确率: {best_acc:.2f}%")
    
    # 测试模式
    if config.test_image or config.test_video:
        print("\n" + "="*70)
        print("  测试模式")
        print("="*70)
        
        if not config.checkpoint:
            print("错误: 测试模式需要指定 --checkpoint 参数加载模型!")
            return
        
        if config.test_image:
            test_image(model, config.test_image, label_encoder, DEVICE)
        
        if config.test_video:
            test_video(model, config.test_video, label_encoder, DEVICE, config.frame_num)
        
        return
    
    # 训练
    print(f"\n[3/5] 开始训练 (按 Ctrl+C 中断并保存)...")
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        if INTERRUPTED:
            print(f"\n检测到中断信号!")
            # 保存中断时的状态
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, output_dir, label_encoder)
            print(f"已保存检查点，可以恢复训练")
            break
        
        epoch_start = time.time()
        
        if epoch == config.unfreeze_epoch:
            print("  解冻骨干网络...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            for pg in optimizer.param_groups:
                pg['lr'] /= 5
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, config, DEVICE
        )
        
        if INTERRUPTED:
            break
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % config.eval_interval == 0 or epoch == config.num_epochs:
            val_loss, val_acc, all_preds, all_labels = evaluate(
                model, test_loader, criterion, DEVICE
            )
            
            metrics = calculate_metrics(all_labels, all_preds)
            
            visualizer.log_training(
                epoch, train_loss, val_loss, train_acc, val_acc,
                current_lr, time.time() - epoch_start
            )
            
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                best_epoch = epoch
                # 保存最佳模型
                torch.save(model.state_dict(), output_dir / "best_model.pth")
            
            print(f"\n  Epoch {epoch}/{config.num_epochs}:")
            print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"    Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"    Precision: {metrics['precision_macro']:.4f}")
            print(f"    F1 Score: {metrics['f1_macro']:.4f}")
            print(f"    LR: {current_lr:.2e}")
            if is_best:
                print(f"    [NEW BEST!]")
        
        if DEVICE.type == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()
        
        # 每5轮保存一次检查点
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, output_dir, label_encoder)
    
    total_time = time.time() - start_time
    print(f"\n[4/5] 训练完成!")
    print(f"  总用时: {total_time/60:.2f} 分钟")
    print(f"  最佳验证准确率: {best_acc:.2f}%")
    
    # 最终评估
    print(f"\n[5/5] 最终评估...")
    final_loss, final_acc, final_preds, final_labels = evaluate(
        model, test_loader, criterion, DEVICE
    )
    final_metrics = calculate_metrics(final_labels, final_preds)
    
    visualizer.set_confusion_matrix(final_metrics['confusion_matrix'])
    
    # 生成图表
    print("\n生成论文级可视化图表...")
    plots, report = visualizer.generate_all_plots()
    
    # 保存报告
    full_report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(DEVICE),
        'config': {
            'model': config.model_name,
            'num_samples': len(train_dataset),
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'frame_num': config.frame_num,
            'num_workers': config.num_workers,
        },
        'final_metrics': {
            'val_accuracy': final_acc,
            'precision_macro': final_metrics['precision_macro'],
            'f1_macro': final_metrics['f1_macro'],
        },
        'best_metrics': {
            'best_val_accuracy': best_acc,
            'best_epoch': best_epoch if 'best_epoch' in dir() else 0,
        },
        'training_time': {'total_time_minutes': total_time / 60},
    }
    
    report_path = output_dir / "training_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    # 保存标签映射
    label_map = {i: c for i, c in enumerate(label_encoder.classes_)}
    with open(output_dir / "label_map.json", 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("  Training Complete!")
    print("="*70)
    print(f"\n结果保存:")
    print(f"  - 模型目录: {output_dir}")
    print(f"  - 最佳模型: {output_dir / 'best_model.pth'}")
    print(f"  - 标签映射: {output_dir / 'label_map.json'}")
    print(f"  - 图表目录: {results_dir}")
    print("\n使用以下命令恢复训练:")
    print(f"  python scripts/train_model_video.py --checkpoint {output_dir / 'latest_checkpoint.pth'}")
    print("="*70)


if __name__ == "__main__":
    main()
