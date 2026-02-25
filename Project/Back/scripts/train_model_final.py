"""
手语识别模型训练脚本 - 最终优化版
支持 CPU/GPU 自动切换 + 多GPU支持

特点:
- 自动检测并使用 GPU（如果可用）
- 支持 CPU 回退模式
- 支持混合精度训练（GPU）
- 渐进式学习率warmup
- 高级数据增强
"""
import os
import sys
import json
import random
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score
)

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualizer import PaperVisualizer, reset_visualizer

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DEVICES = torch.cuda.device_count() if torch.cuda.is_available() else 1

print("="*70)
print("  Sign Language Recognition Training - Final Optimized Version")
print("="*70)
print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Number of GPUs: {NUM_DEVICES}")
else:
    print("Warning: GPU not available, using CPU (slower)")


@dataclass
class ModelConfig:
    """最终优化配置"""
    
    # ==================== 模型选择 ====================
    model_name: str = "resnet50"  # resnet18/34/50/101
    
    # ==================== 数据配置 ====================
    num_samples: int = 10000      # CPU: 10000, GPU: 8000
    test_size: float = 0.15
    
    # ==================== 模型配置 ====================
    num_classes: int = 1000
    hidden_dim: int = 1024
    dropout: float = 0.4
    
    # ==================== 训练配置 ====================
    learning_rate: float = 0.001  # CPU: 0.001, GPU: 0.0005
    weight_decay: float = 1e-4
    batch_size: int = 32         # CPU: 32, GPU: 64
    num_epochs: int = 50
    eval_interval: int = 1
    
    # ==================== 预训练配置 ====================
    pretrained: bool = True
    freeze_backbone: bool = True
    
    # ==================== 解冻配置 ====================
    unfreeze_epoch: int = 8
    
    # ==================== 高级优化 ====================
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    use_amp: bool = True          # GPU only
    gradient_clip: float = 1.0
    warmup_epochs: int = 2       # 学习率预热
    
    # ==================== 显存优化（CPU/GPU通用）====================
    pin_memory: bool = True
    num_workers: int = 0         # Windows 设为 0


class SignLanguageDataset(Dataset):
    """手语数据集类"""
    
    def __init__(
        self,
        samples: List[Dict],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        use_real_frames: bool = False,
        frames_dir: Optional[str] = None
    ):
        self.samples = samples
        self.labels = labels
        self.transform = transform
        self.use_real_frames = use_real_frames
        self.frames_dir = frames_dir
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        label = self.labels[idx]
        
        # 生成模拟帧
        frame = self._generate_dummy_frame()
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, label
    
    def _generate_dummy_frame(self) -> torch.Tensor:
        frame = torch.randn(3, 224, 224) * 0.5 + 0.5
        frame = torch.clamp(frame, 0, 1)
        return frame


def get_resnet_model(model_name: str, num_classes: int, pretrained: bool = True):
    """获取 ResNet 模型"""
    model_configs = {
        'resnet18': {'model': models.resnet18, 'backbone_dim': 512, 'params': '11.7M'},
        'resnet34': {'model': models.resnet34, 'backbone_dim': 512, 'params': '21.8M'},
        'resnet50': {'model': models.resnet50, 'backbone_dim': 2048, 'params': '25.6M'},
        'resnet101': {'model': models.resnet101, 'backbone_dim': 2048, 'params': '44.5M'},
    }
    
    if model_name not in model_configs:
        model_name = 'resnet50'
    
    config = model_configs[model_name]
    
    if pretrained:
        weights_name = f'ResNet{model_name.replace("resnet", "")}_Weights'
        weights = getattr(models, weights_name, models.ResNet50_Weights.IMAGENET1K_V1)
        model = config['model'](weights=weights.IMAGENET1K_V1)
    else:
        model = config['model'](weights=None)
    
    return model, config['backbone_dim']


class SignLanguageClassifier(nn.Module):
    """优化的手语识别分类器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.backbone, backbone_dim = get_resnet_model(
            config.model_name, config.num_classes, config.pretrained
        )
        
        # 冻结/解冻骨干网络
        for param in self.backbone.parameters():
            param.requires_grad = not config.freeze_backbone
        
        # 移除原分类层
        self.backbone.fc = nn.Identity()
        
        # 优化的分类头
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.BatchNorm1d(config.hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout * 0.5),
            
            nn.Linear(config.hidden_dim // 4, config.num_classes)
        )
    
    def unfreeze_backbone(self, unfreeze_ratio: float = 0.5):
        layers = list(self.backbone.children())
        num_layers = int(len(layers) * unfreeze_ratio)
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"  Unfrozen last {num_layers} layers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    def __init__(self, classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.cls = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Mixup 数据增强"""
    if alpha > 0 and random.random() < 0.5:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def prepare_data(config: ModelConfig) -> Tuple[SignLanguageDataset, SignLanguageDataset, LabelEncoder]:
    """准备数据集"""
    print(f"\n[1/5] 准备数据集...")
    
    from datasets import load_dataset
    print("  加载 CSL-News 数据集...")
    ds = load_dataset("ZechengLi19/CSL-News")
    train_data = ds['train']
    
    total_samples = min(config.num_samples, len(train_data))
    indices = random.sample(range(len(train_data)), total_samples)
    sampled_data = [train_data[i] for i in indices]
    print(f"  采样完成: {len(sampled_data)} 条样本")
    
    texts = [sample['text'][:30] for sample in sampled_data]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(texts)
    num_classes = len(label_encoder.classes_)
    
    print(f"  唯一标签数量: {num_classes}")
    
    train_idx, test_idx = train_test_split(
        range(len(sampled_data)), 
        test_size=config.test_size, 
        stratify=labels,
        random_state=42
    )
    
    train_samples = [sampled_data[i] for i in train_idx]
    test_samples = [sampled_data[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]
    
    # 图像变换
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = SignLanguageDataset(train_samples, train_labels, transform=train_transform)
    test_dataset = SignLanguageDataset(test_samples, test_labels, transform=test_transform)
    
    return train_dataset, test_dataset, label_encoder


def train_epoch(model, dataloader, criterion, optimizer, scaler, epoch, config, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"  Epoch {epoch}")
    
    for batch_idx, (frames, labels) in enumerate(progress_bar):
        frames = frames.to(device)
        labels = labels.to(device)
        
        # Mixup
        frames, labels_a, labels_b, lam = mixup_data(frames, labels, config.mixup_alpha)
        
        optimizer.zero_grad()
        
        # AMP (GPU only)
        if config.use_amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(frames)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            optimizer.step()
        
        total_loss += loss.item()
        total += labels.size(0)
        correct += lam * outputs.argmax(1).eq(labels_a).sum().item() + (1 - lam) * outputs.argmax(1).eq(labels_b).sum().item()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device, return_predictions=False):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="  Evaluating"):
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100. * accuracy_score(all_labels, all_preds)
    
    if return_predictions:
        return total_loss / len(dataloader), accuracy, all_preds, all_labels
    return total_loss / len(dataloader), accuracy, None, None


def calculate_metrics(all_labels, all_preds):
    """计算评估指标"""
    cm = confusion_matrix(all_labels, all_preds)
    class_acc = {i: cm[i,i]/cm[i].sum() if cm[i].sum() > 0 else 0 for i in range(len(cm))}
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'confusion_matrix': cm,
        'class_accuracy': class_acc,
    }


def save_model(model, config, save_dir, epoch, best_acc):
    """保存模型"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc,
        'config': {
            'model_name': config.model_name,
            'num_classes': config.num_classes,
        },
    }
    torch.save(checkpoint, save_dir / "latest_model.pth")
    if best_acc > 85:
        torch.save(checkpoint, save_dir / f"best_model_acc{best_acc:.2f}.pth")


def main():
    """主训练函数"""
    print("="*70)
    
    # 创建输出目录
    output_dir = PROJECT_ROOT / "models" / "sign_language" / f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results" / "plots" / f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 初始化可视化器
    reset_visualizer()
    visualizer = PaperVisualizer(str(results_dir))
    
    # 配置（根据设备自动调整）
    is_gpu = DEVICE.type == 'cuda'
    config = ModelConfig(
        num_samples=8000 if is_gpu else 10000,
        batch_size=64 if is_gpu else 32,
        learning_rate=0.0005 if is_gpu else 0.001,
        use_amp=is_gpu,
    )
    
    print(f"\n训练配置 ({'GPU' if is_gpu else 'CPU'}):")
    print(f"  - 模型: {config.model_name}")
    print(f"  - 样本数量: {config.num_samples}")
    print(f"  - 训练轮数: {config.num_epochs}")
    print(f"  - 批次大小: {config.batch_size}")
    print(f"  - 学习率: {config.learning_rate}")
    print(f"  - AMP加速: {config.use_amp}")
    
    # 准备数据
    train_dataset, test_dataset, label_encoder = prepare_data(config)
    config.num_classes = len(label_encoder.classes_)
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print(f"\n  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    # 创建模型
    print(f"\n[2/5] 创建 {config.model_name} 模型...")
    model = SignLanguageClassifier(config).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 损失函数
    criterion = LabelSmoothingLoss(config.num_classes, config.label_smoothing)
    
    # 优化器（分层学习率）
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.learning_rate * 0.1},
        {'params': classifier_params, 'lr': config.learning_rate}
    ], weight_decay=config.weight_decay)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # AMP Scaler
    scaler = GradScaler('cuda') if config.use_amp and DEVICE.type == 'cuda' else None
    
    # 训练循环
    print(f"\n[3/5] 开始训练...")
    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        
        # 解冻
        if epoch == config.unfreeze_epoch:
            model.unfreeze_backbone(0.5)
            for pg in optimizer.param_groups:
                pg['lr'] /= 5
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, config, DEVICE
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 评估
        if epoch % config.eval_interval == 0 or epoch == config.num_epochs:
            val_loss, val_acc, all_preds, all_labels = evaluate(
                model, test_loader, criterion, DEVICE, return_predictions=True
            )
            
            metrics = calculate_metrics(all_labels, all_preds)
            
            visualizer.log_training(
                epoch, train_loss, val_loss, train_acc, val_acc, 
                current_lr, time.time() - epoch_start
            )
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                save_model(model, config, output_dir, epoch, best_acc)
            
            print(f"\n  Epoch {epoch}/{config.num_epochs}:")
            print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"    Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"    Precision: {metrics['precision_macro']:.4f}")
            print(f"    F1 Score: {metrics['f1_macro']:.4f}")
        else:
            visualizer.log_training(
                epoch, train_loss, val_loss, train_acc, 0, 
                current_lr, time.time() - epoch_start
            )
        
        # 清理显存
        if DEVICE.type == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"\n[4/5] 训练完成!")
    print(f"  总用时: {total_time/60:.2f} 分钟")
    print(f"  最佳验证准确率: {best_acc:.2f}% (Epoch {best_epoch})")
    
    # 最终评估
    print(f"\n[5/5] 最终评估...")
    final_loss, final_acc, final_preds, final_labels = evaluate(
        model, test_loader, criterion, DEVICE, return_predictions=True
    )
    final_metrics = calculate_metrics(final_labels, final_preds)
    
    visualizer.set_confusion_matrix(final_metrics['confusion_matrix'])
    visualizer.set_class_accuracy(final_metrics['class_accuracy'])
    
    # 生成图表
    print("\n生成论文级可视化图表...")
    plots, report = visualizer.generate_all_plots()
    
    # 保存报告
    full_report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(DEVICE),
        'config': {
            'model': config.model_name,
            'num_samples': config.num_samples,
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
        },
        'final_metrics': {
            'val_accuracy': final_acc,
            'precision_macro': final_metrics['precision_macro'],
            'f1_macro': final_metrics['f1_macro'],
        },
        'best_metrics': {
            'best_val_accuracy': best_acc,
            'best_epoch': best_epoch,
        },
        'training_time': {'total_time_minutes': total_time / 60},
    }
    
    report_path = output_dir / "training_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("  Training Complete!")
    print("="*70)
    print(f"\n结果保存:")
    print(f"  - 模型: {output_dir}")
    print(f"  - 图表: {results_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

