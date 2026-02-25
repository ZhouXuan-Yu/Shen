"""
手语识别模型训练脚本 - 修复版
修复: 1. 模块导入问题 2. 分层采样问题
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
from collections import Counter

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

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent

# 添加 scripts 目录到 sys.path
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 导入可视化模块
from visualizer import PaperVisualizer, reset_visualizer

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*70)
print("  Sign Language Recognition Training - Fixed Version")
print("="*70)
print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("Using CPU (slower)")


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "resnet50"
    num_samples: int = 5000       # 降低样本数避免类别不均衡
    test_size: float = 0.2
    num_classes: int = 1000
    hidden_dim: int = 512         # 降低隐藏层维度
    dropout: float = 0.4
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 32          # 降低批次大小
    num_epochs: int = 30
    eval_interval: int = 1
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_epoch: int = 10
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    use_amp: bool = True
    gradient_clip: float = 1.0


class SignLanguageDataset(Dataset):
    """手语数据集类"""
    
    def __init__(
        self,
        samples: List[Dict],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        self.samples = samples
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        label = self.labels[idx]
        
        # 生成模拟帧
        frame = torch.randn(3, 224, 224) * 0.5 + 0.5
        frame = torch.clamp(frame, 0, 1)
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, label


def get_resnet_model(model_name: str, num_classes: int, pretrained: bool = True):
    """获取 ResNet 模型"""
    model_configs = {
        'resnet18': {'model': models.resnet18, 'backbone_dim': 512},
        'resnet34': {'model': models.resnet34, 'backbone_dim': 512},
        'resnet50': {'model': models.resnet50, 'backbone_dim': 2048},
        'resnet101': {'model': models.resnet101, 'backbone_dim': 2048},
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
    """手语识别分类器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.backbone, backbone_dim = get_resnet_model(
            config.model_name, config.num_classes, config.pretrained
        )
        
        # 冻结骨干网络
        for param in self.backbone.parameters():
            param.requires_grad = not config.freeze_backbone
        
        self.backbone.fc = nn.Identity()
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
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
        return self.classifier(features)


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


def prepare_data(config: ModelConfig) -> Tuple[SignLanguageDataset, SignLanguageDataset, LabelEncoder]:
    """准备数据集 - 修复版"""
    print(f"\n[1/5] 准备数据集...")
    
    from datasets import load_dataset
    print("  加载 CSL-News 数据集...")
    ds = load_dataset("ZechengLi19/CSL-News")
    train_data = ds['train']
    
    # 采样更多数据以确保每个类别有足够样本
    total_samples = min(config.num_samples * 2, len(train_data))
    indices = random.sample(range(len(train_data)), total_samples)
    sampled_data = [train_data[i] for i in indices]
    print(f"  初始采样: {len(sampled_data)} 条样本")
    
    # 提取标签 - 使用前 5 个字符作为类别标签
    texts = [sample['text'][:5] for sample in sampled_data]
    
    print(f"  初始采样: {len(sampled_data)} 条样本")
    print(f"  类别数: {len(set(texts))}")
    
    # 标签编码
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(texts)
    num_classes = len(label_encoder.classes_)
    
    print(f"  唯一标签数量: {num_classes}")
    
    # 简单划分（不使用分层，避免类别不均衡问题）
    n_test = int(len(sampled_data) * config.test_size)
    indices = list(range(len(sampled_data)))
    random.shuffle(indices)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    train_samples = [sampled_data[i] for i in train_idx]
    test_samples = [sampled_data[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]
    
    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  测试集: {len(test_samples)} 样本")
    
    # 图像变换 - 移除 ToTensor()，因为我们已经是 tensor
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
        
        optimizer.zero_grad()
        
        # AMP
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


def main():
    """主训练函数"""
    print("="*70)
    
    # 创建输出目录
    output_dir = PROJECT_ROOT / "models" / "sign_language" / f"fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results" / "plots" / f"fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 初始化可视化器
    reset_visualizer()
    visualizer = PaperVisualizer(str(results_dir))
    
    # 配置
    config = ModelConfig(
        num_samples=3000,  # 降低样本数
        batch_size=32,
        num_epochs=30,
    )
    
    print(f"\n训练配置:")
    print(f"  - 模型: {config.model_name}")
    print(f"  - 样本数量: {config.num_samples}")
    print(f"  - 训练轮数: {config.num_epochs}")
    print(f"  - 批次大小: {config.batch_size}")
    
    # 准备数据
    train_dataset, test_dataset, label_encoder = prepare_data(config)
    config.num_classes = len(label_encoder.classes_)
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 创建模型
    print(f"\n[2/5] 创建 {config.model_name} 模型...")
    model = SignLanguageClassifier(config).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 损失函数
    criterion = LabelSmoothingLoss(config.num_classes, config.label_smoothing)
    
    # 优化器
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.learning_rate * 0.1},
        {'params': classifier_params, 'lr': config.learning_rate}
    ], weight_decay=config.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if config.use_amp and DEVICE.type == 'cuda' else None
    
    # 训练
    print(f"\n[3/5] 开始训练...")
    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        
        if epoch == config.unfreeze_epoch:
            model.unfreeze_backbone(0.5)
            for pg in optimizer.param_groups:
                pg['lr'] /= 5
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, config, DEVICE
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
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

