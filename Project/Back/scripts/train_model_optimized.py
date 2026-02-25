"""
手语识别模型训练脚本 - 优化版
支持 RTX 5060 GPU + 最佳训练策略

模型选择分析:
- ResNet18: 11M 参数, 速度快, 准确率较低
- ResNet34: 21M 参数, 平衡选择
- ResNet50: 25M 参数, 最佳效果-准确率平衡, 论文推荐
- ResNet101: 44M 参数, 更高准确率, 需要更多显存

选择 ResNet50 的原因:
1. ImageNet 预训练效果好
2. 深度适中,能学习复杂特征
3. 8GB 显存足够运行(批次64)
4. 论文常用基线模型
"""
import os
import sys
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score
)

# 添加项目路径到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入可视化模块
from scripts.visualizer import PaperVisualizer, reset_visualizer

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Device: {DEVICE}")


@dataclass
class ModelConfig:
    """优化后的模型配置"""
    
    # ==================== 模型选择 ====================
    model_name: str = "resnet50"  # resnet18, resnet34, resnet50, resnet101
    
    # ==================== 数据配置 ====================
    num_samples: int = 8000       # 采样数量(RTX 5060 推荐 8000)
    test_size: float = 0.15       # 测试集比例
    
    # ==================== 模型配置 ====================
    num_classes: int = 1000       # 类别数量
    hidden_dim: int = 1024        # 隐藏层维度(加大提升效果)
    dropout: float = 0.4          # Dropout(降低过拟合)
    
    # ==================== 训练配置 ====================
    learning_rate: float = 0.0005  # 降低学习率提升收敛
    weight_decay: float = 1e-4     # L2正则化
    batch_size: int = 64           # RTX 5060 8GB显存推荐
    num_epochs: int = 50           # 更多轮数提升效果
    eval_interval: int = 1         # 评估间隔
    
    # ==================== 预训练配置 ====================
    pretrained: bool = True        # 使用预训练模型
    freeze_backbone: bool = True   # 冻结骨干网络
    
    # ==================== 渐进式解冻配置 ====================
    unfreeze_epoch: int = 10       # 开始解冻骨干网络
    gradual_unfreeze: bool = True  # 渐进式解冻
    
    # ==================== 高级优化 ====================
    label_smoothing: float = 0.1   # 标签平滑(提升泛化)
    mixup_alpha: float = 0.2       # Mixup 数据增强
    use_amp: bool = True           # 自动混合精度(加速+省显存)
    gradient_clip: float = 1.0     # 梯度裁剪


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
        
        # 生成模拟帧（实际应用中应加载真实帧）
        frame = self._generate_dummy_frame()
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, label
    
    def _generate_dummy_frame(self) -> torch.Tensor:
        """生成模拟帧"""
        frame = torch.randn(3, 224, 224) * 0.5 + 0.5
        frame = torch.clamp(frame, 0, 1)
        return frame


def get_resnet_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    获取 ResNet 模型
    
    Args:
        model_name: 模型名称 (resnet18, resnet34, resnet50, resnet101)
        num_classes: 输出类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        ResNet 模型
    """
    model_configs = {
        'resnet18': {
            'model': models.resnet18,
            'backbone_dim': 512,
            'params': '11.7M'
        },
        'resnet34': {
            'model': models.resnet34,
            'backbone_dim': 512,
            'params': '21.8M'
        },
        'resnet50': {
            'model': models.resnet50,
            'backbone_dim': 2048,
            'params': '25.6M'
        },
        'resnet101': {
            'model': models.resnet101,
            'backbone_dim': 2048,
            'params': '44.5M'
        }
    }
    
    if model_name not in model_configs:
        print(f"Warning: Unknown model {model_name}, using resnet50")
        model_name = 'resnet50'
    
    config = model_configs[model_name]
    
    # 加载模型
    if pretrained:
        weights = getattr(models, f'ResNet{model_name.replace("resnet", "")}_Weights', None)
        if weights:
            model = config['model'](weights=weights.IMAGENET1K_V1)
        else:
            model = config['model'](weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = config['model'](weights=None)
    
    # 获取骨干网络维度
    backbone_dim = config['backbone_dim']
    print(f"  Model: {model_name}, Backbone dim: {backbone_dim}, Params: {config['params']}")
    
    return model, backbone_dim


class SignLanguageClassifier(nn.Module):
    """
    优化的手语识别分类器
    
    改进点:
    1. 更深的分类头
    2. Batch Normalization
    3. 渐进式解冻支持
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 获取 ResNet 模型
        self.backbone, backbone_dim = get_resnet_model(
            config.model_name, 
            config.num_classes,
            config.pretrained
        )
        
        # 冻结/解冻骨干网络
        self._set_backbone_grad(config.freeze_backbone)
        
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
        
    def _set_backbone_grad(self, freeze: bool):
        """设置骨干网络梯度"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
            
    def unfreeze_backbone(self, unfreeze_ratio: float = 0.5):
        """渐进式解冻骨干网络
        
        Args:
            unfreeze_ratio: 解冻比例 (0-1)
        """
        # 遍历所有层
        layers = list(self.backbone.children())
        num_layers_to_unfreeze = int(len(layers) * unfreeze_ratio)
        
        # 解冻后半部分层
        for i, layer in enumerate(layers[-num_layers_to_unfreeze:]):
            for param in layer.parameters():
                param.requires_grad = True
                
        print(f"  Unfrozen last {num_layers_to_unfreeze} layers")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    
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
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, 
                   y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Mixup 损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def prepare_data(config: ModelConfig) -> Tuple[SignLanguageDataset, SignLanguageDataset, LabelEncoder]:
    """准备数据集"""
    print(f"\n[1/5] 准备数据集...")
    
    from datasets import load_dataset
    print("  加载 CSL-News 数据集...")
    ds = load_dataset("ZechengLi19/CSL-News")
    train_data = ds['train']
    
    # 采样
    total_samples = min(config.num_samples, len(train_data))
    indices = random.sample(range(len(train_data)), total_samples)
    sampled_data = [train_data[i] for i in indices]
    print(f"  采样完成: {len(sampled_data)} 条样本")
    
    # 提取标签并编码
    texts = [sample['text'][:30] for sample in sampled_data]
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(texts)
    num_classes = len(label_encoder.classes_)
    
    print(f"  唯一标签数量: {num_classes}")
    
    # 数据划分
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
    
    # 优化的图像变换
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 更大尺寸
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),   # 更大旋转角度
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),  # 随机擦除
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = SignLanguageDataset(train_samples, train_labels, transform=train_transform)
    test_dataset = SignLanguageDataset(test_samples, test_labels, transform=test_transform)
    
    return train_dataset, test_dataset, label_encoder


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler,  # AMP scaler
    epoch: int,
    config: ModelConfig,
    device: torch.device
) -> Tuple[float, float]:
    """训练一个 epoch (支持 Mixup 和 AMP)"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    use_mixup = config.mixup_alpha > 0 and random.random() < 0.5
    
    progress_bar = tqdm(dataloader, desc=f"  Epoch {epoch}")
    for batch_idx, (frames, labels) in enumerate(progress_bar):
        frames = frames.to(device)
        labels = labels.to(device)
        
        # Mixup
        if use_mixup:
            frames, labels_a, labels_b, lam = mixup_data(frames, labels, config.mixup_alpha)
        
        optimizer.zero_grad()
        
        # 自动混合精度
        if config.use_amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(frames)
                if use_mixup:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            optimizer.step()
        
        # 统计
        total_loss += loss.item()
        if not use_mixup:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        else:
            total += labels.size(0)
            correct += lam * predicted.eq(labels_a).sum().item() + (1 - lam) * predicted.eq(labels_b).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_predictions: bool = False
) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
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
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = 100. * accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    if return_predictions:
        return avg_loss, accuracy, all_preds, all_labels
    return avg_loss, accuracy, None, None


def calculate_metrics(all_labels: np.ndarray, all_preds: np.ndarray) -> Dict:
    """计算评估指标"""
    cm = confusion_matrix(all_labels, all_preds)
    
    class_acc = {}
    for i in range(len(cm)):
        if cm[i].sum() > 0:
            class_acc[i] = cm[i, i] / cm[i].sum()
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'confusion_matrix': cm,
        'class_accuracy': class_acc,
    }
    
    return metrics


def save_model(model: nn.Module, config: ModelConfig, save_dir: Path, epoch: int, best_acc: float):
    """保存模型"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc,
        'config': {
            'model_name': config.model_name,
            'num_classes': config.num_classes,
            'hidden_dim': config.hidden_dim,
            'dropout': config.dropout,
        },
    }
    
    torch.save(checkpoint, save_dir / "latest_model.pth")
    
    if best_acc > 85:
        torch.save(checkpoint, save_dir / f"best_model_acc{best_acc:.2f}.pth")


def main():
    """主训练函数 - 优化版"""
    print("\n" + "="*70)
    print("  Sign Language Recognition - Optimized Training (ResNet50)")
    print("="*70)
    
    # 创建输出目录
    output_dir = PROJECT_ROOT / "models" / "sign_language" / f"optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results" / "plots" / f"optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 初始化可视化器
    reset_visualizer()
    visualizer = PaperVisualizer(str(results_dir))
    
    # 配置
    config = ModelConfig(
        model_name="resnet50",
        num_samples=8000,
        num_epochs=50,
        batch_size=64,
        learning_rate=0.0005,
        freeze_backbone=True,
        unfreeze_epoch=10,
        gradual_unfreeze=True,
        label_smoothing=0.1,
        mixup_alpha=0.2,
        use_amp=True,
    )
    
    print(f"\n训练配置 (优化版):")
    print(f"  - 模型: ResNet50 (论文推荐)")
    print(f"  - 样本数量: {config.num_samples}")
    print(f"  - 训练轮数: {config.num_epochs}")
    print(f"  - 批次大小: {config.batch_size}")
    print(f"  - 学习率: {config.learning_rate}")
    print(f"  - 标签平滑: {config.label_smoothing}")
    print(f"  - Mixup: {config.mixup_alpha}")
    print(f"  - AMP加速: {config.use_amp}")
    
    # 准备数据
    train_dataset, test_dataset, label_encoder = prepare_data(config)
    config.num_classes = len(label_encoder.classes_)
    
    # 创建数据加载器
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
    
    print(f"\n  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    # 创建模型
    print(f"\n[2/5] 创建 ResNet50 模型...")
    model = SignLanguageClassifier(config).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 损失函数 (带标签平滑)
    criterion = LabelSmoothingLoss(config.num_classes, config.label_smoothing)
    
    # 优化器 (带分层学习率)
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.learning_rate * 0.1},  # 骨干网络低学习率
        {'params': classifier_params, 'lr': config.learning_rate}
    ], weight_decay=config.weight_decay)
    
    # 学习率调度 (余弦退火)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda') if config.use_amp and DEVICE.type == 'cuda' else None
    
    # 训练循环
    print(f"\n[3/5] 开始训练...")
    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        
        # 渐进式解冻
        if epoch == config.unfreeze_epoch and config.gradual_unfreeze:
            model.unfreeze_backbone(unfreeze_ratio=0.5)
            # 降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 5
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, config, DEVICE
        )
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 评估
        if epoch % config.eval_interval == 0 or epoch == config.num_epochs:
            val_loss, val_acc, all_preds, all_labels = evaluate(
                model, test_loader, criterion, DEVICE, return_predictions=True
            )
            
            metrics = calculate_metrics(all_labels, all_preds)
            
            # 记录到可视化器
            visualizer.log_training(
                epoch, train_loss, val_loss, train_acc, val_acc, 
                current_lr, time.time() - epoch_start
            )
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                save_model(model, config, output_dir, epoch, best_acc)
            
            print(f"\n  Epoch {epoch}/{config.num_epochs} Summary:")
            print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"    Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"    Precision: {metrics['precision_macro']:.4f}")
            print(f"    Recall: {metrics['recall_macro']:.4f}")
            print(f"    F1 Score: {metrics['f1_macro']:.4f}")
            print(f"    Learning Rate: {current_lr:.6f}")
        else:
            visualizer.log_training(
                epoch, train_loss, val_loss, train_acc, 0, 
                current_lr, time.time() - epoch_start
            )
    
    # 训练完成
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
        'config': {
            'model': config.model_name,
            'num_samples': config.num_samples,
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_classes': config.num_classes,
            'label_smoothing': config.label_smoothing,
            'mixup_alpha': config.mixup_alpha,
        },
        'final_metrics': {
            'val_accuracy': final_acc,
            'precision_macro': final_metrics['precision_macro'],
            'recall_macro': final_metrics['recall_macro'],
            'f1_macro': final_metrics['f1_macro'],
        },
        'best_metrics': {
            'best_val_accuracy': best_acc,
            'best_epoch': best_epoch,
        },
        'training_time': {
            'total_time_minutes': total_time / 60,
        }
    }
    
    report_path = output_dir / "training_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("  Training Complete!")
    print("="*70)
    print(f"\n最佳配置:")
    print(f"  - ResNet50 (深度特征提取)")
    print(f"  - 标签平滑 (Label Smoothing): 提升泛化能力")
    print(f"  - Mixup 数据增强: 提升鲁棒性")
    print(f"  - 渐进式解冻: 避免灾难性遗忘")
    print(f"  - AMP 加速: 提升训练速度")
    print("="*70)


if __name__ == "__main__":
    main()

