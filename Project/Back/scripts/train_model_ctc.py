"""
基于CTC的连续手语识别模型训练
参考 AAA/执行.md 文档实现

架构:
1. ResNet-50 - 空间特征提取
2. Bi-LSTM - 时序建模
3. CTC Loss - 序列对齐
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# 添加scripts目录到路径
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==================== 配置 ====================
class Config:
    # 数据路径
    DATA_ROOT = 'D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL'
    VIDEO_ROOT = os.path.join(DATA_ROOT, 'video')
    LABEL_PATH = os.path.join(DATA_ROOT, 'label')
    
    # 采样配置
    NUM_FRAMES = 32  # 每视频采样帧数
    IMG_SIZE = 224
    
    # 模型配置
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # 训练配置
    BATCH_SIZE = 8
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    MAX_GRAD_NORM = 1.0
    
    # 其他
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 输出路径
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models', 'sign_language_ctc')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

cfg = Config()

# ==================== 数据预处理 ====================
def build_vocab(label_path, min_count=5):
    """构建手势词词汇表"""
    df = pd.read_csv(os.path.join(label_path, 'train.csv'))
    
    # 统计所有手势词
    gloss_counter = Counter()
    for gloss in df['Gloss']:
        words = str(gloss).split('/')
        for w in words:
            w = w.strip()
            if w:
                gloss_counter[w] += 1
    
    print(f'Total unique glosses: {len(gloss_counter)}')
    
    # 筛选高频词
    vocab = {word: idx + 2 for idx, (word, count) in 
             enumerate(gloss_counter.most_common()) if count >= min_count}
    vocab['<blank>'] = 0  # CTC blank token
    vocab['<unk>'] = 1    # unknown
    
    print(f'Vocab size (min_count={min_count}): {len(vocab)}')
    
    return vocab

def encode_gloss(gloss_str, vocab):
    """将手势词序列编码为整数列表"""
    words = str(gloss_str).split('/')
    return [vocab.get(w.strip(), vocab['<unk>']) for w in words if w.strip()]

# ==================== 数据集类 ====================
class CSLVideoDataset(Dataset):
    """连续手语视频数据集"""
    
    def __init__(self, video_paths, labels, vocab, num_frames=32, cache_frames=False):
        self.video_paths = video_paths
        self.labels = labels
        self.vocab = vocab
        self.num_frames = num_frames
        self.cache_frames = cache_frames
        self.frame_cache = {}
        
    def __len__(self):
        return len(self.video_paths)
    
    def _extract_frames(self, video_path):
        """从视频中均匀采样帧"""
        if not os.path.exists(video_path):
            return None
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        # 均匀采样
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (cfg.IMG_SIZE, cfg.IMG_SIZE))
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # 转换为tensor: (T, H, W, C) -> (C, T, H, W)
        frames = np.array(frames, dtype=np.float32)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # 提取帧
        frames = self._extract_frames(video_path)
        
        if frames is None:
            # 返回全零张量
            frames = torch.zeros(3, self.num_frames, cfg.IMG_SIZE, cfg.IMG_SIZE)
        
        return frames, label

def collate_fn_ctc(batch):
    """CTC任务的批处理函数"""
    # 按帧数排序（从长到短）
    batch.sort(key=lambda x: x[0].size(1), reverse=True)
    
    frames_list = []
    labels_list = []
    label_lengths = []
    
    for frames, label in batch:
        frames_list.append(frames)
        labels_list.append(label)
        label_lengths.append(len(label))
    
    # 填充帧序列
    max_t = max(f.size(1) for f in frames_list)
    padded_frames = torch.zeros(len(batch), 3, max_t, cfg.IMG_SIZE, cfg.IMG_SIZE)
    
    for i, frames in enumerate(frames_list):
        padded_frames[i, :, :frames.size(1), :, :] = frames
    
    # 合并标签
    labels_tensor = torch.tensor(sum(labels_list, []), dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    
    return padded_frames, labels_tensor, label_lengths

# ==================== 模型定义 ====================
class ResNetFeatureExtractor(nn.Module):
    """ResNet特征提取器（预训练）"""
    
    def __init__(self, output_dim=512):
        super().__init__()
        from torchvision import models
        # 加载预训练的ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 移除最后的FC层
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, output_dim)
        
        # 冻结部分层（可选）
        self._freeze_layers(10)  # 冻结前10层
        
    def _freeze_layers(self, num_layers):
        """冻结前num_layers层"""
        layers = list(self.backbone.children())
        for i, layer in enumerate(layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # 合并时间和批次维度
        x = x.view(B * C, T, H, W)
        
        # 提取特征
        features = self.backbone(x)  # (B*T, 2048, 7, 7)
        features = self.pool(features)  # (B*T, 2048, 1, 1)
        features = features.view(B, T, -1)  # (B, T, 2048)
        
        # 降维
        features = self.fc(features)  # (B, T, output_dim)
        
        return features


class BidirectionalLSTM(nn.Module):
    """双向LSTM层"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        # 拼接双向hidden state
        h_n = torch.cat([h_n[0], h_n[1]], dim=1)  # (B, 2*hidden)
        return output, h_n


class CTCSignLanguageModel(nn.Module):
    """CTC连续手语识别模型"""
    
    def __init__(self, num_classes, hidden_size=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        
        # 空间特征提取
        self.resnet = ResNetFeatureExtractor(output_dim=hidden_size)
        
        # 时序建模（双层Bi-LSTM）
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,  # 双向，所以减半
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 分类头
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths=None):
        """
        x: (B, C, T, H, W)
        lengths: 实际帧数列表（用于padding mask）
        """
        # 特征提取
        features = self.resnet(x)  # (B, T, hidden)
        
        # Bi-LSTM
        lstm_out, hidden = self.lstm(features)  # (B, T, hidden)
        
        # 分类
        logits = self.classifier(lstm_out)  # (B, T, num_classes)
        
        # 转换为 (T, B, num_classes) 用于CTC
        logits = logits.permute(1, 0, 2)
        
        return logits


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (frames, labels, label_lengths) in enumerate(pbar):
        frames = frames.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        # 计算输入序列长度（每帧一个时间步）
        input_lengths = torch.full(
            size=(frames.size(0),),
            fill_value=frames.size(2),
            dtype=torch.long
        )
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(frames)  # (T, B, num_classes)
        
        # CTC Loss
        loss = criterion(logits, labels, input_lengths, label_lengths)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
        
        optimizer.step()
        
        total_loss += loss.item() * frames.size(0)
        total_samples += frames.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total_samples


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, labels, label_lengths in tqdm(dataloader, desc='Evaluating'):
            frames = frames.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            
            input_lengths = torch.full(
                size=(frames.size(0),),
                fill_value=frames.size(2),
                dtype=torch.long
            )
            
            logits = model(frames)
            loss = criterion(logits, labels, input_lengths, label_lengths)
            
            total_loss += loss.item() * frames.size(0)
            total_samples += frames.size(0)
            
            # 简单的准确率计算（需要greedy decoding）
            pred_indices = logits.argmax(dim=-1)  # (T, B)
            
    return total_loss / total_samples


def greedy_decode(logits, blank=0):
    """贪心解码"""
    # logits: (T, B, num_classes)
    predictions = logits.argmax(dim=-1)  # (T, B)
    
    results = []
    for pred_seq in predictions.T:  # 遍历batch
        # 移除重复和blank
        decoded = []
        prev = -1
        for p in pred_seq:
            if p != prev and p != blank:
                decoded.append(p)
            prev = p
        results.append(decoded)
    
    return results


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='CTC Sign Language Recognition')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--min-count', type=int, default=5, help='Minimum gloss count for vocabulary')
    args = parser.parse_args()
    
    cfg.NUM_EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LEARNING_RATE = args.lr
    
    print('=' * 60)
    print('CTC-based Continuous Sign Language Recognition')
    print('=' * 60)
    print(f'Device: {cfg.DEVICE}')
    print(f'Epochs: {cfg.NUM_EPOCHS}')
    print(f'Batch size: {cfg.BATCH_SIZE}')
    print(f'Learning rate: {cfg.LEARNING_RATE}')
    print('=' * 60)
    
    # 1. 构建词汇表
    print('\n[1/5] Building vocabulary...')
    vocab = build_vocab(cfg.LABEL_PATH, min_count=args.min_count)
    
    # 保存词汇表
    vocab_path = os.path.join(cfg.OUTPUT_DIR, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f'Vocabulary saved to {vocab_path}')
    
    # 2. 加载数据
    print('\n[2/5] Loading dataset...')
    df = pd.read_csv(os.path.join(cfg.LABEL_PATH, 'train.csv'))
    
    video_paths = []
    labels = []
    
    for idx, row in df.iterrows():
        number = row['Number']
        translator = row['Translator']
        gloss = row['Gloss']
        
        # 构建视频路径: video/train/A/train-00001.mp4
        video_path = os.path.join(cfg.VIDEO_ROOT, 'train', translator, f'{number}.mp4')
        
        if os.path.exists(video_path):
            video_paths.append(video_path)
            labels.append(encode_gloss(gloss, vocab))
    
    print(f'Valid samples: {len(video_paths)} / {len(df)}')
    
    # 3. 划分数据集
    print('\n[3/5] Splitting dataset...')
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.1, random_state=42
    )
    
    print(f'Train: {len(train_paths)}, Val: {len(val_paths)}')
    
    # 4. 创建数据加载器
    print('\n[4/5] Creating dataloaders...')
    train_dataset = CSLVideoDataset(train_paths, train_labels, vocab, cfg.NUM_FRAMES)
    val_dataset = CSLVideoDataset(val_paths, val_labels, vocab, cfg.NUM_FRAMES)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn_ctc,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn_ctc,
        pin_memory=True
    )
    
    # 5. 创建模型
    print('\n[5/5] Creating model...')
    model = CTCSignLanguageModel(
        num_classes=len(vocab),
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # 6. 训练
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print('\n' + '=' * 60)
    print('Starting training...')
    print('=' * 60)
    
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE, epoch)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = evaluate(model, val_loader, criterion, cfg.DEVICE)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step()
        
        print(f'\nEpoch {epoch}/{cfg.NUM_EPOCHS}')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  LR: {scheduler.get_last_lr()[0]:.2e}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab': vocab
            }
            torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
            print(f'  -> Saved best model (val_loss: {val_loss:.4f})')
    
    # 7. 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'training_curve.png'), dpi=150)
    plt.close()
    
    print(f'\nTraining completed! Best val loss: {best_val_loss:.4f}')
    print(f'Model saved to: {cfg.OUTPUT_DIR}')


if __name__ == '__main__':
    main()

