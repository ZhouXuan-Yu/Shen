"""
快速训练测试脚本
只运行 1 个 epoch 来测试整个流程
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

sys.path.insert(0, '.')

from dataset import create_dataloaders
from models.translator import SignLanguageTranslator, count_parameters

# 配置
DATA_ROOT = 'D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL'
OUTPUT_DIR = './output'
NUM_FRAMES = 16  # 减小帧数以加快训练
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载数据
print("\n[1/5] Loading dataset...")
train_loader, val_loader, test_loader, vocab = create_dataloaders(
    data_root=DATA_ROOT,
    batch_size=BATCH_SIZE,
    num_frames=NUM_FRAMES,
    num_workers=0,
)

# 创建模型
print("\n[2/5] Creating model...")
model = SignLanguageTranslator(
    num_gloss_classes=len(vocab),
    num_chinese_classes=len(vocab),
    resnet_model='resnet50',
    resnet_pretrained=True,
    resnet_output_dim=512,
    resnet_freeze_layers=10,  # 冻结 ResNet 前 10 层
    lstm_hidden_size=512,
    lstm_num_layers=2,
    lstm_dropout=0.3,
    use_ctc=True,
    use_transformer=True,
).to(device)

total, trainable = count_parameters(model)
print(f"Model Parameters: {total:,} total, {trainable:,} trainable")

# 优化器
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# 训练函数
def train_epoch(epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    for batch_idx, (frames, gloss_ids, gloss_lengths, video_lengths) in enumerate(pbar):
        # 移动到设备
        frames = frames.to(device)
        gloss_ids = gloss_ids.to(device)
        gloss_lengths = gloss_lengths.to(device)
        video_lengths = video_lengths.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        results = model(
            video_frames=frames,
            video_lengths=video_lengths,
            gloss_targets=gloss_ids,
            gloss_target_lengths=gloss_lengths,
            mode='ctc',  # 只使用 CTC 损失
        )
        
        loss = results['losses']['total_loss']
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 只跑 10 个 batch 用于测试
        if batch_idx >= 10:
            break
    
    return total_loss / num_batches

# 验证函数
def validate():
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (frames, gloss_ids, gloss_lengths, video_lengths) in enumerate(tqdm(val_loader, desc="Validating")):
            frames = frames.to(device)
            gloss_ids = gloss_ids.to(device)
            gloss_lengths = gloss_lengths.to(device)
            video_lengths = video_lengths.to(device)
            
            results = model(
                video_frames=frames,
                video_lengths=video_lengths,
                gloss_targets=gloss_ids,
                gloss_target_lengths=gloss_lengths,
                mode='ctc',
            )
            
            loss = results['losses']['total_loss']
            total_loss += loss.item()
            num_batches += 1
            
            # 只跑 3 个 batch
            if batch_idx >= 3:
                break
    
    return total_loss / num_batches

# 训练循环
print("\n[3/5] Starting training...")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"{'='*50}")
    
    # 训练
    train_loss = train_epoch(epoch)
    
    # 验证
    val_loss = validate()
    
    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    
    # 保存检查点
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'test_checkpoint.pth'))
    print(f"\nCheckpoint saved to {OUTPUT_DIR}/test_checkpoint.pth")

total_time = time.time() - start_time
print(f"\n{'='*50}")
print(f"Training completed in {total_time:.2f} seconds!")
print(f"{'='*50}")
print(f"\nOutput directory: {OUTPUT_DIR}")




