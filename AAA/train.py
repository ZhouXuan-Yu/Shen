"""
手语翻译模型训练脚本

参考 AAA/执行.md 文档实现:
- ResNet-50: 空间特征提取
- Bi-LSTM: 时序建模
- CTC Loss: 序列对齐
- Transformer: 端到端翻译

用法:
  python train.py --config config.yaml
  python train.py --data-root /path/to/dataset --epochs 50 --batch-size 8
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg
from dataset import CSLDataset, Vocabulary, ctc_collate_fn, create_dataloaders
from models.translator import SignLanguageTranslator, count_parameters


# ==================== 日志配置 ====================
def setup_logging(log_dir: str):
    """配置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler(),
        ]
    )
    
    return logging.getLogger(__name__)


# ==================== 评估指标 ====================
def calculate_bleu(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
    """
    计算 BLEU 分数
    
    Args:
        reference: 参考句子 (词列表)
        hypothesis: 预测句子 (词列表)
        max_n: 最大 n-gram
        
    Returns:
        bleu: BLEU 分数 (0-100)
    """
    from collections import Counter
    
    # 截断到相同长度
    min_len = min(len(reference), len(hypothesis))
    
    if min_len == 0:
        return 0.0
    
    # 计算各 n-gram 的 precision
    precisions = []
    
    for n in range(1, min(max_n + 1, min_len + 1)):
        # 生成 n-gram
        ref_ngrams = Counter([' '.join(reference[i:i+n]) for i in range(len(reference)-n+1)])
        hyp_ngrams = Counter([' '.join(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)])
        
        # 匹配数
        matches = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())
        
        if total > 0:
            precisions.append(matches / total)
        else:
            precisions.append(0)
    
    if not precisions or precisions[0] == 0:
        return 0.0
    
    # 几何平均
    log_precision = sum(np.log(p + 1e-10) for p in precisions) / len(precisions)
    bleu = np.exp(log_precision)
    
    # 长度惩罚
    bp = min(1.0, np.exp(1 - len(reference) / (len(hypothesis) + 1e-10)))
    
    return 100 * bleu * bp


# ==================== 训练器 ====================
class Trainer:
    """手语翻译模型训练器"""
    
    def __init__(
        self,
        model: SignLanguageTranslator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: torch.device,
        config: cfg,
        logger: logging.Logger,
        output_dir: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger
        self.output_dir = output_dir
        
        # 混合精度
        self.scaler = GradScaler() if config.training.USE_AMP else None
        
        # 最佳指标
        self.best_bleu = 0.0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # 词汇表
        self.vocab = None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_ctc_loss = 0.0
        total_trans_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            # 获取数据
            frames = batch[0].to(self.device)
            gloss_ids = batch[1].to(self.device)
            gloss_lengths = batch[2].to(self.device)
            video_lengths = batch[3].to(self.device)
            
            # 准备目标
            gloss_targets = gloss_ids
            gloss_target_lengths = gloss_lengths
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.config.training.USE_AMP:
                with autocast():
                    results = self.model(
                        video_frames=frames,
                        video_lengths=video_lengths,
                        gloss_targets=gloss_targets,
                        gloss_target_lengths=gloss_target_lengths,
                        mode='both',
                    )
                    
                    loss = results['losses']['total_loss']
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config.training.GRAD_CLIP_NORM > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.GRAD_CLIP_NORM
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                results = self.model(
                    video_frames=frames,
                    video_lengths=video_lengths,
                    gloss_targets=gloss_targets,
                    gloss_target_lengths=gloss_target_lengths,
                    mode='both',
                )
                
                loss = results['losses']['total_loss']
                loss.backward()
                
                # 梯度裁剪
                if self.config.training.GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.GRAD_CLIP_NORM
                    )
                
                self.optimizer.step()
            
            # 统计
            total_loss += results['losses']['total_loss'].item()
            if 'ctc_loss' in results['losses']:
                total_ctc_loss += results['losses']['ctc_loss'].item()
            if 'translation_loss' in results['losses']:
                total_trans_loss += results['losses']['translation_loss'].item()
            
            num_batches += 1
            pbar.set_postfix({
                'loss': f"{results['losses']['total_loss'].item():.4f}",
                'ctc': f"{results['losses'].get('ctc_loss', 0):.4f}",
                'trans': f"{results['losses'].get('translation_loss', 0):.4f}",
            })
        
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {
            'train_loss': total_loss / num_batches,
            'train_ctc_loss': total_ctc_loss / num_batches,
            'train_trans_loss': total_trans_loss / num_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        total_ctc_loss = 0.0
        total_trans_loss = 0.0
        num_batches = 0
        
        all_references = []
        all_hypotheses = []
        
        pbar = tqdm(self.val_loader, desc="[Validate]")
        
        for batch in pbar:
            # 获取数据
            frames = batch[0].to(self.device)
            gloss_ids = batch[1].to(self.device)
            gloss_lengths = batch[2].to(self.device)
            video_lengths = batch[3].to(self.device)
            
            # 准备目标
            gloss_targets = gloss_ids
            gloss_target_lengths = gloss_lengths
            
            # 前向传播
            results = self.model(
                video_frames=frames,
                video_lengths=video_lengths,
                gloss_targets=gloss_targets,
                gloss_target_lengths=gloss_target_lengths,
                mode='both',
            )
            
            # 统计损失
            total_loss += results['losses']['total_loss'].item()
            if 'ctc_loss' in results['losses']:
                total_ctc_loss += results['losses']['ctc_loss'].item()
            if 'translation_loss' in results['losses']:
                total_trans_loss += results['losses']['translation_loss'].item()
            
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_ctc_loss': total_ctc_loss / num_batches,
            'val_trans_loss': total_trans_loss / num_batches,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_bleu': self.best_bleu,
            'best_loss': self.best_loss,
        }
        
        # 保存最新检查点
        save_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, save_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.output_dir, 'model_best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
        
        # 定期保存
        if (epoch + 1) % self.config.training.SAVE_INTERVAL == 0:
            epoch_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, epoch_path)
        
        self.logger.info(f"Saved checkpoint to {save_path}")
    
    def train(self, num_epochs: int, start_epoch: int = 0):
        """完整训练"""
        self.logger.info("=" * 70)
        self.logger.info("Starting Training")
        self.logger.info("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            self.logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate()
            
            # 日志
            self.logger.info(
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )
            
            # 检查是否为最佳
            is_best = val_metrics['val_loss'] < self.best_loss
            
            if val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 早停
            if self.patience_counter >= self.config.training.EARLY_STOPPING_PATIENCE:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"\nTraining completed in {total_time / 60:.2f} minutes")
        self.logger.info(f"Best validation loss: {self.best_loss:.4f}")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='Sign Language Translation Training')
    
    # 数据参数
    parser.add_argument('--data-root', type=str, default='D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL')
    parser.add_argument('--num-frames', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip-norm', type=float, default=1.0)
    parser.add_argument('--use-amp', action='store_true', default=True)
    
    # 模型参数
    parser.add_argument('--resnet-model', type=str, default='resnet50')
    parser.add_argument('--lstm-hidden-size', type=int, default=512)
    parser.add_argument('--lstm-num-layers', type=int, default=2)
    parser.add_argument('--vocab-size', type=int, default=1000)
    
    # 其他
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 输出目录
    output_dir = os.path.join(args.output_dir, f"exp_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 日志
    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("Sign Language Translation Training")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # 创建数据加载器
    logger.info("\nLoading datasets...")
    
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        img_size=args.img_size,
        num_workers=args.workers,
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # 创建模型
    logger.info("\nCreating model...")
    
    model = SignLanguageTranslator(
        num_gloss_classes=len(vocab),
        num_chinese_classes=len(vocab),
        resnet_model=args.resnet_model,
        resnet_pretrained=True,
        resnet_output_dim=512,
        resnet_freeze_layers=10,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        lstm_dropout=0.3,
        use_ctc=True,
        use_transformer=True,
    ).to(device)
    
    # 统计参数
    total, trainable = count_parameters(model)
    logger.info(f"Model parameters: {total:,} total, {trainable:,} trainable")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2,
        eta_min=1e-6,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=cfg,
        logger=logger,
        output_dir=output_dir,
    )
    
    # 保存配置
    config_path = os.path.join(output_dir, 'config.json')
    cfg.save(config_path)
    
    # 开始训练
    trainer.train(num_epochs=args.epochs)
    
    logger.info("\n" + "=" * 70)
    logger.info("Training completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()




