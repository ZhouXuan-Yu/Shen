"""
CE-CSL 手语翻译数据集

参考 AAA/执行.md 文档实现

数据集结构:
- label/train.csv: 训练集标注 (Number, Translator, Chinese Sentences, Gloss, Note)
- label/dev.csv: 验证集标注
- label/test.csv: 测试集标注
- video/train/A/*.mp4: 训练集视频 (A-L 12个翻译者)
- video/dev/A/*.mp4: 验证集视频
- video/test/A/*.mp4: 测试集视频
"""

import os
import sys
import json
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import Counter
import cv2
from torchvision import transforms


# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import cfg
except ImportError:
    # 如果导入失败，使用默认值
    sys.path.insert(0, str(Path(__file__).parent))
    from config import cfg


# ==================== 词汇表管理 ====================
class Vocabulary:
    """手语词汇表管理类"""
    
    def __init__(
        self,
        min_count: int = 2,
        pad_token: str = '<pad>',
        sos_token: str = '<sos>',
        eos_token: str = '<eos>',
        unk_token: str = '<unk>',
        blank_token: str = '<blank>'
    ):
        self.min_count = min_count
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.blank_token = blank_token
        
        # 特殊 token
        self.word2idx = {
            pad_token: 0,
            sos_token: 1,
            eos_token: 2,
            unk_token: 3,
            blank_token: 4,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_count = Counter()
        self.n_words = 5  # 特殊token数量
    
    def build_vocab(self, gloss_list: List[str]):
        """从词汇列表构建词汇表"""
        # 统计词频
        self.word_count.update(gloss_list)
        
        # 筛选高频词
        for word, count in self.word_count.most_common():
            if count >= self.min_count:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1
        
        print(f"Vocabulary built: {self.n_words} words (min_count={self.min_count})")
        return self
    
    def add_words(self, words: List[str]):
        """添加新词汇"""
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1
    
    def encode(self, word: str) -> int:
        """编码单个词"""
        return self.word2idx.get(word, self.word2idx[self.unk_token])
    
    def decode(self, idx: int) -> str:
        """解码单个索引"""
        return self.idx2word.get(idx, self.unk_token)
    
    def encode_sequence(self, words: List[str], add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """编码词序列"""
        encoded = [self.encode(w) for w in words]
        
        if add_sos:
            encoded = [self.word2idx[self.sos_token]] + encoded
        if add_eos:
            encoded = encoded + [self.word2idx[self.eos_token]]
        
        return encoded
    
    def decode_sequence(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """解码索引序列"""
        decoded = []
        for idx in indices:
            word = self.decode(idx)
            if skip_special:
                if word in [self.pad_token, self.sos_token, self.eos_token, self.blank_token]:
                    continue
            decoded.append(word)
        return decoded
    
    def save(self, save_path: str):
        """保存词汇表"""
        save_data = {
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()},
            'word_count': dict(self.word_count),
            'min_count': self.min_count,
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to: {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'Vocabulary':
        """加载词汇表"""
        with open(load_path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        vocab = cls(min_count=save_data.get('min_count', 2))
        vocab.word2idx = save_data['word2idx']
        vocab.idx2word = {int(k): v for k, v in save_data['idx2word'].items()}
        vocab.word_count = Counter(save_data.get('word_count', {}))
        vocab.n_words = len(vocab.word2idx)
        
        print(f"Vocabulary loaded: {vocab.n_words} words")
        return vocab
    
    def __len__(self):
        return self.n_words
    
    def __contains__(self, word):
        return word in self.word2idx


# ==================== 图像变换 ====================
def get_train_transform(img_size: int = 224) -> transforms.Compose:
    """获取训练集图像变换"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.1),
    ])


def get_test_transform(img_size: int = 224) -> transforms.Compose:
    """获取测试集图像变换"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ==================== 视频处理 ====================
def sample_frames(video_path: str, num_frames: int, transform: Optional[transforms.Compose] = None):
    """
    从视频中均匀采样帧
    """
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # 均匀采样帧索引
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # 转换为张量
    frames = np.array(frames, dtype=np.float32)
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2)  # (C, T, H, W)
    
    # 应用变换
    if transform is not None:
        # 对每帧应用变换
        C, T, H, W = frames.shape
        frames = frames.permute(1, 0, 2, 3)  # (T, C, H, W)
        transformed_frames = []
        for i in range(T):
            frame = transform(frames[i])
            transformed_frames.append(frame)
        frames = torch.stack(transformed_frames).permute(1, 0, 2, 3)  # (C, T, H, W)
    
    return frames


# ==================== CSL 数据集类 ====================
class CSLDataset(Dataset):
    """
    CE-CSL 手语翻译数据集
    
    支持:
    - 训练集、验证集、测试集
    - 视频帧采样
    - 实时数据增强
    - 词汇表构建
    """
    
    def __init__(
        self,
        csv_path: str,
        video_root: str,
        vocab: Optional[Vocabulary] = None,
        num_frames: int = 32,
        img_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = False,
        gloss_tokenizer: Optional[Callable] = None,
        chinese_tokenizer: Optional[Callable] = None,
        max_gloss_len: int = 50,
        max_chinese_len: int = 100,
    ):
        self.csv_path = csv_path
        self.video_root = video_root
        self.vocab = vocab
        self.num_frames = num_frames
        self.img_size = img_size
        self.transform = transform or (get_train_transform(img_size) if is_train else get_test_transform(img_size))
        self.is_train = is_train
        self.max_gloss_len = max_gloss_len
        self.max_chinese_len = max_chinese_len
        
        # 加载标注数据
        self.df = pd.read_csv(csv_path)
        
        # 过滤有效样本
        self.samples = self._filter_valid_samples()
        
        print(f"Dataset loaded: {len(self.samples)} samples from {csv_path}")
    
    def _filter_valid_samples(self) -> List[Dict]:
        """过滤有效的视频样本"""
        valid_samples = []
        
        for idx, row in self.df.iterrows():
            number = row['Number']
            translator = row['Translator']
            
            # 构建视频路径
            video_name = f"{number}.mp4"
            
            # 尝试多个可能的路径
            possible_paths = [
                os.path.join(self.video_root, translator, video_name),
                os.path.join(self.video_root, video_name),
            ]
            
            # 查找视频文件
            video_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    video_path = path
                    break
            
            if video_path is not None:
                valid_samples.append({
                    'number': number,
                    'translator': translator,
                    'video_path': video_path,
                    'chinese': row['Chinese Sentences'],
                    'gloss': row['Gloss'],
                    'note': row.get('Note', ''),
                })
        
        return valid_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 提取视频帧
        frames = sample_frames(
            sample['video_path'],
            self.num_frames,
            self.transform
        )
        
        # 如果视频无效，返回全零张量
        if frames is None:
            frames = torch.zeros(3, self.num_frames, self.img_size, self.img_size)
        
        # 处理 gloss 标注
        gloss_words = str(sample['gloss']).split('/')
        gloss_words = [w.strip() for w in gloss_words if w.strip()]
        
        # 编码 gloss
        if self.vocab is not None:
            gloss_ids = self.vocab.encode_sequence(
                gloss_words,
                add_sos=True,
                add_eos=True
            )
            gloss_ids = gloss_ids[:self.max_gloss_len]  # 截断
            gloss_ids = torch.tensor(gloss_ids, dtype=torch.long)
        else:
            gloss_ids = torch.tensor([0], dtype=torch.long)
        
        # 处理中文标注
        chinese = sample['chinese']
        chinese_ids = None
        
        return {
            'frames': frames,
            'gloss_ids': gloss_ids,
            'gloss_words': gloss_words,
            'chinese': chinese,
            'number': sample['number'],
            'translator': sample['translator'],
            'video_path': sample['video_path'],
        }


# ==================== 批处理函数 ====================
def ctc_collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    CTC 任务的批处理函数
    
    Args:
        batch: 样本列表
        
    Returns:
        frames: 填充后的帧张量 (B, C, T, H, W)
        gloss_ids: 合并的 gloss ID 序列
        gloss_lengths: 每个样本的 gloss 长度
        video_lengths: 每个样本的实际帧数
    """
    # 过滤无效帧
    valid_batch = [b for b in batch if b['frames'] is not None]
    if len(valid_batch) == 0:
        batch_size = len(batch)
        frames = torch.zeros(batch_size, 3, 32, 224, 224)
        gloss_ids = torch.zeros(batch_size, 1, dtype=torch.long)
        gloss_lengths = torch.ones(batch_size, dtype=torch.long)
        return frames, gloss_ids, gloss_lengths, [32] * batch_size
    
    batch = valid_batch
    
    # 按帧数排序
    batch.sort(key=lambda x: x['frames'].size(1), reverse=True)
    
    frames_list = []
    gloss_ids_list = []
    gloss_lengths = []
    video_lengths = []
    
    for sample in batch:
        frames = sample['frames']
        frames_list.append(frames)
        gloss_ids_list.append(sample['gloss_ids'])
        gloss_lengths.append(len(sample['gloss_words']))
        video_lengths.append(frames.size(1))
    
    # 填充帧序列
    max_t = max(v.size(1) for v in frames_list)
    padded_frames = torch.zeros(len(batch), 3, max_t, 224, 224)
    
    for i, frames in enumerate(frames_list):
        padded_frames[i, :, :frames.size(1), :, :] = frames
    
    # 合并 gloss IDs
    gloss_ids = torch.cat(gloss_ids_list)
    gloss_lengths = torch.tensor(gloss_lengths, dtype=torch.long)
    video_lengths = torch.tensor(video_lengths, dtype=torch.long)
    
    return padded_frames, gloss_ids, gloss_lengths, video_lengths


def translation_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    翻译任务的批处理函数
    
    Returns:
        frames: 填充后的帧张量 (B, C, T, H, W)
        gloss_ids: 填充后的 gloss ID 序列 (B, L)
        chinese_ids: 填充后的中文 ID 序列 (B, L)
        gloss_mask: gloss 注意力掩码 (B, L)
        chinese_mask: 中文注意力掩码 (B, L)
    """
    # 过滤无效帧
    valid_batch = [b for b in batch if b['frames'] is not None]
    if len(valid_batch) == 0:
        batch_size = len(batch)
        return {
            'frames': torch.zeros(batch_size, 3, 32, 224, 224),
            'gloss_ids': torch.zeros(batch_size, 1, dtype=torch.long),
            'chinese_ids': torch.zeros(batch_size, 1, dtype=torch.long),
            'gloss_mask': torch.ones(batch_size, 1, dtype=torch.bool),
            'chinese_mask': torch.ones(batch_size, 1, dtype=torch.bool),
        }
    
    batch = valid_batch
    batch_size = len(batch)
    
    # 找到最大长度
    max_gloss_len = max(len(b['gloss_ids']) for b in batch)
    max_chinese_len = max(len(b['chinese']) for b in batch)
    
    frames_list = []
    gloss_ids_list = []
    chinese_ids_list = []
    
    for sample in batch:
        frames_list.append(sample['frames'])
        
        # 填充 gloss IDs
        gloss = sample['gloss_ids']
        padded_gloss = torch.zeros(max_gloss_len, dtype=torch.long)
        padded_gloss[:len(gloss)] = gloss
        gloss_ids_list.append(padded_gloss)
        
        # 简单处理中文（实际应用中应该使用 tokenizer）
        chinese = sample['chinese'][:max_chinese_len]
        chinese_ids_list.append(torch.tensor([ord(c) for c in chinese], dtype=torch.long))
    
    # 堆叠张量
    frames = torch.stack(frames_list)
    gloss_ids = torch.stack(gloss_ids_list)
    chinese_ids = torch.stack(chinese_ids_list)
    
    # 创建掩码
    gloss_mask = (gloss_ids != 0)
    chinese_mask = (chinese_ids != 0)
    
    return {
        'frames': frames,
        'gloss_ids': gloss_ids,
        'chinese_ids': chinese_ids,
        'gloss_mask': gloss_mask,
        'chinese_mask': chinese_mask,
    }


# ==================== 数据加载器工厂 ====================
def create_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_frames: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    train_ratio: float = 0.9,
    vocab: Optional[Vocabulary] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """
    创建训练、验证、测试数据加载器
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_frames: 每视频采样帧数
        img_size: 图像尺寸
        num_workers: 数据加载线程数
        train_ratio: 训练集划分比例
        vocab: 预构建的词汇表
        
    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    label_dir = os.path.join(data_root, 'label')
    video_dir = os.path.join(data_root, 'video')
    
    # 加载训练集标注以构建词汇表
    train_csv = os.path.join(label_dir, 'train.csv')
    train_df = pd.read_csv(train_csv)
    
    # 构建词汇表
    if vocab is None:
        vocab = Vocabulary(min_count=2)
        all_glosses = []
        for gloss in train_df['Gloss']:
            words = str(gloss).split('/')
            words = [w.strip() for w in words if w.strip()]
            all_glosses.extend(words)
        vocab.build_vocab(all_glosses)
        
        # 保存词汇表
        vocab.save(os.path.join(data_root, 'vocab.json'))
    
    # 加载数据集
    train_video_dir = os.path.join(video_dir, 'train')
    val_video_dir = os.path.join(video_dir, 'dev')
    test_video_dir = os.path.join(video_dir, 'test')
    
    # 创建数据集
    train_dataset = CSLDataset(
        csv_path=train_csv,
        video_root=train_video_dir,
        vocab=vocab,
        num_frames=num_frames,
        img_size=img_size,
        is_train=True,
    )
    
    val_dataset = CSLDataset(
        csv_path=os.path.join(label_dir, 'dev.csv'),
        video_root=val_video_dir,
        vocab=vocab,
        num_frames=num_frames,
        img_size=img_size,
        is_train=False,
    )
    
    test_dataset = CSLDataset(
        csv_path=os.path.join(label_dir, 'test.csv'),
        video_root=test_video_dir,
        vocab=vocab,
        num_frames=num_frames,
        img_size=img_size,
        is_train=False,
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Vocabulary: {len(vocab)} words")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, vocab


# ==================== 测试代码 ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CSL Dataset')
    parser.add_argument('--data-root', type=str, default='D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Testing CSL Dataset")
    print("=" * 70)
    
    # 创建数据集
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
    )
    
    # 测试加载器
    print("\nTesting train_loader:")
    for i, batch in enumerate(train_loader):
        if i >= args.num_samples:
            break
        frames, gloss_ids, gloss_lengths, video_lengths = batch
        print(f"  Batch {i}: frames={frames.shape}, gloss_ids={gloss_ids.shape}, "
              f"gloss_lengths={gloss_lengths.tolist()}, video_lengths={video_lengths.tolist()}")
    
    # 测试数据集
    print("\nTesting single sample:")
    sample = val_loader.dataset[0]
    print(f"  Number: {sample['number']}")
    print(f"  Translator: {sample['translator']}")
    print(f"  Gloss: {sample['gloss_words']}")
    print(f"  Chinese: {sample['chinese']}")
    print(f"  Frames: {sample['frames'].shape}")
    
    print("\n" + "=" * 70)
    print("Dataset test passed!")
    print("=" * 70)

