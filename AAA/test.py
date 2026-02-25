"""
手语翻译模型测试和推理脚本

功能:
1. 模型评估 (BLEU, WER, CER)
2. 单个视频/图片推理
3. 批量测试
4. 结果保存

用法:
  python test.py --checkpoint model.pth --test-video video.mp4
  python test.py --checkpoint model.pth --test-dir /path/to/videos
  python test.py --checkpoint model.pth --eval --data-root /path/to/dataset
"""

import os
import sys
import json
import argparse
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg
from dataset import CSLDataset, Vocabulary, ctc_collate_fn, sample_frames, get_test_transform
from models.translator import SignLanguageTranslator, count_parameters
from models.ctc import greedy_decode_ctc, beam_search_decode_ctc


# ==================== 工具函数 ====================
def load_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best metric: {checkpoint.get('best_bleu', 'N/A')}")
    
    return checkpoint


def preprocess_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """预处理单张图片"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    
    transform = get_test_transform(img_size)
    img_tensor = transform(img)
    
    return img_tensor


def preprocess_video(video_path: str, num_frames: int = 32, img_size: int = 224) -> torch.Tensor:
    """预处理视频"""
    frames = sample_frames(video_path, num_frames)
    
    if frames is None:
        raise ValueError(f"Cannot extract frames from: {video_path}")
    
    return frames


# ==================== 推理类 ====================
class SignLanguageTranslatorInference:
    """手语翻译推理类"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device = None,
        vocab_path: str = None,
        use_beam_search: bool = False,
        beam_size: int = 5,
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            device: 推理设备
            vocab_path: 词汇表路径
            use_beam_search: 是否使用束搜索
            beam_size: 束搜索大小
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_beam_search = use_beam_search
        self.beam_size = beam_size
        
        # 加载词汇表
        if vocab_path and os.path.exists(vocab_path):
            self.vocab = Vocabulary.load(vocab_path)
        else:
            # 尝试从检查点加载
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'vocab' in checkpoint:
                self.vocab = checkpoint['vocab']
            else:
                self.vocab = None
        
        # 创建模型
        num_classes = len(self.vocab) if self.vocab else 1000
        
        self.model = SignLanguageTranslator(
            num_gloss_classes=num_classes,
            num_chinese_classes=num_classes,
            resnet_model='resnet50',
            resnet_pretrained=False,
            resnet_output_dim=512,
            lstm_hidden_size=512,
            use_ctc=True,
            use_transformer=True,
        ).to(self.device)
        
        # 加载权重
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.device)
        self.model.eval()
        
        # 打印信息
        total, trainable = count_parameters(self.model)
        print(f"\nModel loaded:")
        print(f"  Parameters: {total:,}")
        print(f"  Device: {self.device}")
        
        # 打印词汇表信息
        if self.vocab:
            print(f"  Vocabulary size: {len(self.vocab)}")
    
    @torch.no_grad()
    def predict_video(self, video_path: str) -> Dict:
        """
        预测单个视频
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            results: 预测结果字典
        """
        # 预处理
        frames = preprocess_video(video_path, num_frames=32)
        frames = frames.unsqueeze(0).to(self.device)  # (1, C, T, H, W)
        
        # 生成
        result = self.model.generate(
            video_frames=frames,
            sos_idx=self.vocab.word2idx.get('<sos>', 1),
            eos_idx=self.vocab.word2idx.get('<eos>', 2),
            max_len=100,
        )
        
        # 解码
        predictions = result['ctc_predictions']
        if predictions is not None:
            # 贪心解码
            indices = greedy_decode_ctc(predictions, blank=0)
            gloss_words = [self.vocab.decode(idx) for idx in indices[0]]
        else:
            gloss_words = []
        
        translations = result['translations']
        if translations is not None:
            chinese_ids = translations[0].tolist()
            # 移除 padding 和特殊 token
            chinese_words = [
                self.vocab.decode(idx) 
                for idx in chinese_ids 
                if idx not in [0, 1, 2, 3, 4]
            ]
            chinese_text = ''.join(chinese_words)
        else:
            chinese_text = ""
        
        return {
            'video_path': video_path,
            'gloss': gloss_words,
            'gloss_text': '/'.join(gloss_words),
            'translation': chinese_text,
            'confidence': 0.0,
        }
    
    @torch.no_grad()
    def predict_image(self, image_path: str) -> Dict:
        """
        预测单张图片
        
        Returns:
            results: 预测结果字典
        """
        # 预处理
        img_tensor = preprocess_image(image_path)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 复制为视频格式
        frames = img_tensor.repeat(1, 1, 32, 1, 1)
        
        # 生成
        result = self.model.generate(
            video_frames=frames,
            sos_idx=self.vocab.word2idx.get('<sos>', 1),
            eos_idx=self.vocab.word2idx.get('<eos>', 2),
            max_len=50,
        )
        
        # 解码
        predictions = result['ctc_predictions']
        if predictions is not None:
            indices = greedy_decode_ctc(predictions, blank=0)
            gloss_words = [self.vocab.decode(idx) for idx in indices[0]]
        else:
            gloss_words = []
        
        return {
            'image_path': image_path,
            'gloss': gloss_words,
            'gloss_text': '/'.join(gloss_words),
            'confidence': 0.0,
        }
    
    def predict_directory(self, dir_path: str, extensions: List[str] = None) -> List[Dict]:
        """
        预测目录中的所有文件
        
        Args:
            dir_path: 目录路径
            extensions: 支持的文件扩展名
            
        Returns:
            results: 预测结果列表
        """
        extensions = extensions or ['.mp4', '.avi', '.mov', '.jpg', '.jpeg', '.png']
        
        results = []
        
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    
                    try:
                        if file.endswith(('.mp4', '.avi', '.mov')):
                            result = self.predict_video(file_path)
                        else:
                            result = self.predict_image(file_path)
                        
                        result['file_path'] = file_path
                        results.append(result)
                        
                        print(f"{file}: {result.get('gloss_text', '')}")
                        
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        results.append({
                            'file_path': file_path,
                            'error': str(e),
                        })
        
        return results
    
    def save_results(self, results: List[Dict], save_path: str):
        """保存结果到 JSON 文件"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {save_path}")


# ==================== 评估函数 ====================
def calculate_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> float:
    """计算 BLEU 分数"""
    from collections import Counter
    
    total_score = 0.0
    
    for ref, hyp in zip(references, hypotheses):
        # 截断到相同长度
        min_len = min(len(ref), len(hyp))
        
        if min_len == 0:
            continue
        
        # 计算各 n-gram 的 precision
        precisions = []
        
        for n in range(1, min(max_n + 1, min_len + 1)):
            ref_ngrams = Counter([' '.join(ref[i:i+n]) for i in range(len(ref)-n+1)])
            hyp_ngrams = Counter([' '.join(hyp[i:i+n]) for i in range(len(hyp)-n+1)])
            
            matches = sum((ref_ngrams & hyp_ngrams).values())
            total = sum(hyp_ngrams.values())
            
            if total > 0:
                precisions.append(matches / total)
            else:
                precisions.append(0)
        
        if precisions and precisions[0] > 0:
            log_precision = sum(np.log(p + 1e-10) for p in precisions) / len(precisions)
            bleu = np.exp(log_precision)
            bp = min(1.0, np.exp(1 - len(ref) / (len(hyp) + 1e-10)))
            total_score += bleu * bp
    
    return 100 * total_score / len(references) if references else 0.0


def evaluate_dataset(inference: SignLanguageTranslatorInference, data_loader, num_samples: int = 100) -> Dict:
    """评估数据集"""
    inference.model.eval()
    
    references = []
    hypotheses = []
    
    pbar = tqdm(data_loader, desc="Evaluating")
    
    for i, batch in enumerate(pbar):
        if i >= num_samples:
            break
        
        # 获取数据
        frames = batch[0].to(inference.device)
        gloss_ids = batch[1]
        
        # 生成
        result = inference.model.generate(
            video_frames=frames,
            sos_idx=inference.vocab.word2idx.get('<sos>', 1),
            eos_idx=inference.vocab.word2idx.get('<eos>', 2),
            max_len=50,
        )
        
        # 解码
        predictions = result['ctc_predictions']
        if predictions is not None:
            indices = greedy_decode_ctc(predictions, blank=0)
            hyp = [inference.vocab.decode(idx) for idx in indices[0]]
        else:
            hyp = []
        
        # 参考
        ref = [inference.vocab.decode(idx.item()) for idx in gloss_ids[0] if idx.item() not in [0, 1, 2, 3, 4]]
        
        references.append(ref)
        hypotheses.append(hyp)
        
        pbar.set_postfix({
            'refs': len(references),
        })
    
    # 计算 BLEU
    bleu = calculate_bleu(references, hypotheses)
    
    return {
        'bleu': bleu,
        'num_samples': len(references),
        'references': references,
        'hypotheses': hypotheses,
    }


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='Sign Language Translation Testing')
    
    # 检查点
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    
    # 推理模式
    parser.add_argument('--test-video', type=str, default='', help='Test single video')
    parser.add_argument('--test-image', type=str, default='', help='Test single image')
    parser.add_argument('--test-dir', type=str, default='', help='Test directory')
    parser.add_argument('--eval', action='store_true', help='Evaluate on validation set')
    
    # 数据参数
    parser.add_argument('--data-root', type=str, default='D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-frames', type=int, default=32)
    
    # 输出参数
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--save-results', action='store_true', help='Save results to JSON')
    
    # 解码参数
    parser.add_argument('--beam-search', action='store_true', help='Use beam search')
    parser.add_argument('--beam-size', type=int, default=5, help='Beam size')
    
    # 其他
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建推理器
    print("=" * 70)
    print("Sign Language Translation - Testing")
    print("=" * 70)
    
    inference = SignLanguageTranslatorInference(
        checkpoint_path=args.checkpoint,
        device=device,
        use_beam_search=args.beam_search,
        beam_size=args.beam_size,
    )
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # 执行测试
    if args.test_video:
        print(f"\nTesting video: {args.test_video}")
        result = inference.predict_video(args.test_video)
        print(f"\nResult:")
        print(f"  Gloss: {result.get('gloss_text', '')}")
        print(f"  Translation: {result.get('translation', '')}")
    
    elif args.test_image:
        print(f"\nTesting image: {args.test_image}")
        result = inference.predict_image(args.test_image)
        print(f"\nResult:")
        print(f"  Gloss: {result.get('gloss_text', '')}")
    
    elif args.test_dir:
        print(f"\nTesting directory: {args.test_dir}")
        results = inference.predict_directory(args.test_dir)
        
        if args.save_results:
            save_path = os.path.join(output_dir, 'test_results.json')
            inference.save_results(results, save_path)
    
    elif args.eval:
        print(f"\nEvaluating on validation set...")
        print(f"Number of samples: {args.num_samples}")
        
        # 创建数据加载器
        from dataset import create_dataloaders
        _, val_loader, _, vocab = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
        )
        
        # 评估
        eval_results = evaluate_dataset(inference, val_loader, args.num_samples)
        
        print(f"\nEvaluation Results:")
        print(f"  BLEU: {eval_results['bleu']:.2f}")
        print(f"  Samples: {eval_results['num_samples']}")
        
        # 保存结果
        results = {
            'bleu': eval_results['bleu'],
            'num_samples': eval_results['num_samples'],
            'references': eval_results['references'][:10],
            'hypotheses': eval_results['hypotheses'][:10],
        }
        
        save_path = os.path.join(output_dir, 'eval_results.json')
        inference.save_results(results, save_path)
    
    else:
        print("\nNo test mode specified. Use --test-video, --test-image, --test-dir, or --eval")
        parser.print_help()
    
    print("\n" + "=" * 70)
    print("Testing completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()




