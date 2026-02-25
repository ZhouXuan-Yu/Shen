"""
ResNet 视觉特征提取器

参考 AAA/执行.md 文档实现:
- 使用预训练的 ResNet-50 提取空间特征
- 移除最后的分类层
- 支持层冻结和微调

架构说明:
- 输入: (B, C, T, H, W) - 视频帧序列
- 输出: (B, T, feature_dim) - 时序特征序列
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple
import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import cfg
except ImportError:
    from config import cfg


class ResNetFeatureExtractor(nn.Module):
    """
    基于 ResNet 的视频特征提取器
    
    特点:
    1. 使用预训练的 ResNet-50 作为骨干网络
    2. 提取帧级别的空间特征
    3. 支持层冻结和微调
    4. 降维以减少后续计算量
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        output_dim: int = 512,
        freeze_layers: int = 10,
        dropout: float = 0.0,
    ):
        """
        Args:
            model_name: ResNet 模型名称 ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            pretrained: 是否使用 ImageNet 预训练权重
            output_dim: 输出特征维度
            freeze_layers: 冻结的层数 (0 表示不冻结)
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze_layers = freeze_layers
        
        # 获取 ResNet 配置
        resnet_configs = {
            'resnet18': {'model': models.resnet18, 'backbone_dim': 512, 'params': '11.7M'},
            'resnet34': {'model': models.resnet34, 'backbone_dim': 512, 'params': '21.8M'},
            'resnet50': {'model': models.resnet50, 'backbone_dim': 2048, 'params': '25.6M'},
            'resnet101': {'model': models.resnet101, 'backbone_dim': 2048, 'params': '44.5M'},
        }
        
        if model_name not in resnet_configs:
            print(f"Warning: {model_name} not found, using resnet50")
            model_name = 'resnet50'
        
        config = resnet_configs[model_name]
        backbone_model = config['model']
        backbone_dim = config['backbone_dim']
        
        # 加载预训练的 ResNet
        if pretrained:
            weights_name = f'ResNet{model_name.replace("resnet", "")}_Weights'
            try:
                weights = getattr(models, weights_name, models.ResNet50_Weights.IMAGENET1K_V1)
                self.backbone = backbone_model(weights=weights.IMAGENET1K_V1)
            except:
                self.backbone = backbone_model(pretrained=True)
        else:
            self.backbone = backbone_model(weights=None)
        
        # 移除最后的全局平均池化和分类层
        # ResNet: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # 降维层
        self.fc = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(backbone_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )
        
        # 冻结层
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # 初始化 fc 层
        self._init_fc()
    
    def _freeze_layers(self, num_layers: int):
        """冻结前 num_layers 层"""
        layers = list(self.backbone.children())
        for i, layer in enumerate(layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"  Frozen first {num_layers} layers of ResNet")
    
    def _init_fc(self):
        """初始化全连接层"""
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def unfreeze_last_n_layers(self, n: int = 10):
        """解冻最后 n 层"""
        layers = list(self.backbone.children())
        num_layers = len(layers)
        start_idx = max(0, num_layers - n)
        
        for i in range(start_idx, num_layers):
            for param in layers[i].parameters():
                param.requires_grad = True
        
        print(f"  Unfrozen last {n} layers of ResNet")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, T, H, W)
            
        Returns:
            特征张量 (B, T, output_dim)
        """
        B, C, T, H, W = x.shape
        
        # 合并批次和时间维度: (B, C, T, H, W) -> (B*T, C, H, W)
        x = x.view(B * C, T, H, W)
        x = x.reshape(B * T, C, H, W)
        
        # 提取特征
        features = self.backbone(x)  # (B*T, backbone_dim, 7, 7) 或 (B*T, backbone_dim, 1, 1)
        
        # 全局平均池化
        features = F.adaptive_avg_pool2d(features, (1, 1))  # (B*T, backbone_dim, 1, 1)
        features = features.view(B * T, -1)  # (B*T, backbone_dim)
        
        # 降维
        features = self.fc(features)  # (B*T, output_dim)
        
        # 恢复批次和时间维度: (B*T, output_dim) -> (B, T, output_dim)
        features = features.view(B, T, self.output_dim)
        
        return features


class VideoResNet(nn.Module):
    """
    适用于视频的 ResNet 变体
    
    在时间维度上应用 3D 卷积或时间池化
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        output_dim: int = 512,
        temporal_pooling: str = 'mean',  # 'mean', 'max', 'attention'
    ):
        super().__init__()
        
        self.pooling_type = temporal_pooling
        self.output_dim = output_dim
        
        # 加载 2D ResNet
        self.resnet = ResNetFeatureExtractor(
            model_name=model_name,
            pretrained=pretrained,
            output_dim=output_dim,
            freeze_layers=0,
        )
        
        # 时间池化层
        if temporal_pooling == 'attention':
            self.temporal_attention = nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.Tanh(),
                nn.Linear(output_dim // 4, 1),
            )
        elif temporal_pooling == 'lstm':
            self.temporal_lstm = nn.LSTM(
                input_size=output_dim,
                hidden_size=output_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.output_dim = output_dim  # bidirectional * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
            
        Returns:
            (B, output_dim) 或 (B, T, output_dim)
        """
        B, C, T, H, W = x.shape
        
        # 提取每帧特征
        frame_features = self.resnet(x)  # (B, T, output_dim)
        
        # 时间池化
        if self.pooling_type == 'mean':
            pooled = frame_features.mean(dim=1)  # (B, output_dim)
        elif self.pooling_type == 'max':
            pooled = frame_features.max(dim=1)[0]  # (B, output_dim)
        elif self.pooling_type == 'attention':
            attn_weights = self.temporal_attention(frame_features)  # (B, T, 1)
            attn_weights = F.softmax(attn_weights, dim=1)
            pooled = (frame_features * attn_weights).sum(dim=1)  # (B, output_dim)
        elif self.pooling_type == 'lstm':
            pooled, (h_n, _) = self.temporal_lstm(frame_features)
            # 拼接双向 hidden state
            pooled = torch.cat([h_n[0], h_n[1]], dim=1)  # (B, output_dim)
        else:
            pooled = frame_features.mean(dim=1)
        
        return pooled, frame_features


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation 模块
    用于特征重标定
    """
    
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, T)
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1)
        x = x * y.expand_as(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        return x


# ==================== 模型参数统计 ====================
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_info(model: nn.Module, input_shape: Tuple = (1, 3, 32, 224, 224)):
    """打印模型信息"""
    total_params, trainable_params = count_parameters(model)
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 测试前向传播
    device = next(model.parameters()).device
    x = torch.randn(input_shape).to(device)
    
    model.eval()
    with torch.no_grad():
        try:
            output = model(x)
            print(f"\nForward pass test:")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output mean: {output.mean().item():.4f}")
            print(f"  Output std: {output.std().item():.4f}")
        except Exception as e:
            print(f"  Forward pass failed: {e}")


# ==================== 测试代码 ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ResNet Feature Extractor')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--output-dim', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-frames', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Testing ResNet Feature Extractor")
    print("=" * 70)
    
    # 创建模型
    model = ResNetFeatureExtractor(
        model_name=args.model,
        pretrained=args.pretrained,
        output_dim=args.output_dim,
        freeze_layers=10,
    )
    
    print_model_info(
        model,
        input_shape=(args.batch_size, 3, args.num_frames, args.img_size, args.img_size)
    )
    
    # 详细测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    x = torch.randn(args.batch_size, 3, args.num_frames, args.img_size, args.img_size).to(device)
    
    print("\nDetailed forward pass:")
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"  Input:  {x.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Min: {output.min().item():.4f}, Max: {output.max().item():.4f}")
    
    # 梯度测试
    print("\nGradient test:")
    model.train()
    output = model(x)
    loss = output.mean()
    loss.backward()
    
    grad_norm = torch.sqrt(sum([(p.grad ** 2).sum() for p in model.parameters() if p.grad is not None])).item()
    print(f"  Gradient norm: {grad_norm:.4f}")
    
    # 检查可训练参数
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,}")
    
    print("\n" + "=" * 70)
    print("ResNet Feature Extractor test passed!")
    print("=" * 70)

