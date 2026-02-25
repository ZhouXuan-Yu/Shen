"""
专业论文级别可视化模块
生成科研论文风格的可视化图表
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

# 设置 matplotlib 支持中文和科研风格
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
})


# 自定义颜色主题
COLORS = {
    'primary': '#2E86AB',      # 深蓝
    'secondary': '#A23B72',    # 紫红
    'tertiary': '#F18F01',     # 橙色
    'quaternary': '#C73E1D',   # 红色
    'success': '#3A7D44',     # 绿色
    'warning': '#F4B942',      # 黄色
    'loss': '#E63946',         # 红色系
    'accuracy': '#457B9D',     # 蓝色系
    'gradient_loss': plt.cm.Reds,
    'gradient_acc': plt.cm.Blues,
}

# 训练日志存储
TRAINING_HISTORY = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'lr': [],
    'epoch_time': [],
    'confusion_matrix': None,
    'class_accuracy': {},
    'misclassification': [],
}


def setup_chinese_font():
    """设置中文字体"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass


class PaperVisualizer:
    """
    科研论文风格可视化器
    
    生成功能:
    1. 训练曲线 (Loss/Accuracy)
    2. 混淆矩阵热力图
    3. 学习曲线分析
    4. t-SNE/PCA 特征可视化
    5. 各类别性能柱状图
    6. 模型性能雷达图
    7. 训练过程热力图
    8. 混淆样本分析
    """
    
    def __init__(self, output_dir: str = "results/plots"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "curves").mkdir(exist_ok=True)
        (self.output_dir / "confusion").mkdir(exist_ok=True)
        (self.output_dir / "features").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        
        # 设置样式
        plt.style.use('seaborn-v0_8-whitegrid')
        setup_chinese_font()
        
    def log_training(self, epoch: int, train_loss: float, val_loss: float, 
                     train_acc: float, val_acc: float, lr: float, epoch_time: float):
        """记录训练指标"""
        TRAINING_HISTORY['train_loss'].append(train_loss)
        TRAINING_HISTORY['val_loss'].append(val_loss)
        TRAINING_HISTORY['train_acc'].append(train_acc)
        TRAINING_HISTORY['val_acc'].append(val_acc)
        TRAINING_HISTORY['lr'].append(lr)
        TRAINING_HISTORY['epoch_time'].append(epoch_time)
        
    def set_confusion_matrix(self, confusion_matrix: np.ndarray):
        """设置混淆矩阵"""
        TRAINING_HISTORY['confusion_matrix'] = confusion_matrix
        
    def set_class_accuracy(self, class_accuracy: Dict[int, float]):
        """设置各类别准确率"""
        TRAINING_HISTORY['class_accuracy'] = class_accuracy
        
    def add_misclassification(self, true_label: int, pred_label: int, confidence: float):
        """添加误分类样本"""
        TRAINING_HISTORY['misclassification'].append({
            'true_label': true_label,
            'pred_label': pred_label,
            'confidence': confidence
        })
    
    # ==================== 1. 训练曲线 ====================
    
    def plot_training_curves(self, save: bool = True) -> plt.Figure:
        """
        绘制训练曲线（论文风格）
        
        包含:
        - 训练/验证损失
        - 训练/验证准确率
        - 双Y轴布局
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(TRAINING_HISTORY['train_loss']) + 1)
        
        # 左图: Loss 曲线
        ax1.plot(epochs, TRAINING_HISTORY['train_loss'], 
                color=COLORS['loss'], linewidth=2.5, label='Training Loss', marker='o', markersize=4)
        ax1.plot(epochs, TRAINING_HISTORY['val_loss'], 
                color=COLORS['primary'], linewidth=2.5, label='Validation Loss', marker='s', markersize=4)
        ax1.fill_between(epochs, TRAINING_HISTORY['train_loss'], alpha=0.1, color=COLORS['loss'])
        ax1.fill_between(epochs, TRAINING_HISTORY['val_loss'], alpha=0.1, color=COLORS['primary'])
        
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Training and Validation Loss', fontsize=14, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.set_xlim([1, len(epochs)])
        
        # 添加最佳点标注
        best_val_loss = min(TRAINING_HISTORY['val_loss'])
        best_epoch = TRAINING_HISTORY['val_loss'].index(best_val_loss) + 1
        ax1.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
        ax1.scatter([best_epoch], [best_val_loss], color='gold', s=100, zorder=5, edgecolors='black')
        ax1.annotate(f'Best: {best_val_loss:.4f}', 
                    xy=(best_epoch, best_val_loss),
                    xytext=(best_epoch+1, best_val_loss+0.1),
                    fontsize=10, color='gray')
        
        # 右图: Accuracy 曲线
        ax2.plot(epochs, TRAINING_HISTORY['train_acc'], 
                color=COLORS['accuracy'], linewidth=2.5, label='Training Accuracy', marker='o', markersize=4)
        ax2.plot(epochs, TRAINING_HISTORY['val_acc'], 
                color=COLORS['secondary'], linewidth=2.5, label='Validation Accuracy', marker='s', markersize=4)
        ax2.fill_between(epochs, TRAINING_HISTORY['train_acc'], alpha=0.1, color=COLORS['accuracy'])
        ax2.fill_between(epochs, TRAINING_HISTORY['val_acc'], alpha=0.1, color=COLORS['secondary'])
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Training and Validation Accuracy', fontsize=14, fontweight='bold', pad=10)
        ax2.legend(loc='lower right', framealpha=0.9)
        ax2.set_xlim([1, len(epochs)])
        
        # 添加最佳点标注
        best_val_acc = max(TRAINING_HISTORY['val_acc'])
        best_epoch = TRAINING_HISTORY['val_acc'].index(best_val_acc) + 1
        ax2.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
        ax2.scatter([best_epoch], [best_val_acc], color='gold', s=100, zorder=5, edgecolors='black')
        ax2.annotate(f'Best: {best_val_acc:.2f}%', 
                    xy=(best_epoch, best_val_acc),
                    xytext=(best_epoch+1, best_val_acc-2),
                    fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "curves" / "training_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")
            
        return fig
    
    # ==================== 2. 混淆矩阵 ====================
    
    def plot_confusion_matrix(self, labels: List[str] = None, save: bool = True, 
                             top_n: int = 20) -> plt.Figure:
        """
        绘制混淆矩阵热力图（论文风格）
        
        Args:
            labels: 类别标签列表
            save: 是否保存
            top_n: 显示前 n 个类别
        """
        if TRAINING_HISTORY['confusion_matrix'] is None:
            print("Warning: No confusion matrix available")
            return None
            
        cm = TRAINING_HISTORY['confusion_matrix']
        
        # 只显示前 top_n 个类别
        if cm.shape[0] > top_n:
            cm = cm[:top_n, :top_n]
            labels = labels[:top_n] if labels else [str(i) for i in range(top_n)]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 计算归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.nan_to_num(cm_normalized, nan=0.0)
        
        # 绘制热力图
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f',
                   cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels,
                   ax=ax,
                   cbar_kws={'label': 'Normalized Accuracy', 'shrink': 0.8},
                   annot_kws={'size': 8})
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Top-20 Classes)', fontsize=14, fontweight='bold', pad=15)
        
        # 调整标签
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "confusion" / "confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")
            
        return fig
    
    # ==================== 3. 各类别性能柱状图 ====================
    
    def plot_class_performance(self, class_names: List[str] = None, save: bool = True,
                               top_n: int = 30) -> plt.Figure:
        """
        绘制各类别性能柱状图
        """
        if not TRAINING_HISTORY['class_accuracy']:
            print("Warning: No class accuracy data available")
            return None
            
        # 排序并选择 top_n
        sorted_acc = sorted(TRAINING_HISTORY['class_accuracy'].items(), 
                           key=lambda x: x[1], reverse=True)
        
        if len(sorted_acc) > top_n:
            sorted_acc = sorted_acc[:top_n]
            
        classes = [f"Class {c[0]}" for c in sorted_acc]
        accuracies = [c[1] * 100 for c in sorted_acc]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 创建渐变色柱状图
        colors = [COLORS['gradient_acc'](i / len(accuracies)) for i in range(len(accuracies))]
        
        bars = ax.bar(range(len(accuracies)), accuracies, color=colors, 
                     edgecolor='white', linewidth=0.5)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.annotate(f'{acc:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=8)
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Class ID', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Classification Accuracy (Top-30 Classes)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim([0, 105])
        
        # 添加平均线
        avg_acc = np.mean(accuracies)
        ax.axhline(y=avg_acc, color='red', linestyle='--', linewidth=2, 
                  label=f'Average: {avg_acc:.1f}%')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "analysis" / "class_performance.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")
            
        return fig
    
    # ==================== 4. 学习率曲线 ====================
    
    def plot_learning_rate(self, save: bool = True) -> plt.Figure:
        """
        绘制学习率变化曲线
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        
        epochs = range(1, len(TRAINING_HISTORY['lr']) + 1)
        
        ax.plot(epochs, TRAINING_HISTORY['lr'], 
               color=COLORS['tertiary'], linewidth=2.5, marker='o', markersize=4)
        ax.fill_between(epochs, TRAINING_HISTORY['lr'], alpha=0.2, color=COLORS['tertiary'])
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim([1, len(epochs)])
        
        # 使用对数刻度（如果学习率变化大）
        if max(TRAINING_HISTORY['lr']) / min(TRAINING_HISTORY['lr']) > 100:
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "curves" / "learning_rate.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")
            
        return fig
    
    # ==================== 5. 训练时间分析 ====================
    
    def plot_training_time(self, save: bool = True) -> plt.Figure:
        """
        绘制训练时间分析图
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(TRAINING_HISTORY['epoch_time']) + 1)
        
        # 左图: 每 epoch 时间
        ax1.plot(epochs, TRAINING_HISTORY['epoch_time'], 
                color=COLORS['quaternary'], linewidth=2, marker='o', markersize=4)
        ax1.fill_between(epochs, TRAINING_HISTORY['epoch_time'], alpha=0.2, 
                        color=COLORS['quaternary'])
        
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Training Time per Epoch', fontsize=14, fontweight='bold', pad=10)
        
        # 右图: 累计时间
        cumulative_time = np.cumsum(TRAINING_HISTORY['epoch_time'])
        ax2.plot(epochs, cumulative_time / 60, 
                color=COLORS['success'], linewidth=2, marker='s', markersize=4)
        ax2.fill_between(epochs, cumulative_time / 60, alpha=0.2, 
                        color=COLORS['success'])
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Time (minutes)', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Training Time', fontsize=14, fontweight='bold', pad=10)
        
        # 添加总时间标注
        total_time = cumulative_time[-1]
        ax2.annotate(f'Total: {total_time/60:.1f} min',
                    xy=(len(epochs), cumulative_time[-1]/60),
                    xytext=(-30, -20),
                    textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "analysis" / "training_time.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")
            
        return fig
    
    # ==================== 6. 综合性能仪表盘 ====================
    
    def plot_performance_dashboard(self, save: bool = True) -> plt.Figure:
        """
        绘制综合性能仪表盘
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. 最终准确率
        ax1 = fig.add_subplot(gs[0, 0])
        final_acc = TRAINING_HISTORY['val_acc'][-1] if TRAINING_HISTORY['val_acc'] else 0
        final_loss = TRAINING_HISTORY['val_loss'][-1] if TRAINING_HISTORY['val_loss'] else 0
        
        ax1.text(0.5, 0.6, f'{final_acc:.2f}%', 
                ha='center', va='center', fontsize=32, fontweight='bold',
                color=COLORS['primary'])
        ax1.text(0.5, 0.3, 'Final Accuracy', 
                ha='center', va='center', fontsize=14, color='gray')
        ax1.axis('off')
        ax1.set_title('Model Performance', fontsize=14, fontweight='bold', pad=10)
        
        # 2. 最佳准确率
        ax2 = fig.add_subplot(gs[0, 1])
        best_acc = max(TRAINING_HISTORY['val_acc']) if TRAINING_HISTORY['val_acc'] else 0
        ax2.text(0.5, 0.6, f'{best_acc:.2f}%', 
                ha='center', va='center', fontsize=32, fontweight='bold',
                color=COLORS['success'])
        ax2.text(0.5, 0.3, 'Best Accuracy', 
                ha='center', va='center', fontsize=14, color='gray')
        ax2.axis('off')
        
        # 3. 训练轮数
        ax3 = fig.add_subplot(gs[0, 2])
        num_epochs = len(TRAINING_HISTORY['train_loss'])
        ax3.text(0.5, 0.6, f'{num_epochs}', 
                ha='center', va='center', fontsize=32, fontweight='bold',
                color=COLORS['secondary'])
        ax3.text(0.5, 0.3, 'Total Epochs', 
                ha='center', va='center', fontsize=14, color='gray')
        ax3.axis('off')
        
        # 4. Loss 曲线
        ax4 = fig.add_subplot(gs[1, :2])
        epochs = range(1, len(TRAINING_HISTORY['train_loss']) + 1)
        ax4.plot(epochs, TRAINING_HISTORY['train_loss'], 
                color=COLORS['loss'], linewidth=2, label='Train Loss')
        ax4.plot(epochs, TRAINING_HISTORY['val_loss'], 
                color=COLORS['primary'], linewidth=2, label='Val Loss')
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Loss', fontsize=11)
        ax4.legend(loc='upper right')
        ax4.set_title('Training Progress', fontsize=14, fontweight='bold', pad=10)
        
        # 5. 训练时间饼图
        ax5 = fig.add_subplot(gs[1, 2])
        total_time = sum(TRAINING_HISTORY['epoch_time'])
        ax5.pie([total_time], labels=['Training Time'], 
               colors=[COLORS['tertiary']], startangle=90)
        ax5.text(0, 0, f'{total_time/60:.1f} min', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax5.set_title('Total Time', fontsize=14, fontweight='bold', pad=10)
        
        # 6. Accuracy 曲线
        ax6 = fig.add_subplot(gs[2, :])
        ax6.plot(epochs, TRAINING_HISTORY['train_acc'], 
                color=COLORS['accuracy'], linewidth=2, label='Train Acc')
        ax6.plot(epochs, TRAINING_HISTORY['val_acc'], 
                color=COLORS['secondary'], linewidth=2, label='Val Acc')
        ax6.fill_between(epochs, TRAINING_HISTORY['train_acc'], alpha=0.1)
        ax6.fill_between(epochs, TRAINING_HISTORY['val_acc'], alpha=0.1)
        ax6.set_xlabel('Epoch', fontsize=11)
        ax6.set_ylabel('Accuracy (%)', fontsize=11)
        ax6.legend(loc='lower right')
        ax6.set_title('Accuracy Over Epochs', fontsize=14, fontweight='bold', pad=10)
        
        plt.suptitle('Sign Language Recognition - Training Dashboard', 
                    fontsize=18, fontweight='bold', y=1.02)
        
        if save:
            save_path = self.output_dir / "performance_dashboard.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")
            
        return fig
    
    # ==================== 7. 生成训练报告 ====================
    
    def generate_report(self) -> Dict:
        """
        生成训练报告
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'final_metrics': {
                'train_loss': TRAINING_HISTORY['train_loss'][-1] if TRAINING_HISTORY['train_loss'] else None,
                'val_loss': TRAINING_HISTORY['val_loss'][-1] if TRAINING_HISTORY['val_loss'] else None,
                'train_acc': TRAINING_HISTORY['train_acc'][-1] if TRAINING_HISTORY['train_acc'] else None,
                'val_acc': TRAINING_HISTORY['val_acc'][-1] if TRAINING_HISTORY['val_acc'] else None,
            },
            'best_metrics': {
                'best_val_loss': min(TRAINING_HISTORY['val_loss']) if TRAINING_HISTORY['val_loss'] else None,
                'best_val_acc': max(TRAINING_HISTORY['val_acc']) if TRAINING_HISTORY['val_acc'] else None,
                'best_epoch': TRAINING_HISTORY['val_loss'].index(min(TRAINING_HISTORY['val_loss'])) + 1 if TRAINING_HISTORY['val_loss'] else None,
            },
            'training_time': {
                'total_time_seconds': sum(TRAINING_HISTORY['epoch_time']),
                'avg_epoch_time': np.mean(TRAINING_HISTORY['epoch_time']) if TRAINING_HISTORY['epoch_time'] else None,
                'num_epochs': len(TRAINING_HISTORY['train_loss']),
            },
            'class_performance': {
                'top_10': sorted(TRAINING_HISTORY['class_accuracy'].items(), 
                                key=lambda x: x[1], reverse=True)[:10],
            }
        }
        
        # 保存报告
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved: {report_path}")
        
        return report
    
    # ==================== 8. 一键生成所有图表 ====================
    
    def generate_all_plots(self, class_names: List[str] = None):
        """
        一键生成所有图表
        """
        print("\n" + "="*60)
        print("Generating Professional Plots for Paper")
        print("="*60)
        
        plots = {}
        
        # 1. 训练曲线
        print("\n[1/6] Generating training curves...")
        plots['training_curves'] = self.plot_training_curves()
        
        # 2. 学习率曲线
        print("[2/6] Generating learning rate curve...")
        plots['learning_rate'] = self.plot_learning_rate()
        
        # 3. 训练时间分析
        print("[3/6] Generating training time analysis...")
        plots['training_time'] = self.plot_training_time()
        
        # 4. 混淆矩阵（如果有）
        if TRAINING_HISTORY['confusion_matrix'] is not None:
            print("[4/6] Generating confusion matrix...")
            plots['confusion_matrix'] = self.plot_confusion_matrix(class_names)
        
        # 5. 各类别性能
        if TRAINING_HISTORY['class_accuracy']:
            print("[5/6] Generating class performance chart...")
            plots['class_performance'] = self.plot_class_performance(class_names)
        
        # 6. 综合仪表盘
        print("[6/6] Generating performance dashboard...")
        plots['dashboard'] = self.plot_performance_dashboard()
        
        # 生成报告
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("All plots generated successfully!")
        print(f"Output directory: {self.output_dir}")
        print("="*60)
        
        return plots, report


# 全局可视化器实例
visualizer = None

def get_visualizer(output_dir: str = None) -> PaperVisualizer:
    """获取全局可视化器实例"""
    global visualizer
    if visualizer is None:
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "results" / "plots"
        visualizer = PaperVisualizer(str(output_dir))
    return visualizer

def reset_visualizer():
    """重置可视化器"""
    global visualizer
    visualizer = None
    # 重置训练历史
    TRAINING_HISTORY.update({
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': [],
        'epoch_time': [],
        'confusion_matrix': None,
        'class_accuracy': {},
        'misclassification': [],
    })


if __name__ == "__main__":
    print("="*60)
    print("Paper Visualizer - Professional Plot Generation")
    print("="*60)
    
    # 示例：生成虚拟数据的图表
    print("\nGenerating example plots with dummy data...")
    
    # 模拟训练数据
    for epoch in range(1, 21):
        TRAINING_HISTORY['train_loss'].append(2.5 * np.exp(-epoch/8) + 0.1 + np.random.normal(0, 0.05))
        TRAINING_HISTORY['val_loss'].append(2.2 * np.exp(-epoch/8) + 0.15 + np.random.normal(0, 0.05))
        TRAINING_HISTORY['train_acc'].append(min(95, 20 + 70 * (1 - np.exp(-epoch/6)) + np.random.normal(0, 2)))
        TRAINING_HISTORY['val_acc'].append(min(90, 15 + 65 * (1 - np.exp(-epoch/6)) + np.random.normal(0, 2)))
        TRAINING_HISTORY['lr'].append(0.001 * np.exp(-epoch/12))
        TRAINING_HISTORY['epoch_time'].append(30 + np.random.normal(0, 5))
    
    # 模拟混淆矩阵
    n_classes = 50
    cm = np.random.randint(0, 100, (n_classes, n_classes))
    np.fill_diagonal(cm, np.random.randint(80, 100, n_classes))
    TRAINING_HISTORY['confusion_matrix'] = cm
    
    # 模拟类别准确率
    for i in range(n_classes):
        TRAINING_HISTORY['class_accuracy'][i] = np.random.uniform(0.5, 1.0)
    
    # 生成所有图表
    viz = PaperVisualizer("results/plots_example")
    viz.generate_all_plots()
    
    print("\n" + "="*60)
    print("Example plots generated in 'results/plots_example'")
    print("="*60)

