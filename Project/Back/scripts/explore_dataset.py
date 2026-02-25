"""
CSL-News 数据集探索脚本
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset

def explore_dataset():
    """探索 CSL-News 数据集结构"""
    print("=" * 60)
    print("加载 CSL-News 数据集...")
    print("=" * 60)
    
    # 加载数据集（只下载训练集）
    ds = load_dataset("ZechengLi19/CSL-News", trust_remote_code=True)
    
    print("\n数据集信息:")
    print(f"  数据集名称: ZechengLi19/CSL-News")
    print(f"  数据集配置: {ds}")
    
    # 查看所有配置
    print("\n数据集划分:")
    for split, data in ds.items():
        print(f"  {split}: {len(data)} 条样本")
    
    # 查看训练集详细信息
    train_data = ds['train']
    print(f"\n训练集字段:")
    print(f"  字段列表: {train_data.column_names}")
    
    # 查看前几条样本
    print("\n样本示例 (前3条):")
    for i in range(min(3, len(train_data))):
        sample = train_data[i]
        print(f"\n样本 {i+1}:")
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            elif isinstance(value, list) and len(value) > 5:
                print(f"  {key}: [{len(value)} items] - 前5个: {value[:5]}")
            else:
                print(f"  {key}: {value}")
    
    # 统计标签分布
    if 'label' in train_data.column_names:
        print("\n标签分布统计:")
        from collections import Counter
        labels = [sample['label'] for sample in train_data]
        label_counts = Counter(labels)
        print(f"  唯一标签数量: {len(label_counts)}")
        print(f"  标签分布 (前10个):")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    标签 {label}: {count} 样本")
    
    return ds

if __name__ == "__main__":
    explore_dataset()

