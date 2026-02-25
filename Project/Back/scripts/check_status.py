"""
训练进度检查脚本
"""
import os
import sys
from pathlib import Path
import json

# 检查模型文件是否存在
MODEL_DIR = Path(__file__).parent.parent / "models" / "sign_language"

def check_training_status():
    """检查训练状态"""
    print("=" * 60)
    print("训练进度检查")
    print("=" * 60)
    
    # 检查必要的文件
    files_to_check = [
        ("sign_language_model.pth", "模型权重文件"),
        ("label_encoder.pkl", "标签编码器文件"),
        ("label_map.json", "标签映射文件"),
    ]
    
    all_exist = True
    for filename, description in files_to_check:
        filepath = MODEL_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"[OK] {description}: {filename} ({size_mb:.2f} MB)")
        else:
            print(f"[MISSING] {description}: {filename} (not found)")
            all_exist = False
    
    # 检查模型目录
    if MODEL_DIR.exists():
        print(f"\n模型目录: {MODEL_DIR}")
        print(f"目录内容: {list(MODEL_DIR.iterdir())}")
    else:
        print(f"\n模型目录不存在: {MODEL_DIR}")
    
    return all_exist

if __name__ == "__main__":
    check_training_status()

