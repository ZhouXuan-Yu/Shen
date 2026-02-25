import os
import pandas as pd

# 读取数据
df = pd.read_csv('D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL/label/train.csv')
print('CSV columns:', df.columns.tolist())
print('Sample row:', df.iloc[0].to_dict())

# 检查视频路径
sample = df.iloc[0]
number = sample['Number']
translator = sample['Translator']
print(f'\nNumber: {number}')
print(f'Translator: {translator}')

# 构建路径
video_path = f'D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL/video/{translator}/{number}.mp4'
print(f'Video path: {video_path}')
print(f'Exists: {os.path.exists(video_path)}')

# 列出目录
if os.path.exists(f'D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL/video/{translator}'):
    files = os.listdir(f'D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL/video/{translator}')
    print(f'First 5 files: {files[:5]}')




