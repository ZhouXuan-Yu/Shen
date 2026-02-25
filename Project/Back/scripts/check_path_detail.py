import os
import pandas as pd

# 读取CSV
df = pd.read_csv('D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL/label/train.csv')

# 检查第一行
row = df.iloc[0]
number = row['Number']
translator = row['Translator']

# 尝试不同的路径格式
path1 = f'D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL/video/{translator}/{number}.mp4'
path2 = f'D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL/video/{translator.lower()}/{number}.mp4'

print(f'Number: {number}')
print(f'Translator: {translator}')
print(f'Path 1: {path1}')
print(f'Path 1 exists: {os.path.exists(path1)}')
print(f'Path 2: {path2}')
print(f'Path 2 exists: {os.path.exists(path2)}')

# 列出目录中的文件
video_dir = f'D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL/video/{translator}'
if os.path.exists(video_dir):
    files = os.listdir(video_dir)
    print(f'\nFiles in {video_dir}:')
    print(f'  Count: {len(files)}')
    print(f'  First 5: {sorted(files)[:5]}')
else:
    print(f'\nDirectory does not exist: {video_dir}')




