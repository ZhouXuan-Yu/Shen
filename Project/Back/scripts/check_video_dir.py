import os

# 检查目录结构
video_root = 'D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL/video'
print('Video root exists:', os.path.exists(video_root))

# 列出A目录
if os.path.exists(os.path.join(video_root, 'A')):
    files = os.listdir(os.path.join(video_root, 'A'))
    print(f'Files in A directory (first 10): {sorted(files)[:10]}')
    
# 检查实际文件名格式
for root, dirs, files in os.walk(video_root):
    if files:
        print(f'\n{root}:')
        print(f'  Sample files: {sorted(files)[:5]}')
        break




