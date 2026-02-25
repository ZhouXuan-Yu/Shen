import pandas as pd
import os
import cv2

# Load dataset
df = pd.read_csv('D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL/label/train.csv')

# Parse gloss
df['gloss_list'] = df['Gloss'].apply(lambda x: [g for g in str(x).split('/') if g.strip()])
df['gloss_len'] = df['gloss_list'].apply(len)

print('=== Dataset Statistics ===')
print('Total samples:', len(df))

# Unique gloss count
all_glosses = [g for gl in df['gloss_list'] for g in gl]
unique_glosses = set(all_glosses)
print('Unique gloss classes:', len(unique_glosses))
print('Avg gloss per sentence:', df['gloss_len'].mean())
print('Min:', df['gloss_len'].min(), 'Max:', df['gloss_len'].max())

# Check video
sample_path = 'D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL/video/train/A/train-00001.mp4'
print('\nVideo exists:', os.path.exists(sample_path))

cap = cv2.VideoCapture(sample_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = frame_count / fps if fps > 0 else 0
cap.release()
print('Frames:', frame_count, 'FPS:', fps, 'Duration:', duration)




