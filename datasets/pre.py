import os
import random

folder_path = 'labels'
train_ratio = 0.85

filenames = os.listdir(folder_path)
filenames = [f.split('.')[0] for f in filenames] 

train_num = int(len(filenames) * train_ratio)
random.shuffle(filenames)

train_set = filenames[:train_num]
val_set = filenames[train_num:]

with open('seg_txt/train.txt', 'w') as f:
    f.write('\n'.join(train_set))
    
with open('seg_txt/val.txt', 'w') as f:
    f.write('\n'.join(val_set))