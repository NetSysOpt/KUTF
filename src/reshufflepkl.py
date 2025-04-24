import os
import shutil
import random

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--type','-t', type=str, default='')
parser.add_argument('--ratio','-r', type=float, default=0.05)
parser.add_argument('--seed','-s', type=int, default=0)
args = parser.parse_args()

random.seed(args.seed)

train_folder = f'../pkl/{args.type}_train'
valid_folder = f'../pkl/{args.type}_valid'

train_files = os.listdir(train_folder)
valid_files = os.listdir(valid_folder)

# print(train_files)
# print(valid_files)

# stp1: move valids to train
for fnm in valid_files:
    print(f'{valid_folder}/{fnm}', f'{train_folder}/{fnm}')
    shutil.move(f'{valid_folder}/{fnm}', f'{train_folder}/{fnm}') 

train_files += valid_files
train_files.sort()
random.shuffle(train_files)

n_valid = int(round(len(train_files)*args.ratio))
print(n_valid)
for fnm in train_files[:n_valid]:
    print(f'{train_folder}/{fnm}', f'{valid_folder}/{fnm}' )
    shutil.move(f'{train_folder}/{fnm}', f'{valid_folder}/{fnm}') 


