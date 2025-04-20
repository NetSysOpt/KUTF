import os
from alive_progress import alive_bar
import argparse
import shutil
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--folder','-f', type=str, default='mm')
parser.add_argument('--ratiotrain','-rt', type=float, default=1)
parser.add_argument('--ratiovalid','-rv', type=float, default=0)
args = parser.parse_args()
import random
random.seed(0)

src_dir = f'../{args.folder}'

tar_train = f'../gen_train_{args.folder}'
tar_valid = f'../gen_valid_{args.folder}'
tar_test = f'../gen_test_{args.folder}'
print(f'train folder: {tar_train}')
print(f'valid folder: {tar_valid}')
print(f'test folder: {tar_valid}')
if os.path.exists(tar_train):
    shutil.rmtree(tar_train)
if os.path.exists(tar_valid):
    shutil.rmtree(tar_valid)
if os.path.exists(tar_test):
    shutil.rmtree(tar_test)
os.mkdir(tar_train)
if args.ratiovalid != 0:
    os.mkdir(tar_valid)
if args.ratiovalid + args.ratiotrain < 1.0:
    os.mkdir(tar_test)


f_all = os.listdir(src_dir)
random.shuffle(f_all)


n_train = int(len(f_all) * args.ratiotrain)
n_valid = int(len(f_all) * args.ratiovalid)
n_test = len(f_all) - n_train - n_valid
print(f'n_train: {n_train}, n_valid: {n_valid}, n_test: {n_test}')

for i in range(n_train):
    f = f_all[i]
    src = os.path.join(src_dir, f)
    tar = os.path.join(tar_train, f)
    os.system(f'cp {src} {tar}')
    print(f'train: {f}')

if args.ratiovalid != 0:
    for i in range(n_train, n_valid+n_train):
        f = f_all[i]
        src = os.path.join(src_dir, f)
        tar = os.path.join(tar_valid, f)
        os.system(f'cp {src} {tar}')
        print(f'valid: {f}')

if args.ratiovalid + args.ratiotrain < 1.0:
    for i in range(n_train+n_valid, n_train+n_valid+n_test):    
        f = f_all[i]
        src = os.path.join(src_dir, f)
        tar = os.path.join(tar_test, f)
        os.system(f'cp {src} {tar}')
        print(f'test: {f}')