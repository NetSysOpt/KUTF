from helper import *
import os
from alive_progress import alive_bar
import gzip
import pickle
import random
random.seed(0)

train_tar_dir = '../pkl/train'
valid_tar_dir = '../pkl/valid'

    
train_ori_dir = '../ins/train'
valid_ori_dir = '../ins/valid'



mode = ''
# mode = 'single'
mode = 'cont201'
mode = 'qplib8938'
mode = ''
mode = 'qplib8602'
mode = 'qplib8785'
mode = 'qplib8906'
# mode = 'qplib8845'
ident = ''
train_files = os.listdir(train_ori_dir)
valid_files = os.listdir(valid_ori_dir)


if mode == 'single':
    # train_ori_dir = '../ins/gen_train_cont'
    # train_files = ['CONT-201_0.QPS.pkl']
    # valid_files = []
    valid_ori_dir = '../ins/train'
    train_files = ['CONT-201.QPS.pkl']
    valid_files = ['CONT-201.QPS.pkl']
    ident = 'single'
elif mode == 'cont201':
    train_tar_dir = '../pkl/cont_train'
    valid_tar_dir = '../pkl/cont_valid'
    train_ori_dir = '../ins/gen_train_cont'
    valid_ori_dir = '../ins/valid'
    train_files = os.listdir(train_ori_dir)
    valid_files = os.listdir(valid_ori_dir)
    ident = 'cont201'
elif mode == 'qplib8938':
    train_tar_dir = '../pkl/8938_train'
    valid_tar_dir = '../pkl/8938_valid'
    train_ori_dir = '../ins/gen_train_8938'
    valid_ori_dir = '../ins/valid'
    train_files = os.listdir(train_ori_dir)
    valid_files = os.listdir(valid_ori_dir)
    ident = 'qplib8938'
elif mode == 'qplib8785':
    train_tar_dir = '../pkl/8785_train'
    valid_tar_dir = '../pkl/8785_valid'
    train_ori_dir = '../ins/gen_train_8785'
    valid_ori_dir = '../ins/valid'
    train_files = os.listdir(train_ori_dir)
    valid_files = os.listdir(valid_ori_dir)
    ident = 'qplib8785'
elif mode == 'qplib8602':
    train_tar_dir = '../pkl/8602_train'
    valid_tar_dir = '../pkl/8602_valid'
    train_ori_dir = '../ins/gen_train_8602'
    valid_ori_dir = '../ins/valid'
    train_files = os.listdir(train_ori_dir)
    valid_files = os.listdir(valid_ori_dir)
    ident = 'qplib8602'
elif mode == 'qplib8906':
    train_tar_dir = '../pkl/8906_train'
    valid_tar_dir = '../pkl/8906_valid'
    train_ori_dir = '../ins/gen_train_8906'
    valid_ori_dir = '../ins/valid'
    train_files = os.listdir(train_ori_dir)
    valid_files = os.listdir(valid_ori_dir)
    ident = 'qplib8906'
elif mode == 'qplib8845':
    train_tar_dir = '../pkl/8845_train'
    valid_tar_dir = '../pkl/8845_valid'
    train_ori_dir = '../ins/gen_train_8845'
    valid_ori_dir = '../ins/valid'
    train_files = os.listdir(train_ori_dir)
    valid_files = os.listdir(valid_ori_dir)
    ident = 'qplib8845'
elif mode == 'qplib_9008':
    train_tar_dir = '../pkl/9008_train'
    valid_tar_dir = '../pkl/9008_valid'
    train_ori_dir = '../ins/gen_train_9008'
    valid_ori_dir = '../ins/valid'
    train_files = os.listdir(train_ori_dir)
    valid_files = os.listdir(valid_ori_dir)
    ident = 'qplib_9008'

if not os.path.exists(train_tar_dir):
    os.mkdir(train_tar_dir)
if not os.path.exists(valid_tar_dir):
    os.mkdir(valid_tar_dir)
    
old_files = os.listdir(train_tar_dir)
print(f'Cleaning {len(old_files)} old training files')
for fi in old_files:
    os.remove(f"{train_tar_dir}/{fi}")
failed = 0

with alive_bar(len(train_files),title=f"Generating Training samples") as bar:
    for fnm in train_files:
        try:
            # v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_scaled(f'{train_ori_dir}/{fnm}')
            v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_unscaled(f'{train_ori_dir}/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y = extract_solfile(f'../ins/train/{fnm}')
            

            to_pack = {}
            to_pack['vf'] = v_feat
            to_pack['cf'] = c_feat
            to_pack['Q'] = Q.float()
            to_pack['A'] = A.float()
            to_pack['c'] = torch.as_tensor(c).float()
            to_pack['b'] = torch.as_tensor(b).float()
            to_pack['x'] = x
            to_pack['y'] = y
            to_pack['vscale'] = vscale
            to_pack['cscale'] = cscale
            to_pack['constscale'] = constscale
            to_pack['var_lb'] = var_lb
            to_pack['var_ub'] = var_ub
            to_pack['vars_ident_l'] = vars_ident_l
            to_pack['vars_ident_u'] = vars_ident_u
            to_pack['cons_ident'] = cons_ident
            f_tar = gzip.open(f'{train_tar_dir}/{fnm}.pkl','wb')
            pickle.dump(to_pack,f_tar)
            f_tar.close()
        except:
            print(f'skip this file {fnm}')
            failed+=1
            # quit()
        bar()


    
with alive_bar(len(valid_files),title=f"Generating Validating samples") as bar:
    for fnm in valid_files:
        try:
            # v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_scaled(f'{valid_ori_dir}/{fnm}')
            v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_unscaled(f'{valid_ori_dir}/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y = extract_solfile_scaled(f'../ins/valid/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y = extract_solfile(f'../ins/train/{fnm}')
            to_pack = {}
            to_pack['vf'] = v_feat
            to_pack['cf'] = c_feat
            to_pack['Q'] = Q.float()
            to_pack['A'] = A.float()
            # to_pack['c'] = c
            # to_pack['b'] = b
            to_pack['c'] = torch.as_tensor(c).float()
            to_pack['b'] = torch.as_tensor(b).float()
            to_pack['x'] = x
            to_pack['y'] = y
            to_pack['vscale'] = vscale
            to_pack['cscale'] = cscale
            to_pack['constscale'] = constscale
            to_pack['var_lb'] = var_lb
            to_pack['var_ub'] = var_ub
            to_pack['vars_ident_l'] = vars_ident_l
            to_pack['vars_ident_u'] = vars_ident_u
            to_pack['cons_ident'] = cons_ident
            f_tar = gzip.open(f'{valid_tar_dir}/{fnm}.pkl','wb')
            pickle.dump(to_pack,f_tar)
            f_tar.close()
        except:
            print(f'skip this file {fnm}')
            failed+=1
            # quit()
        bar()
    

if len(valid_files) == 0:
    # no valid file generated, seperate the training set
    ori_folder = f'{train_tar_dir}/'
    tar_folder = f'{valid_tar_dir}/'
    sample_files = os.listdir(ori_folder)
    old_files = os.listdir(tar_folder)
    print(f'Cleaning {len(old_files)} old validating files')
    for fi in old_files:
        os.remove(f"{tar_folder}{fi}")
    length = len(sample_files)
    rate = 0.1
    random.shuffle(sample_files)
    print(f'Splitting into {int(round(length*(1-rate)))} training files, {int(round(length*rate))} validating files')
    for sample in sample_files[:int(round(length*rate))]:
        ori_name = f'{ori_folder}{sample}'
        tar_name = f'{tar_folder}{sample}'
        os.rename(ori_name, tar_name)

print(f'Failed: {failed}')