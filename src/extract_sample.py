from helper import *
import os
from alive_progress import alive_bar
import gzip
import pickle
import random
random.seed(0)
import multiprocessing



import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--type','-t', type=str, default='')
parser.add_argument('--start','-s', type=int, default=0)
parser.add_argument('--end','-e', type=int, default=-1)
args = parser.parse_args()

train_tar_dir = '../pkl/train'
valid_tar_dir = '../pkl/valid'
test_tar_dir = '../pkl/test'

    
train_ori_dir = '../ins/train'
valid_ori_dir = '../ins/valid'



mode = ''
# mode = 'single'
mode = 'cont201'
mode = 'qplib8938'
mode = ''
mode = 'qplib8602'
# mode = 'qplib8845'
mode = 'qplib9008'

mode = 'qplib8602'



mode = 'qplib8547'
mode = 'twod'
mode = 'qplib8785'
mode = 'qplib8906'
mode = 'qplib3913'
mode = 'qplib8845'
mode = 'syn'
mode = args.type
ident = ''
# train_files = os.listdir(train_ori_dir)
# valid_files = os.listdir(valid_ori_dir)


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
elif mode == 'qplib9008':
    train_tar_dir = '../pkl/9008_train'
    valid_tar_dir = '../pkl/9008_valid'
    train_ori_dir = '../ins/gen_train_9008'
    valid_ori_dir = '../ins/valid'
    train_files = os.listdir(train_ori_dir)
    valid_files = os.listdir(valid_ori_dir)
    ident = 'qplib9008'
elif mode == 'qplib8547':
    train_tar_dir = '../pkl/8547_train'
    valid_tar_dir = '../pkl/8547_valid'
    train_ori_dir = '../ins/gen_train_8547'
    valid_ori_dir = '../ins/valid'
    train_files = os.listdir(train_ori_dir)
    valid_files = os.listdir(valid_ori_dir)
    ident = 'qplib8547'
else:
    mode1 = mode.replace('qplib','')
    train_tar_dir = f'../pkl/{mode1}_train'
    valid_tar_dir = f'../pkl/{mode1}_valid'
    train_ori_dir = f'../ins/gen_train_{mode1}'
    valid_ori_dir = f'../ins/valid'
    train_files = os.listdir(train_ori_dir)
    # valid_files = os.listdir(valid_ori_dir)

flitt = set(os.listdir('../transformed_ins/train'))
for ff in train_files:
    mode1 = mode.replace('qplib','')
    pdd = f'../transformed_ins/train/{ff}'
    if ff not in flitt:
        # print(f'file missing: {ff}')
        fdir = f'../../../ins/gen_train_{mode1}/{ff}'
        st = 'julia scripts/solve_save.jl'
        st = f'{st} --instance_path={fdir} --output_directory=../../../logs --time_sec_limit=3600 --solve=1'
        print(st)

    
cont = input("Proceed? (y/N)")
if not ('1' in cont or 'Y' in cont or 'y' in cont):
    quit()
    

if args.end>0:
    train_files=train_files[args.start:args.end]
else:
    if not os.path.exists("../pkl"):
        os.mkdir('../pkl')
    if not os.path.exists(train_tar_dir):
        os.mkdir(train_tar_dir)
    if not os.path.exists(valid_tar_dir):
        os.mkdir(valid_tar_dir)
    if not os.path.exists(test_tar_dir):
        os.mkdir(test_tar_dir)
        
    old_files = os.listdir(train_tar_dir)
    print(f'Cleaning {len(old_files)} old training files')
    for fi in old_files:
        os.remove(f"{train_tar_dir}/{fi}")
    old_files = os.listdir(valid_tar_dir)
    print(f'Cleaning {len(old_files)} old training files')
    for fi in old_files:
        os.remove(f"{valid_tar_dir}/{fi}")
    old_files = os.listdir(test_tar_dir)
    print(f'Cleaning {len(old_files)} old training files')
    for fi in old_files:
        os.remove(f"{test_tar_dir}/{fi}")
        
        
failed = 0

failed_ins = []

instances = train_files
n_files = len(instances)
train_files = instances[:int(round(n_files*0.85))]
valid_files = instances[int(round(n_files*0.85)):int(round(n_files*0.95))]
test_files = instances[int(round(n_files*0.95)):]

print('   train|   valid|    test')
print(f'{len(train_files):<8}|{len(valid_files):<8}|{len(test_files):<8}')

cont = input("Proceed? (y/N)")
if not ('1' in cont or 'Y' in cont or 'y' in cont):
    quit()

with alive_bar(len(train_files),title=f"Generating Training samples") as bar:
    for fnm in train_files:
        # v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_scaled_sparse(f'{valid_ori_dir}/{fnm}')
        # _, _, Q_ori, A_ori, c_ori, b_ori, x_ori, y_ori, vscale_ori, cscale_ori, constscale_ori, var_lb_ori, var_ub_ori, vars_ident_l_ori, vars_ident_u_ori, cons_ident_ori = extract_solfile_unscaled_sparse(f'{valid_ori_dir}/{fnm}')

        # for i in range(vars_ident_u.shape[0]):
        #     if vars_ident_u[i]!=vars_ident_u_ori[i]:
        #         print(vars_ident_u[i],vars_ident_u_ori[i])
        # quit()
        try:
            v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_scaled_sparse(f'{valid_ori_dir}/{fnm}')
            _, _, Q_ori, A_ori, c_ori, b_ori, x_ori, y_ori, vscale_ori, cscale_ori, constscale_ori, var_lb_ori, var_ub_ori, vars_ident_l_ori, vars_ident_u_ori, cons_ident_ori = extract_solfile_unscaled_sparse(f'{valid_ori_dir}/{fnm}')

            # v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_scaled(f'{train_ori_dir}/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_unscaled(f'{train_ori_dir}/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y = extract_solfile(f'../ins/train/{fnm}')
            

            to_pack = {}
            to_pack['vf'] = v_feat
            to_pack['cf'] = c_feat
            to_pack['Q'] = Q.float()
            to_pack['A'] = A.float()
            to_pack['c'] = torch.as_tensor(c).float()
            to_pack['b'] = torch.as_tensor(b).float()
            to_pack['Q_ori'] = Q_ori.float()
            to_pack['A_ori'] = A_ori.float()
            to_pack['c_ori'] = torch.as_tensor(c_ori).float()
            to_pack['b_ori'] = torch.as_tensor(b_ori).float()

            to_pack['x'] = x
            to_pack['y'] = y

            to_pack['vscale'] = torch.as_tensor(vscale).float()
            to_pack['cscale'] =  torch.as_tensor(cscale).float()
            to_pack['constscale'] =  torch.as_tensor(constscale).float()

            to_pack['var_lb'] = var_lb
            to_pack['var_ub'] = var_ub
            to_pack['var_lb_ori'] = var_lb_ori
            to_pack['var_ub_ori'] = var_ub_ori
            to_pack['vars_ident_l'] = vars_ident_l
            to_pack['vars_ident_u'] = vars_ident_u
            to_pack['cons_ident'] = cons_ident
            f_tar = gzip.open(f'{train_tar_dir}/{fnm}.pkl','wb')
            pickle.dump(to_pack,f_tar)
            f_tar.close()
        except:
            print(f'skip this file {fnm}')
            failed+=1
            failed_ins.append(fnm)
            # quit()
        bar()


    
with alive_bar(len(valid_files),title=f"Generating Validating samples") as bar:
    for fnm in valid_files:
        try:
            v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_scaled_sparse(f'{valid_ori_dir}/{fnm}')
            _, _, Q_ori, A_ori, c_ori, b_ori, x_ori, y_ori, vscale_ori, cscale_ori, constscale_ori, var_lb_ori, var_ub_ori, vars_ident_l_ori, vars_ident_u_ori, cons_ident_ori = extract_solfile_unscaled_sparse(f'{valid_ori_dir}/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_unscaled_sparse(f'{valid_ori_dir}/{fnm}')

            # v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_scaled(f'{valid_ori_dir}/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_unscaled(f'{valid_ori_dir}/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y = extract_solfile_scaled(f'../ins/valid/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y = extract_solfile(f'../ins/train/{fnm}')
            to_pack = {}
            to_pack['vf'] = v_feat
            to_pack['cf'] = c_feat
            to_pack['Q'] = Q.float()
            to_pack['A'] = A.float()
            to_pack['Q_ori'] = Q_ori.float()
            to_pack['A_ori'] = A_ori.float()
            to_pack['c_ori'] = torch.as_tensor(c_ori).float()
            to_pack['b_ori'] = torch.as_tensor(b_ori).float()
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
            failed_ins.append(fnm)
            # quit()
        bar()



with alive_bar(len(test_files),title=f"Generating Testing samples") as bar:
    for fnm in test_files:
        try:
            v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_scaled_sparse(f'{valid_ori_dir}/{fnm}')
            _, _, Q_ori, A_ori, c_ori, b_ori, x_ori, y_ori, vscale_ori, cscale_ori, constscale_ori, var_lb_ori, var_ub_ori, vars_ident_l_ori, vars_ident_u_ori, cons_ident_ori = extract_solfile_unscaled_sparse(f'{valid_ori_dir}/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_unscaled_sparse(f'{valid_ori_dir}/{fnm}')

            # v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_scaled(f'{valid_ori_dir}/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident = extract_solfile_unscaled(f'{valid_ori_dir}/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y = extract_solfile_scaled(f'../ins/valid/{fnm}')
            # v_feat, c_feat, Q, A, c, b, x, y = extract_solfile(f'../ins/train/{fnm}')
            to_pack = {}
            to_pack['vf'] = v_feat
            to_pack['cf'] = c_feat
            to_pack['Q'] = Q.float()
            to_pack['A'] = A.float()
            to_pack['Q_ori'] = Q_ori.float()
            to_pack['A_ori'] = A_ori.float()
            to_pack['c_ori'] = torch.as_tensor(c_ori).float()
            to_pack['b_ori'] = torch.as_tensor(b_ori).float()
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
            f_tar = gzip.open(f'{test_tar_dir}/{fnm}.pkl','wb')
            pickle.dump(to_pack,f_tar)
            f_tar.close()
        except:
            print(f'skip this file {fnm}')
            failed+=1
            failed_ins.append(fnm)
            # quit()
        bar()
    

if len(valid_files) == 0:
    # no valid file generated, seperate the training set
    ori_folder = f'{train_tar_dir}/'
    tar_folder = f'{valid_tar_dir}/'
    sample_files = os.listdir(ori_folder)
    if args.end<=0:
        old_files = os.listdir(tar_folder)
        print(f'Cleaning {len(old_files)} old validating files')
        for fi in old_files:
            os.remove(f"{tar_folder}{fi}")
    length = len(sample_files)
    rate = 0.05
    random.shuffle(sample_files)
    print(f'Splitting into {int(round(length*(1-rate)))} training files, {int(round(length*rate))} validating files')
    for sample in sample_files[:int(round(length*rate))]:
        ori_name = f'{ori_folder}{sample}'
        tar_name = f'{tar_folder}{sample}'
        os.rename(ori_name, tar_name)

print(f'Failed: {failed}')
print(failed_ins)