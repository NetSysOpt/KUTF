from model import *
from helper import *
import pickle
import gzip
import os
from alive_progress import alive_bar

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--type','-t', type=str, default='')
args = parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# max_k = 100
# m = PDQP_Net_AR(1,1,64,max_k = max_k, threshold = 1e-4,nlayer=1).to(device)
# max_k = 20
# m = PDQP_Net_AR(1,1,128,max_k = max_k, threshold = 1e-4,nlayer=1).to(device)
# max_k = 40
# m = PDQP_Net_AR(1,1,128,max_k = max_k, threshold = 1e-4,nlayer=2).to(device)
# max_k = 300
# m = PDQP_Net_shared(1,1,64,max_k = max_k, threshold = 1e-8,nlayer=2).to(device)
# max_k = 1
# m = PDQP_Net_shared(1,1,128,max_k = max_k, threshold = 1e-8,nlayer=16).to(device)
config = getConfig(args.type)
max_k = int(config['max_k'])
nlayer = int(config['nlayer'])
net_width = int(config['net_width'])
model_mode = int(config['model_mode'])
mode = config['mode']
use_dual = True

accum_loss = True
if int(config['accum_loss'])==0:
    accum_loss = False


inf_time = max_k
if 'inf_time' in config:
    inf_time = int(config['inf_time'])


pareto = True
if int(config['pareto']) == 0:
    pareto = False
draw = True
if int(config['draw']) == 0:
    draw = False
use_dual = True

if int(config['usedual']) == 0:
    use_dual = False
# m = PDQP_Net_shared(1,1,128,max_k = max_k, threshold = 1e-8,nlayer=16).to(device)
# m = PDQP_Net_AR(1,1,128,max_k = max_k, threshold = 1e-8,nlayer=8).to(device)
m = None

eta_opt=None
# if 'eta_opt' in config:
#     eta_opt = float(config['eta_opt'])

ident = f'k{max_k}_{nlayer}'

if model_mode == 0:
    m = PDQP_Net_shared(1,1,net_width,max_k = 1, threshold = 1e-8,nlayer=nlayer,type='linf').to(device)
elif model_mode == 1:
    m = PDQP_Net_AR(1,1,net_width,max_k = 1, threshold = 1e-8,nlayer=nlayer,type='linf',use_dual=use_dual).to(device)
    ident += '_AR'
elif model_mode == 2:
    m = PDQP_Net_AR_geq(1,1,net_width,max_k = 1, threshold = 1e-8,nlayer=nlayer,tfype='linf',use_dual=use_dual).to(device)
    ident += '_ARgeq'
elif model_mode == 3:
    m = PDQP_Net_AR_geq(1,1,net_width,max_k = 1, threshold = 1e-8,nlayer=nlayer,tfype='linf',use_dual=use_dual,eta_opt=eta_opt,mode=0).to(device)
    ident += '_ARgeq'

# modf = relKKT_real()
modf = relKKT_general(mode = 'linf',eta_opt = eta_opt)
# modf = relKKT_general(type_modef)

# m = PDQP_Net_AR(1,1,128,max_k = 30, threshold = 1e-4,nlayer=1).to(device)
with torch.no_grad():
    # create model
    # layer_sizes = [64,64,64,64,1]
    # layer_sizes = [128,128,128,128,1]
    # m = PDQP_Net(2,3,layer_sizes).to(device)
    # m = PDQP_Net_new(1,1,128).to(device)
    lr1 = 1e-6

    train_tar_dir = '../pkl/train'
    valid_tar_dir = '../pkl/valid'
    train_files = os.listdir(train_tar_dir)
    valid_files = os.listdir(valid_tar_dir)

    # mode = 'cont'
    # mode = 'cont_temp'
    # mode = 'qplib_8938'
    # singe_mode = False
    if mode == 'single':
        valid_tar_dir = '../pkl/train'
        train_files = ['CONT-201.QPS.pkl']
        # train_files = ['CONT-201_0.QPS.pkl']
        valid_files = []
        ident += '_single'
    if mode == 'cont':
        valid_tar_dir = '../pkl/cont_valid'
        train_files = []
        valid_files = os.listdir(valid_tar_dir)
        ident += '_cont'
    if mode == 'cont_temp':
        valid_tar_dir = '../pkl/cont_valid'
        train_files = []
        valid_files = os.listdir(valid_tar_dir)
        ident += '_cont_temp'
    if mode == 'qplib_8938':
        valid_tar_dir = '../pkl/8938_valid'
        train_files = []
        valid_files = os.listdir(valid_tar_dir)
        ident += '_qplib_8938'
    if mode == 'qplib_8785':
        valid_tar_dir = '../pkl/8785_valid'
        train_files = []
        valid_files = os.listdir(valid_tar_dir)
        ident += '_qplib_8785'
    if mode == 'qplib_8906':
        valid_tar_dir = '../pkl/8906_valid'
        train_files = []
        valid_files = os.listdir(valid_tar_dir)
        ident += '_qplib_8906'
    elif mode == 'qplib_8602':
        train_tar_dir = '../pkl/8602_train'
        valid_tar_dir = '../pkl/8602_valid'
        train_files = os.listdir(train_tar_dir)
        valid_files = os.listdir(valid_tar_dir)
        ident += '_qplib_8602'
    elif mode == 'qplib_8845':
        train_tar_dir = '../pkl/8845_train'
        valid_tar_dir = '../pkl/8845_valid'
        train_files = os.listdir(train_tar_dir)
        valid_files = os.listdir(valid_tar_dir)
        ident += '_qplib_8845'
    elif mode == 'qplib_9008':
        train_tar_dir = '../pkl/9008_train'
        valid_tar_dir = '../pkl/9008_valid'
        train_files = os.listdir(train_tar_dir)
        valid_files = os.listdir(valid_tar_dir)
        ident += '_qplib_9008'
    elif mode == 'qplib_8547':
        train_tar_dir = '../pkl/8547_train'
        valid_tar_dir = '../pkl/8547_valid'
        train_files = os.listdir(train_tar_dir)
        valid_files = os.listdir(valid_tar_dir)
        ident += '_qplib_8547'
    else:
        mode1 = mode.replace('qplib_','')
        train_tar_dir = f'../pkl/{mode1}_train'
        valid_tar_dir = f'../pkl/{mode1}_valid'
        train_files = os.listdir(train_tar_dir)
        valid_files = os.listdir(valid_tar_dir)
        if len(valid_files) == 0:
            valid_files.append(train_files[0])
            valid_tar_dir = train_tar_dir
        if len(train_files) ==1:
            for i in range(100):
                train_files.append(train_files[0])
        ident += f'_{mode}'

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr1)
    max_epoch = 10000
    best_loss = 1e+20

    if accum_loss:
        ident+='_accLoss'
    last_epoch=0
    if os.path.exists(f"../model/best_pdqp{ident}.mdl"):
        checkpoint = torch.load(f"../model/best_pdqp{ident}.mdl")
        m.load_state_dict(checkpoint['model'])
        if 'nepoch' in checkpoint:
            last_epoch=checkpoint['nepoch']
        best_loss=checkpoint['best_loss']
        print(f'Last best val loss gen:  {best_loss}')
        print('Model Loaded')


    with alive_bar(len(valid_files),title=f"Validating part") as bar:
        for fnm in valid_files:
            inference(m,fnm,last_epoch,valid_tar_dir,pareto,device,modf,inf_time)
            bar()

        


