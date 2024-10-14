from model import *
from helper import *
import pickle
import gzip
import os
from alive_progress import alive_bar
import random 

# torch.backends.cudnn.enabled=False

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--type','-t', type=str, default='')
parser.add_argument('--sl','-s', type=int, default=0)
args = parser.parse_args()

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
r = torch.cuda.mem_get_info(device)
# device = torch.device("cpu")
# create model
# m = PDQP_Net_new(2,3,128).to(device)

mode = 'single'
mode = 'qplib_8938'
mode = 'cont'

pareto = False
pareto = True


config = getConfig(args.type)
max_k = int(config['max_k'])
nlayer = int(config['nlayer'])
lr1 = float(config['lr'])
net_width = int(config['net_width'])
model_mode = int(config['model_mode'])
mode = config['mode']
eta_opt = float(config['eta_opt'])

if 'gpu' in config:
    dev = int(config['gpu'])
    device = torch.device(f"cuda:{dev}" if torch.cuda.is_available() else "cpu")


div = 1.0
if 'div' in config:
    div = float(config['div'])

accum_loss = True
if int(config['accum_loss'])==0:
    accum_loss = False

if int(config['pareto']) == 0:
    pareto = False
draw = True
if int(config['draw']) == 0:
    draw = False
use_dual = True
if int(config['usedual']) == 0:
    use_dual = False


Contu = False
if int(config['Contu'])==1:
    Contu = True
choose_weight = False
if int(config['choose_weight'])==1:
    choose_weight = True
# choose_weight = True

# m = PDQP_Net_AR(1,1,128,max_k = max_k, threshold = 1e-8,nlayer=2,type='linf').to(device)
m = None

ident = f'k{max_k}_{nlayer}'

type_modef = 'linf'
type_modef = 'l2'


if model_mode == 0:
    m = PDQP_Net_shared(1,1,net_width,max_k = 1, threshold = 1e-8,nlayer=nlayer,type=type_modef).to(device)
elif model_mode == 1:
    m = PDQP_Net_AR(1,1,net_width,max_k = 1, threshold = 1e-8,nlayer=nlayer,type=type_modef,use_dual=use_dual).to(device)
    ident += '_AR'
elif model_mode == 2:
    m = PDQP_Net_AR_geq(1,1,net_width,max_k = 1, threshold = 1e-8,nlayer=nlayer,tfype=type_modef,use_dual=use_dual,eta_opt=eta_opt,div=div).to(device)
    ident += '_ARgeq'
elif model_mode == 3:
    m = PDQP_Net_AR_geq(1,1,net_width,max_k = 1, threshold = 1e-8,nlayer=nlayer,tfype=type_modef,use_dual=use_dual,eta_opt=eta_opt,div=div,mode=0).to(device)
    ident += '_ARgeq'

    

# modf=None
# if type_modef == 'linf':
#     modf = relKKT().to(device)
# elif type_modef == 'l2':
#     modf = relKKT_l2().to(device)

# modf = relKKT_general(type_modef)
modf = relKKT_real()
modf = relKKT_general(mode = 'linf')



# m = PDQP_Net_AR(1,1,128,nlayer=2).to(device)

train_tar_dir = '../pkl/train'
valid_tar_dir = '../pkl/valid'
train_files = os.listdir(train_tar_dir)
valid_files = os.listdir(valid_tar_dir)


if mode == 'single':
    valid_tar_dir = '../pkl/train'
    train_files = ['CONT-201.QPS.pkl']
    valid_files = ['CONT-201.QPS.pkl']
    ident += '_single'
elif mode == 'cont':
    train_tar_dir = '../pkl/cont_train'
    valid_tar_dir = '../pkl/cont_valid'
    train_files = os.listdir(train_tar_dir)
    valid_files = os.listdir(valid_tar_dir)
    ident += '_cont'
elif mode == 'qplib_8938':
    train_tar_dir = '../pkl/8938_train'
    valid_tar_dir = '../pkl/8938_valid'
    train_files = os.listdir(train_tar_dir)
    valid_files = os.listdir(valid_tar_dir)
    ident += '_qplib_8938'
    # ident += '_qplib_8938_test'
elif mode == 'qplib_8785':
    train_tar_dir = '../pkl/8785_train'
    valid_tar_dir = '../pkl/8785_valid'
    train_files = os.listdir(train_tar_dir)
    valid_files = os.listdir(valid_tar_dir)
    ident += '_qplib_8785'
elif mode == 'qplib_8906':
    train_tar_dir = '../pkl/8906_train'
    valid_tar_dir = '../pkl/8906_valid'
    train_files = os.listdir(train_tar_dir)
    valid_files = os.listdir(valid_tar_dir)
    ident += '_qplib_8906'
    # ident += '_qplib_8938_test'
elif mode == 'qplib_8602':
    train_tar_dir = '../pkl/8602_train'
    valid_tar_dir = '../pkl/8602_valid'
    train_files = os.listdir(train_tar_dir)
    valid_files = os.listdir(valid_tar_dir)
    ident += '_qplib_8602'
    # ident += '_qplib_8938_test'
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
        for i in range(20):
            train_files.append(train_files[0])
    ident += f'_{mode}'


loss_func = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(m.parameters(), lr=lr1)
optimizer = torch.optim.AdamW(m.parameters(), lr=lr1)
max_epoch = 1000000
best_loss = 1e+20
flog = open('../logs/training_log.log','w')
last_epoch=0

if accum_loss:
    ident+='_accLoss'

loss_log = open(f'../logs/train_{mode}.log','a+')

loaded = False
if os.path.exists(f"../model/best_pdqp{ident}.mdl") and Contu:
    checkpoint = torch.load(f"../model/best_pdqp{ident}.mdl")
    m.load_state_dict(checkpoint['model'])
    if 'nepoch' in checkpoint:
        last_epoch=checkpoint['nepoch']
    best_loss=checkpoint['best_loss']
    print(f'Last best val loss gen:  {best_loss}')
    print('Model Loaded')
    loaded=True

save_log = True
if args.sl == 0:
    save_log = False
if save_log:
    valid_files.sort()
    tar = f'{valid_tar_dir}/{valid_files[-1]}'
    f_gg = open(f'../plots/distance/logs/{args.type}_{valid_files[-1]}.rec','a+')

for epoch in range(last_epoch,max_epoch):
    avg_train_loss = process(m,train_files,epoch,train_tar_dir,pareto=pareto,device=device,optimizer=optimizer,choose_weight=choose_weight,autoregression_iteration=max_k,accu_loss = accum_loss)
    avg_train_loss = avg_train_loss[-1] / len(train_files)

    avg_valid_loss,avg_sc, avg_scprimal, avg_scdual, avg_scgap = process(m,valid_files,epoch,valid_tar_dir,pareto=pareto,device=device,optimizer=modf,choose_weight=choose_weight,autoregression_iteration=max_k,training=False)
    avg_valid_loss = avg_valid_loss[-1] / len(valid_files)
    avg_sc = avg_sc[-1]/len(valid_files)

    for i in range(max_k):
        avg_scprimal[i] = avg_scprimal[i]/len(valid_files)
        avg_scdual[i] = avg_scdual[i]/len(valid_files)
        avg_scgap[i] = avg_scgap[i]/len(valid_files)


    st =f'{epoch} {avg_train_loss} {avg_valid_loss}\n'
    loss_log.write(st)
    loss_log.flush()

    st = f'epoch{epoch}: train: {avg_train_loss} | valid: {avg_valid_loss}\n'
    flog.write(st)
    flog.flush()
    print(f'Epoch{epoch}: train loss:{avg_train_loss}    valid loss:{avg_valid_loss}')

    if save_log:
        x,y,sc,pres,dres,gap,x_norm,y_norm = sol_check_model(tar,device,modf,m)
        st = f'{epoch} {pres.item()} {dres.item()} {gap.item()} {sc.item()} {x_norm.item()} {y_norm.item()}\n'
        f_gg.write(st)
        f_gg.flush()

    # if best_loss > avg_valid_loss:
    #     best_loss = avg_valid_loss
    #     state={'model':m.state_dict(),'optimizer':optimizer.state_dict(),'best_loss':best_loss,'nepoch':epoch}
    #     torch.save(state,f'../model/best_pdqp{ident}.mdl')
    #     print(f'Saving new best model with valid loss: {best_loss}')
    #     st = f'     Saving new best model with valid loss: {best_loss}\n'
    #     flog.write(st)
    #     flog.flush()
    if loaded:
        draw_plot(y=avg_scprimal, ident=f'primal_{mode}')
        draw_plot(y=avg_scdual, ident=f'dual_{mode}')
        draw_plot(y=avg_scgap, ident=f'gap_{mode}')
        loaded=False
    if best_loss > avg_sc:
        draw_plot(y=avg_scprimal, ident=f'primal_{mode}')
        draw_plot(y=avg_scdual, ident=f'dual_{mode}')
        draw_plot(y=avg_scgap, ident=f'gap_{mode}')
        best_loss = avg_sc
        state={'model':m.state_dict(),'optimizer':optimizer.state_dict(),'best_loss':avg_sc,'nepoch':epoch}
        torch.save(state,f'../model/best_pdqp{ident}.mdl')
        print(f'Saving new best model with valid loss: {avg_sc}')
        st = f'     Saving new best model with valid loss: {avg_sc}\n'
        flog.write(st)
        flog.flush()




flog.close()
f_gg.close()