from model import *
from helper import *
import pickle
import gzip
import os
from alive_progress import alive_bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create model
layer_sizes = [64,64,64,64,1]
layer_sizes = [128,128,128,128,1]
m = PDQP_Net(2,3,layer_sizes).to(device)
lr1 = 1e-6

train_files = os.listdir('../pkl/train')
valid_files = os.listdir('../pkl/valid/')

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(m.parameters(), lr=lr1)
max_epoch = 10000
best_loss = 1e+20
flog = open('../logs/training_log.log','w')

last_epoch=0
if os.path.exists(f"../model/best_pdqp.mdl"):
    checkpoint = torch.load(f"../model/best_pdqp.mdl")
    m.load_state_dict(checkpoint['model'])
    if 'nepoch' in checkpoint:
        last_epoch=checkpoint['nepoch']
    best_loss=checkpoint['best_loss']
    print(f'Last best val loss gen:  {best_loss}')
    print('Model Loaded')


with alive_bar(len(valid_files),title=f"Validating part") as bar:
    for fnm in valid_files:
        f_tar = gzip.open(f'../pkl/valid/{fnm}','rb')
        to_pack = pickle.load(f_tar)
        v_feat = to_pack['vf'].to(device)
        c_feat = to_pack['cf'].to(device)
        Q = to_pack['Q'].to(device)
        A = to_pack['A'].to(device)
        c = to_pack['c'].to(device)
        b = to_pack['b'].to(device)
        x = to_pack['x'].to(device)
        y = to_pack['y'].to(device)
        vscale = torch.as_tensor(to_pack['vscale']).to(device).unsqueeze(-1)
        cscale = torch.as_tensor(to_pack['cscale'] ).to(device).unsqueeze(-1)
        constscale = torch.as_tensor(to_pack['constscale']).to(device).unsqueeze(-1)
        f_tar.close()
        
        x_pred,y_pred = m(A,Q,b,c,v_feat,c_feat)
        x_pred = torch.div(x_pred,vscale)
        x_pred = x_pred * constscale
        
        y_pred = torch.div(y_pred,cscale)
        y_pred = y_pred * constscale
        
        ff = open(f'../predictions/primal_{fnm}.sol','w')
        st=''
        for xv in x_pred:
            st = st+f'{xv} '
        st = st + '\n'
        ff.write(st)
        ff.close()
        
        ff = open(f'../predictions/dual_{fnm}.sol','w')
        st=''
        for xv in y_pred:
            st = st+f'{xv} '
        st = st + '\n'
        ff.write(st)
        ff.close()
        
        
        bar()
            
with alive_bar(len(train_files),title=f"Training part") as bar:
    for fnm in train_files:
        f_tar = gzip.open(f'../pkl/train/{fnm}','rb')
        to_pack = pickle.load(f_tar)
        v_feat = to_pack['vf'].to(device)
        c_feat = to_pack['cf'].to(device)
        Q = to_pack['Q'].to(device)
        A = to_pack['A'].to(device)
        c = to_pack['c'].to(device)
        b = to_pack['b'].to(device)
        x = to_pack['x'].to(device)
        y = to_pack['y'].to(device)
        vscale = torch.as_tensor(to_pack['vscale']).to(device).unsqueeze(-1)
        cscale = torch.as_tensor(to_pack['cscale'] ).to(device).unsqueeze(-1)
        constscale = torch.as_tensor(to_pack['constscale']).to(device).unsqueeze(-1)
        f_tar.close()
        x_pred,y_pred = m(A,Q,b,c,v_feat,c_feat)
        x_pred = torch.div(x_pred,vscale)
        x_pred = x_pred * constscale
        
        y_pred = torch.div(y_pred,cscale)
        y_pred = y_pred * constscale
        
        ff = open(f'../predictions/primal_{fnm}.sol','w')
        st=''
        for xv in x_pred:
            st = st+f'{xv} '
        st = st + '\n'
        ff.write(st)
        ff.close()
        
        ff = open(f'../predictions/dual_{fnm}.sol','w')
        st=''
        for xv in y_pred:
            st = st+f'{xv} '
        st = st + '\n'
        ff.write(st)
        ff.close()
        
        
        bar()
            

    



flog.close()