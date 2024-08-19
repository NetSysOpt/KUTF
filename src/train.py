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
lr1 = 1e-5

train_files = os.listdir('../pkl/train')
valid_files = os.listdir('../pkl/valid/')

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(m.parameters(), lr=lr1)
max_epoch = 1000000
best_loss = 1e+20
flog = open('../logs/training_log.log','w')
Contu = False
last_epoch=0
if os.path.exists(f"../model/best_pdqp.mdl") and Contu:
    checkpoint = torch.load(f"../model/best_pdqp.mdl")
    m.load_state_dict(checkpoint['model'])
    if 'nepoch' in checkpoint:
        last_epoch=checkpoint['nepoch']
    best_loss=checkpoint['best_loss']
    print(f'Last best val loss gen:  {best_loss}')
    print('Model Loaded')


for epoch in range(max_epoch):
    avg_train_loss = 0.0
    with alive_bar(len(train_files),title=f"Training epoch {epoch}") as bar:
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
            f_tar.close()

            # print(Q.shape)
            # print(A.shape)
            # print(c.shape)
            # print(b.shape)
            # print(x.shape)
            # print(y.shape)
            # quit()
            


            optimizer.zero_grad()
            x_pred,y_pred = m(A,Q,b,c,v_feat,c_feat)
            # print(v_feat,c_feat)

            x_scaled = x
            y_scaled = y
            xm = x.max()
            ym = y.max()
            if xm!=0.0:
                x_scaled = x_scaled/xm
                x_pred = x_pred/xm
            if ym!=0.0:
                y_scaled = y_scaled/ym
                y_pred = y_pred/ym

            # loss1 = loss_func(x_pred, x)
            # loss2 = loss_func(y_pred, y)
            loss1 = loss_func(x_pred, x_scaled)
            loss2 = loss_func(y_pred, y_scaled)

            loss = loss1+loss2
            print(fnm,loss1.item(),loss2.item(),xm,ym)
            # QGROW7.QPS.pkl
            # QGROW15.QPS.pkl
            # QSHARE1B.QPS.pkl
            # POWELL20.QPS.pkl
            # STADAT1.QPS.pkl
            if loss1.item()/x_pred.shape[0]+loss2.item()/y_pred.shape[0]>1e+12:
                continue
                for i in range(min(x_pred.shape[0],100)):
                    print(x_pred[i],x[i])
                print(fnm)
                input('Paused')
            avg_train_loss += loss1.item()/x_pred.shape[0]
            avg_train_loss += loss2.item()/y_pred.shape[0]
            loss.backward()
            optimizer.step()
            bar()
    avg_train_loss = avg_train_loss / len(train_files)
    # quit()
    avg_valid_loss = 0.0
    with alive_bar(len(valid_files),title=f"Validating epoch {epoch}") as bar:
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
            f_tar.close()

            x_pred,y_pred = m(A,Q,b,c,v_feat,c_feat)

            x_scaled = x
            y_scaled = y
            xm = x.max()
            ym = y.max()
            if xm!=0.0:
                x_scaled = x_scaled/xm
                x_pred = x_pred/xm
            if ym!=0.0:
                y_scaled = y_scaled/ym
                y_pred = y_pred/ym

            # loss1 = loss_func(x_pred, x)
            # loss2 = loss_func(y_pred, y)
            loss1 = loss_func(x_pred, x_scaled)
            loss2 = loss_func(y_pred, y_scaled)
            loss = loss1+loss2
            if loss1.item()/x_pred.shape[0]+loss2.item()/y_pred.shape[0]>1e+12:
                continue
            avg_valid_loss += loss1.item()/x_pred.shape[0]
            avg_valid_loss += loss2.item()/y_pred.shape[0]
            bar()
    avg_valid_loss = avg_valid_loss / len(valid_files)

    
    st = f'epoch{epoch}: train: {avg_train_loss} | valid: {avg_valid_loss}\n'
    flog.write(st)
    flog.flush()
    print(f'Epoch{epoch}: train loss:{avg_train_loss}    valid loss:{avg_valid_loss}')

    if best_loss > avg_valid_loss:
        best_loss = avg_valid_loss
        state={'model':m.state_dict(),'optimizer':optimizer.state_dict(),'best_loss':best_loss,'nepoch':epoch}
        torch.save(state,f'../model/best_pdqp.mdl')
        print(f'Saving new best model with valid loss: {best_loss}')
        st = f'     Saving new best model with valid loss: {best_loss}\n'
        flog.write(st)
        flog.flush()




flog.close()