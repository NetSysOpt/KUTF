from model import *
from helper import *
import pickle
import gzip
import os
from alive_progress import alive_bar
import random 

# torch.backends.cudnn.enabled=False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# create model
# m = PDQP_Net_new(2,3,128).to(device)

mode = 'single'
mode = 'qplib_8938'
mode = 'cont'

pareto = False
pareto = True



config = getConfig()
max_k = int(config['max_k'])
nlayer = int(config['nlayer'])
lr1 = float(config['lr'])
net_width = int(config['net_width'])
model_mode = int(config['model_mode'])
mode = config['mode']
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

if model_mode == 0:
    m = PDQP_Net_shared(1,1,net_width,max_k = max_k, threshold = 1e-8,nlayer=nlayer,type='linf').to(device)
elif model_mode == 1:
    m = PDQP_Net_AR(1,1,net_width,max_k = max_k, threshold = 1e-8,nlayer=nlayer,type='linf',use_dual=use_dual).to(device)
    ident += '_AR'

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

modf = relKKT().to(device)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(m.parameters(), lr=lr1)
max_epoch = 1000000
best_loss = 1e+20
flog = open('../logs/training_log.log','w')
last_epoch=0
if os.path.exists(f"../model/best_pdqp{ident}.mdl") and Contu:
    checkpoint = torch.load(f"../model/best_pdqp{ident}.mdl")
    m.load_state_dict(checkpoint['model'])
    if 'nepoch' in checkpoint:
        last_epoch=checkpoint['nepoch']
    best_loss=checkpoint['best_loss']
    print(f'Last best val loss gen:  {best_loss}')
    print('Model Loaded')

for epoch in range(max_epoch):
    avg_train_loss = 0.0
    random.shuffle(train_files)
    with alive_bar(len(train_files),title=f"Training epoch {epoch}") as bar:
        for fnm in train_files:
            mems = torch.cuda.memory_allocated()
            f_tar = gzip.open(f'{train_tar_dir}/{fnm}','rb')
            to_pack = pickle.load(f_tar)
            v_feat = to_pack['vf'].to(device)
            c_feat = to_pack['cf'].to(device)
            Q = to_pack['Q'].to(device)
            A = to_pack['A'].to(device)
            AT = torch.transpose(A,0,1)
            c = to_pack['c'].to(device)
            b = to_pack['b'].to(device)
            c = torch.unsqueeze(c,-1)
            b = torch.unsqueeze(b,-1)
            # AT = torch.transpose(A,0,1)
            x = to_pack['x'].to(device)
            y = to_pack['y'].to(device)
            vscale = torch.as_tensor(to_pack['vscale']).to(device).unsqueeze(-1)
            cscale = torch.as_tensor(to_pack['cscale'] ).to(device).unsqueeze(-1)
            constscale = torch.as_tensor(to_pack['constscale']).to(device).unsqueeze(-1)
            cons_ident = torch.as_tensor(to_pack['cons_ident'], dtype=torch.float32).to(device)
            vars_ident_l = torch.as_tensor(to_pack['vars_ident_l'], dtype=torch.float32).to(device)
            vars_ident_u = torch.as_tensor(to_pack['vars_ident_u'], dtype=torch.float32).to(device)
            var_lb = torch.as_tensor(to_pack['var_lb'], dtype=torch.float32).to(device)
            var_ub = torch.as_tensor(to_pack['var_ub'], dtype=torch.float32).to(device)
            f_tar.close()
            
            if cons_ident.shape[-1]!=1:
                cons_ident = cons_ident.unsqueeze(-1)
            if vars_ident_l.shape[-1]!=1:
                vars_ident_l = vars_ident_l.unsqueeze(-1)
            if vars_ident_u.shape[-1]!=1:
                vars_ident_u = vars_ident_u.unsqueeze(-1)
            if var_lb.shape[-1]!=1:
                var_lb = var_lb.unsqueeze(-1)
            if var_ub.shape[-1]!=1:
                var_ub = var_ub.unsqueeze(-1)

            # print(Q.shape)
            # print(A.shape)
            # print(c.shape)
            # print(b.shape)
            # print(x.shape)
            # print(y.shape)
            # quit()
            
            # in this version, use all 0 start
            v_feat = torch.zeros((v_feat.shape[0],1),dtype=torch.float32).to(device)
            c_feat = torch.zeros((c_feat.shape[0],1),dtype=torch.float32).to(device)

            optimizer.zero_grad()
            # x_pred,y_pred = m(A,Q,b,c,v_feat,c_feat,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub)
            x_pred,y_pred,scs_all,mult = m(AT,A,Q,b,c,v_feat,c_feat,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub)


            # pareto front
            if pareto:
                relkkt1 = scs_all[1]
                relkkt2 = scs_all[2]
                relkkt3 = scs_all[3]
                optimizer.zero_grad()
                relkkt1.backward(retain_graph=True)
                grd1 = compute_weight_grad(m)
                mag_1 = torch.norm(grd1)
                optimizer.zero_grad()
                relkkt2.backward(retain_graph=True)
                grd2 = compute_weight_grad(m)
                mag_2 = torch.norm(grd2)
                optimizer.zero_grad()
                relkkt3.backward(retain_graph=True)
                grd3 = compute_weight_grad(m)
                mag_3 = torch.norm(grd3)
                optimizer.zero_grad()

                relkkt1 = relkkt1 / mag_1
                relkkt2 = relkkt2 / mag_2
                relkkt3 = relkkt3 / mag_3

                loss = relkkt1+relkkt2+relkkt3


            if type(scs_all) == type((1,2)):
                scs = scs_all[0]
            else:
                scs = scs_all

            otm = scs.item()

            if not pareto:
                loss = scs






            # for sc in sc_hist:
            #     if loss is None:
            #         loss = sc
            #     else:
            #         loss +=sc
            avg_train_loss += scs.item()
            
            if mult>0.5 and False:
                x_scaled = x
                y_scaled = y
                # xm = x.max()
                # ym = y.max()
                # if xm!=0.0:
                #     x_scaled = x_scaled/xm
                #     x_pred = x_pred/xm
                # if ym!=0.0:
                #     y_scaled = y_scaled/ym
                #     y_pred = y_pred/ym

                loss1 = loss_func(x_pred, x_scaled)
                loss2 = loss_func(y_pred, y_scaled)
                # print(loss1,loss2)
                # print(loss)
                loss += loss1+loss2
                # print(loss)
                # print(fnm,loss1.item(),loss2.item(),xm,ym)


                
                if loss1.item()/x_pred.shape[0]+loss2.item()/y_pred.shape[0]>1e+12:
                    continue
                avg_train_loss += loss1.item()/x_pred.shape[0]
                avg_train_loss += loss2.item()/y_pred.shape[0]
            print(fnm,scs.item(),':::::',scs_all[1].item(),scs_all[2].item(),scs_all[3].item())
            # quit()

            if choose_weight:
                print(x_pred)
                print(x)
                print(y_pred)
                print(y)
                # print(b)
                # print(torch.norm(A,2))
                # print(cons_ident)
                input('wait')

            loss.backward()
            
            # try:
            #     print(compute_weight_grad(m))
            #     # input('f8i')
            # except:
            #     print('no grad')

            optimizer.step()
            bar()
    avg_train_loss = avg_train_loss / len(train_files)

    # print(x)
    # print(x_pred)
    # print(y)
    # print(y_pred)
    # input()

    # quit()
    avg_valid_loss = 0.0
    avg_sc = 0.0
    with torch.no_grad():
        with alive_bar(len(valid_files),title=f"Validating epoch {epoch}") as bar:
            for fnm in valid_files:
                f_tar = gzip.open(f'{valid_tar_dir}/{fnm}','rb')
                to_pack = pickle.load(f_tar)
                v_feat = to_pack['vf'].to(device)
                c_feat = to_pack['cf'].to(device)
                Q = to_pack['Q'].to(device)
                A = to_pack['A'].to(device)
                AT = torch.transpose(A,0,1)
                c = to_pack['c'].to(device)
                b = to_pack['b'].to(device)
                c = torch.unsqueeze(c,-1)
                b = torch.unsqueeze(b,-1)
                # AT = torch.transpose(A,0,1)
                x = to_pack['x'].to(device)
                y = to_pack['y'].to(device)
                vscale = torch.as_tensor(to_pack['vscale']).to(device).unsqueeze(-1)
                cscale = torch.as_tensor(to_pack['cscale'] ).to(device).unsqueeze(-1)
                constscale = torch.as_tensor(to_pack['constscale']).to(device).unsqueeze(-1)
                cons_ident = torch.as_tensor(to_pack['cons_ident'], dtype=torch.float32).to(device)
                vars_ident_l = torch.as_tensor(to_pack['vars_ident_l'], dtype=torch.float32).to(device)
                vars_ident_u = torch.as_tensor(to_pack['vars_ident_u'], dtype=torch.float32).to(device)
                var_lb = torch.as_tensor(to_pack['var_lb'], dtype=torch.float32).to(device)
                var_ub = torch.as_tensor(to_pack['var_ub'], dtype=torch.float32).to(device)
                f_tar.close()
                
                if cons_ident.shape[-1]!=1:
                    cons_ident = cons_ident.unsqueeze(-1)
                if vars_ident_l.shape[-1]!=1:
                    vars_ident_l = vars_ident_l.unsqueeze(-1)
                if vars_ident_u.shape[-1]!=1:
                    vars_ident_u = vars_ident_u.unsqueeze(-1)
                if var_lb.shape[-1]!=1:
                    var_lb = var_lb.unsqueeze(-1)
                if var_ub.shape[-1]!=1:
                    var_ub = var_ub.unsqueeze(-1)
                    
                    
                v_feat = torch.zeros((v_feat.shape[0],1),dtype=torch.float32).to(device)
                c_feat = torch.zeros((c_feat.shape[0],1),dtype=torch.float32).to(device)



                x_pred,y_pred,scs_all,mult = m(AT,A,Q,b,c,v_feat,c_feat,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub)

                if type(scs_all) == type((1,2)):
                    scs = scs_all[0]
                else:
                    scs = scs_all

                bqual = b.squeeze(-1).to(device)
                cqual = c.squeeze(-1).to(device)
                ttloss, prim_res, dual_res, gaps = modf(Q,A,AT,bqual,cqual,x_pred,y_pred,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub)
                print(f'primal_res: {prim_res.item()}   dual_res: {dual_res.item()}   gaps: {gaps.item()}')
                real_sc = torch.max(prim_res,torch.max(dual_res,gaps))
                avg_sc += real_sc.item()

                loss = scs
                avg_valid_loss += scs.item()
                # for sc in sc_hist:
                #     if loss is None:
                #         loss = sc
                #     else:
                #         loss +=sc
                if mult>0.5 and False:
                    x_scaled = x
                    y_scaled = y
                    # xm = x.max()
                    # ym = y.max()
                    # if xm!=0.0:
                    #     x_scaled = x_scaled/xm
                    #     x_pred = x_pred/xm
                    # if ym!=0.0:
                    #     y_scaled = y_scaled/ym
                    #     y_pred = y_pred/ym

                    loss1 = loss_func(x_pred, x_scaled)
                    loss2 = loss_func(y_pred, y_scaled)

                    loss += loss1+loss2
                        
                    if loss1.item()/x_pred.shape[0]+loss2.item()/y_pred.shape[0]>1e+12:
                        continue
                    avg_valid_loss += loss1.item()/x_pred.shape[0]
                    avg_valid_loss += loss2.item()/y_pred.shape[0]
                bar()
    avg_valid_loss = avg_valid_loss / len(valid_files)
    avg_sc = avg_sc/len(valid_files)

    
    st = f'epoch{epoch}: train: {avg_train_loss} | valid: {avg_valid_loss}\n'
    flog.write(st)
    flog.flush()
    print(f'Epoch{epoch}: train loss:{avg_train_loss}    valid loss:{avg_valid_loss}')

    # if best_loss > avg_valid_loss:
    #     best_loss = avg_valid_loss
    #     state={'model':m.state_dict(),'optimizer':optimizer.state_dict(),'best_loss':best_loss,'nepoch':epoch}
    #     torch.save(state,f'../model/best_pdqp{ident}.mdl')
    #     print(f'Saving new best model with valid loss: {best_loss}')
    #     st = f'     Saving new best model with valid loss: {best_loss}\n'
    #     flog.write(st)
    #     flog.flush()
    if best_loss > avg_sc:
        best_loss = avg_sc
        state={'model':m.state_dict(),'optimizer':optimizer.state_dict(),'best_loss':avg_sc,'nepoch':epoch}
        torch.save(state,f'../model/best_pdqp{ident}.mdl')
        print(f'Saving new best model with valid loss: {avg_sc}')
        st = f'     Saving new best model with valid loss: {avg_sc}\n'
        flog.write(st)
        flog.flush()




flog.close()