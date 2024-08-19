from model import *
from helper import *
import pickle
import gzip
import os
from alive_progress import alive_bar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
config = getConfig()
max_k = int(config['max_k'])
nlayer = int(config['nlayer'])
net_width = int(config['net_width'])
model_mode = int(config['model_mode'])
mode = config['mode']
use_dual = True
if int(config['usedual']) == 0:
    use_dual = False
# m = PDQP_Net_shared(1,1,128,max_k = max_k, threshold = 1e-8,nlayer=16).to(device)
# m = PDQP_Net_AR(1,1,128,max_k = max_k, threshold = 1e-8,nlayer=8).to(device)
m = None

ident = f'k{max_k}_{nlayer}'

if model_mode == 0:
    m = PDQP_Net_shared(1,1,net_width,max_k = max_k, threshold = 1e-8,nlayer=nlayer,type='linf').to(device)
elif model_mode == 1:
    m = PDQP_Net_AR(1,1,net_width,max_k = max_k, threshold = 1e-8,nlayer=nlayer,type='linf',use_dual=use_dual).to(device)
    ident += '_AR'

modf = relKKT()

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

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr1)
    max_epoch = 10000
    best_loss = 1e+20
    flog = open('../logs/training_log.log','w')

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

            # x_pred,y_pred = m(A,Q,b,c,v_feat,c_feat,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub)
            x_pred,y_pred,scs,mult = m(AT,A,Q,b,c,v_feat,c_feat,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub)

            # for idx in range(x_pred.shape[0]):
            #     print(x_pred[idx], x[idx])
            # for idx in range(y_pred.shape[0]):
            #     print(y_pred[idx], y[idx])
            bqual = b.squeeze(-1)
            cqual = c.squeeze(-1)
            # quit()



            # x_pred = torch.div(x_pred,vscale)
            # x_pred = x_pred * constscale
            # y_pred = torch.div(y_pred,cscale)
            # y_pred = y_pred * constscale

            # x_pred = x_pred / constscale
            # x_pred = x_pred*vscale
            # y_pred = y_pred / constscale
            # y_pred = y_pred*cscale
            # x_pred = x_pred.to(torch.float32)
            # y_pred = y_pred.to(torch.float32)
            

            ttloss, prim_res, dual_res, gaps = modf(Q,A,AT,bqual,cqual,x_pred,y_pred,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub)
            print(fnm)
            print(f'primal_res: {prim_res.item()}   dual_res: {dual_res.item()}   gaps: {gaps.item()}')
            print(f'    l2 norm err: {torch.norm(x_pred-x,2)}         SC: {scs}\n\n')

            ff = open(f'../predictions/primal_{fnm}.sol','w')
            st=''
            for xv in x_pred:
                st = st+f'{xv.item()} '
            st = st + '\n'
            ff.write(st)
            ff.close()
            
            ff = open(f'../predictions/dual_{fnm}.sol','w')
            st=''
            for xv in y_pred:
                st = st+f'{xv.item()} '
            st = st + '\n'
            ff.write(st)
            ff.close()
            
            
            bar()
                
    with alive_bar(len(train_files),title=f"Training part") as bar:
        for fnm in train_files:
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

            # x_pred,y_pred = m(A,Q,b,c,v_feat,c_feat,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub)
            x_pred,y_pred,scs,mult = m(AT,A,Q,b,c,v_feat,c_feat,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub)
            # x_pred,y_pred = m(A,Q,b,c,v_feat,c_feat)
            # x_pred = torch.div(x_pred,vscale)
            # x_pred = x_pred * constscale
            # y_pred = torch.div(y_pred,cscale)
            # y_pred = y_pred * constscale
            x_pred = x_pred / constscale
            x_pred = x_pred*vscale
            y_pred = y_pred / constscale
            y_pred = y_pred*cscale
            
            ff = open(f'../predictions/primal_{fnm}.sol','w')
            st=''
            for xv in x_pred:
                st = st+f'{xv.item()} '
            st = st + '\n'
            ff.write(st)
            ff.close()
            
            ff = open(f'../predictions/dual_{fnm}.sol','w')
            st=''
            for xv in y_pred:
                st = st+f'{xv.item()} '
            st = st + '\n'
            ff.write(st)
            ff.close()
            
            
            bar()
                

        



    flog.close()