import gurobipy as gp
import numpy as np
import torch
import random
import pickle
import gzip
import os
from alive_progress import alive_bar
from model import compute_weight_grad
import matplotlib.pyplot as plt

def extract(fnm):
    # return params
    v_feat = None
    c_feat = None
    Q = None
    A = None
    c = None
    b = None

    model = gp.read(f"{fnm}")
    vs = model.getVars()
    cs = model.getConstrs()

    Q = torch.tensor(model.getQ().todense()).to_sparse()
    A = torch.tensor(model.getA().todense()).to_sparse()
    n = Q.shape[0]
    m = len(cs)


    v_feat = torch.zeros((n,2))
    c_feat = torch.zeros((m,3))
    c = torch.zeros((n,1))
    b = torch.zeros((m,1))


    for indx,v in enumerate(vs):
        c[indx] = v.Obj
        # v_feat[indx,0] = v.LB
        # if torch.isinf(v.LB):
        #     v_feat[indx,0] = 1e+8

        # v_feat[indx,1] = v.UB
        # if torch.isinf(v.UB):
        #     v_feat[indx,1] = 1e+8
        v_feat[indx,0] = v.LB
        if torch.isinf(v_feat[indx,0]):
            v_feat[indx,0] = 0.0
        else:
            v_feat[indx,0] = 1.0
        v_feat[indx,1] = v.UB
        if torch.isinf(v_feat[indx,1]):
            v_feat[indx,1] = 0.0
        else:
            v_feat[indx,1] = 1.0

    for indx,cons in enumerate(cs):
        b[indx] = cons.RHS
        ss = cons.Sense
        if ss=='<':
            c_feat[indx,0] = 1.0
        elif ss=='>':
            c_feat[indx,1] = 1.0
        elif ss=='=':
            c_feat[indx,2] = 1.0
        
    return v_feat, c_feat, Q, A, c, b



def normalize(A,Q,b,c):
    m = A.shape[0]
    n = A.shape[1]
    
    b1=b
    c1=c
    b1[b == 0] = 1 
    c1[c == 0] = 1 
    
    # print(b.shape)
    # print(c.shape)
    # quit()
    
    A2 = A/b[:,np.newaxis]
    A2 = A2/c
    
        
    # for i in range(m):
    #     # print(A[i,:],(b[i]))
    #     for j in range(n):
    #         if b[i]!=0:
    #             A[i,j] = A[i,j]/(b[i])
    #         if c[j]!=0:
    #             A[i,j] = A[i,j]/(abs(c[j]))
    #     # print(A[i,:],(b[i]))
    #     # input()
        
    # print(A)
    A = A2
    # print(np.max(A))
    # quit()
        
    amax = np.max(A)
    
    A = A/amax
    print(amax)
    # print(np.max(A))
    # print(np.min(A))
    
    
    # for i in range(n):
    #     for j in range(n):
    #         Q[i,j] = Q[i,j]/(c[i]*c[j])
            
    scaling_matrix = np.outer(c1, c1)
    Q /= scaling_matrix
    
    for j in range(n):
        c[j] = c[j]/abs(c[j])
    b = np.ones(m)
    return A,Q,b,c,amax
        


def extract_sol(fnm):
    # return params
    v_feat = None
    c_feat = None
    Q = None
    A = None
    c = None
    b = None

    model = gp.read(f"{fnm}")
    model.optimize()
    model.Params.QCPDual = 1
    vs = model.getVars()
    cs = model.getConstrs()

    A = model.getA().todense()
    Q = model.getQ().todense()
    n = A.shape[1]
    m = A.shape[0]
    Q = torch.tensor(Q).to_sparse()
    A = torch.tensor(A).to_sparse()


    v_feat = torch.zeros((n,2))
    c_feat = torch.zeros((m,3))
    c = torch.zeros((n,1))
    b = torch.zeros((m,1))
    x = torch.zeros((n,1))
    y = torch.zeros((m,1))


    for indx,v in enumerate(vs):
        c[indx] = v.Obj
        # v_feat[indx,0] = v.LB
        # if torch.isinf(v.LB):
        #     v_feat[indx,0] = 1e+8

        # v_feat[indx,1] = v.UB
        # if torch.isinf(v.UB):
        #     v_feat[indx,1] = 1e+8
        v_feat[indx,0] = v.LB
        if torch.isinf(v_feat[indx,0]):
            v_feat[indx,0] = 0.0
        else:
            v_feat[indx,0] = 1.0
        v_feat[indx,1] = v.UB
        if torch.isinf(v_feat[indx,1]):
            v_feat[indx,1] = 0.0
        else:
            v_feat[indx,1] = 1.0
        x[indx] = v.X

    for indx,cons in enumerate(cs):
        b[indx] = cons.RHS
        ss = cons.Sense
        if ss=='<':
            c_feat[indx,0] = 1.0
        elif ss=='>':
            c_feat[indx,1] = 1.0
        elif ss=='=':
            c_feat[indx,2] = 1.0
        y[indx] = cons.Pi



        
    return v_feat, c_feat, Q, A, c, b, x, y


def extract_solfile(fnm):
    # return params
    v_feat = None
    c_feat = None
    Q = None
    A = None
    c = None
    b = None

    model = gp.read(f"{fnm}")
    vs = model.getVars()
    cs = model.getConstrs()


    # Q = torch.tensor(model.getQ().todense()).to_sparse()
    # A = torch.tensor(model.getA().todense()).to_sparse()
    
    A = model.getA().todense()
    Q = model.getQ().todense()
    n = A.shape[1]
    m = A.shape[0]
    # Q = torch.tensor(Q)
    # A = torch.tensor(A)
    print(A.shape)


    # lb ub haslb hasub
    v_feat = torch.zeros((n,2))
    # 
    c_feat = torch.zeros((m,3))
    c = np.zeros((n,))
    b = np.zeros((m,))
    x = torch.zeros((n,1))
    y = torch.zeros((m,1))

    log_fnm = fnm.split('.')
    log_fnm = [x for x in log_fnm if x!=''][0]
    log_fnm = log_fnm.split('/')[-1]
    xsol_file = f'../logs/{log_fnm}_primal.txt'
    ysol_file = f'../logs/{log_fnm}_dual.txt'

    ff = open(xsol_file,'r')
    ll = 0
    for line in ff:
        x[ll] = float(line)
        ll+=1
    ff.close()
    
    ff = open(ysol_file,'r')
    ll = 0
    for line in ff:
        y[ll] = float(line)
        ll+=1
    ff.close()



    for indx,v in enumerate(vs):
        c[indx] = v.Obj
        # v_feat[indx,0] = v.LB
        # if torch.isinf(v.LB):
        #     v_feat[indx,0] = 1e+8

        # v_feat[indx,1] = v.UB
        # if torch.isinf(v.UB):
        #     v_feat[indx,1] = 1e+8
        v_feat[indx,0] = v.LB
        if torch.isinf(v_feat[indx,0]):
            v_feat[indx,0] = 0.0
        else:
            v_feat[indx,0] = 1.0
        v_feat[indx,1] = v.UB
        if torch.isinf(v_feat[indx,1]):
            v_feat[indx,1] = 0.0
        else:
            v_feat[indx,1] = 1.0
        # print(v.LB,v.UB)
        # print(v_feat[indx,0],v_feat[indx,1])
        # quit()
    # v_feat[v_feat == float("Inf")] = 0.0
    # v_feat[v_feat == -float("Inf")] = 0.0

    for indx,cons in enumerate(cs):
        b[indx] = cons.RHS
        ss = cons.Sense
        
        # for normalization
        if b[indx] < 0:
            if ss=='>':
                ss='<'
            elif ss=='<':
                ss='>'
                
        if ss=='<':
            c_feat[indx,0] = 1.0
        elif ss=='>':
            c_feat[indx,1] = 1.0
        elif ss=='=':
            c_feat[indx,2] = 1.0
            
    # A=[[1.0,2.0,3.0],[5.0,6.6,7.6]]
    # b=[1.5,2.0]
    # c=[3.0,4.0,5.0]
    # A = np.array(A)
    # print(A.shape)
    # b = np.array(b)
    # c = np.array(c)
    
    # A,Q,b,c,amax = normalize(A,Q,b,c)
    # for i in range(n):
    #     x[i] = x[i]*amax
    
    Q = torch.as_tensor(Q)
    A = torch.as_tensor(A)
    Q = Q.to_sparse()
    A = A.to_sparse()
    
    # print(x)
        
    return v_feat, c_feat, Q, A, c, b, x, y


def extract_solfile_scaled(fnm):
    # return params
    v_feat = None
    c_feat = None
    Q = None
    A = None
    c = None
    b = None
    vscale = None
    cscale = None
    constscale = None

    var_ub = None
    var_lb = None
    vars_ident_l = None
    vars_ident_u = None
    cons_ident = None

    fnm = fnm.replace('/ins','/transformed_ins')
    fnm = fnm.replace('/gen_train_cont','/train')
    tsp = fnm.split('/')
    fnm='/'.join([tsp[0],tsp[1],'train',tsp[3]])
    print(fnm,'!!!!!!!!!!')
    
    
    n=0
    m=0
    
    ff = open(fnm,'r')
    print('    File Opened!!!!!!!!!!')
    mode = 'N'
    counter = 0

    nnz = []
    for line in ff:
        line = line.replace('\n','')
        if line == 'Q':
            print(line,'SS')
            mode = 'Q'
            Q = np.zeros((n,n))
            continue
        elif line == 'A':
            print(line)
            mode = 'A'
            A = np.zeros((m,n))
            continue
        elif line == 'c':
            print(line)
            mode = 'c'
            c = np.zeros((n,))
            counter = 0
            continue
        elif line == 'b':
            print(line)
            mode = 'b'
            b = np.zeros((m,))
            counter = 0
            continue
        elif line == 'vscale':
            print(line)
            mode = 'vscale'
            vscale = np.zeros((n,))
            counter = 0
            continue
        elif line == 'cscale':
            print(line)
            mode = 'cscale'
            cscale = np.zeros((m,))
            counter = 0
            continue
        elif line == 'constscale':
            print(line)
            mode = 'constscale'
            constscale = np.zeros((1,))
            counter = 0
            continue
        elif line == 'l':
            print(line)
            mode = 'l'
            var_lb = np.zeros((n,))
            counter = 0
            continue
        elif line == 'u':
            print(line)
            mode = 'u'
            var_ub = np.zeros((n,))
            counter = 0
            continue
        elif line == 'numEquation':
            print(line)
            mode = 'numEquation'
            cons_ident = np.zeros((m,))
            continue
        
        if mode =='N':
            line = line.split(' ')
            m = int(line[0])
            n = int(line[1])
            vars_ident_l = np.zeros((n,))
            vars_ident_u = np.zeros((n,))
        elif mode == 'Q':
            line = line.split(' ')
            yind = int(line[0])-1
            xind = int(line[1])-1
            val = float(line[2])
            Q[xind,yind] = val
        elif mode == 'A':
            line = line.split(' ')
            yind = int(line[0])-1
            xind = int(line[1])-1
            val = float(line[2])
            # A[xind,yind] = val
            nnz.append((xind,yind,val))
        elif mode == 'c':
            line = line.split(' ')[0]
            line = float(line)
            c[counter] = line
            counter+=1
        elif mode == 'b':
            line = line.split(' ')[0]
            line = float(line)
            b[counter] = line
            counter+=1
        elif mode == 'vscale':
            line = line.split(' ')[0]
            line = float(line)
            vscale[counter] = line
            counter+=1
        elif mode == 'cscale':
            line = line.split(' ')[0]
            line = float(line)
            cscale[counter] = line
            counter+=1
        elif mode == 'constscale':
            line = line.split(' ')[0]
            line = float(line)
            constscale[counter] = line
            counter+=1
        elif mode == 'l':
            line = line.split(' ')[0]
            if 'Inf' in line:
                counter+=1
                continue
            line = float(line)
            var_lb[counter] = line
            vars_ident_l[counter] = 1.0
            counter+=1
        elif mode == 'u':
            line = line.split(' ')[0]
            if 'Inf' in line:
                counter+=1
                continue
            line = float(line)
            var_ub[counter] = line
            vars_ident_u[counter] = 1.0
            counter+=1
        elif mode == 'numEquation':
            line = line.split(' ')[0]
            nneq = int(line)
            for i in range(nneq,m):
                cons_ident[i] = 1.0

                     
    ff.close()    
    
    print('!!!!!!!!!!!!!!!!!!!!')
    # for i in range(A.shape[0]):
    #     b[i] = -b[i]
    #     if cons_ident[i] > 0.5:
    #         for j in range(A.shape[1]):
    #             A[i,j] = -A[i,j]
    cons_ident_set = set()
    for i in range(A.shape[0]):
        if cons_ident[i] > 0.5:
            cons_ident_set.add(i)
    for ele in nnz:
        xind,yind,val = ele
        if xind in cons_ident_set:
            # A[xind,yind] = -val
            A[xind,yind] = val
        else:
            A[xind,yind] = val
    # for i in range(A.shape[0]):
    for i in cons_ident_set:
        # b[i] = -b[i]
        b[i] = b[i]
    # print(A.shape)
    # for ent in A[:,0]:
    #     if ent!=0.0:
    #         print(ent)
    # quit()
    #     if cons_ident[i] > 0.5:
    #         for j in range(A.shape[1]):
    #             A[i,j] = -A[i,j]
    
    v_feat = torch.zeros((n,2))
    c_feat = torch.zeros((m,3))
    x = torch.zeros((n,1))
    y = torch.zeros((m,1))

    log_fnm = fnm.split('.')
    log_fnm = [x for x in log_fnm if x!=''][0]
    log_fnm = log_fnm.split('/')[-1]
    xsol_file = f'../logs/{log_fnm}_primal_scaled.txt'
    ysol_file = f'../logs/{log_fnm}_dual_scaled.txt'

    if os.path.isfile(xsol_file):
        ff = open(xsol_file,'r')
        ll = 0
        for line in ff:
            x[ll] = float(line)
            ll+=1
        ff.close()
    
    if os.path.isfile(ysol_file):
        ff = open(ysol_file,'r')
        ll = 0
        for line in ff:
            y[ll] = float(line)
            ll+=1
        ff.close()


    Q = torch.as_tensor(Q)
    A = torch.as_tensor(A)
    Q = Q.to_sparse()
    A = A.to_sparse()
    
    # print(x)
        
    return v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident




def extract_solfile_scaled_sparse(fnm):
    # return params
    v_feat = None
    c_feat = None
    Q = None
    A = None
    c = None
    b = None
    vscale = None
    cscale = None
    constscale = None

    var_ub = None
    var_lb = None
    vars_ident_l = None
    vars_ident_u = None
    cons_ident = None

    fnm = fnm.replace('/ins','/transformed_ins')
    fnm = fnm.replace('/gen_train_cont','/train')
    tsp = fnm.split('/')
    fnm='/'.join([tsp[0],tsp[1],'train',tsp[3]])
    print(fnm,'!!!!!!!!!!')
    
    
    n=0
    m=0
    
    ff = open(fnm,'r')
    print('    File Opened!!!!!!!!!!')
    mode = 'N'
    counter = 0

    nnz = []
    Qind = [[],[]]
    Qval = []
    Aind = [[],[]]
    Aval = []
    for line in ff:
        line = line.replace('\n','')
        if line == 'Q':
            print(line,'SS')
            mode = 'Q'
            continue
        elif line == 'A':
            print(line)
            mode = 'A'
            continue
        elif line == 'c':
            print(line)
            mode = 'c'
            c = np.zeros((n,))
            counter = 0
            continue
        elif line == 'b':
            print(line)
            mode = 'b'
            b = np.zeros((m,))
            counter = 0
            continue
        elif line == 'vscale':
            print(line)
            mode = 'vscale'
            vscale = np.zeros((n,))
            counter = 0
            continue
        elif line == 'cscale':
            print(line)
            mode = 'cscale'
            cscale = np.zeros((m,))
            counter = 0
            continue
        elif line == 'constscale':
            print(line)
            mode = 'constscale'
            constscale = np.zeros((1,))
            counter = 0
            continue
        elif line == 'l':
            print(line)
            mode = 'l'
            var_lb = np.zeros((n,))
            counter = 0
            continue
        elif line == 'u':
            print(line)
            mode = 'u'
            var_ub = np.zeros((n,))
            counter = 0
            continue
        elif line == 'numEquation':
            print(line)
            mode = 'numEquation'
            cons_ident = np.zeros((m,))
            continue
        
        if mode =='N':
            line = line.split(' ')
            m = int(line[0])
            n = int(line[1])
            vars_ident_l = np.zeros((n,))
            vars_ident_u = np.zeros((n,))
        elif mode == 'Q':
            line = line.split(' ')
            yind = int(line[0])-1
            xind = int(line[1])-1
            val = float(line[2])
            Qind[0].append(xind)
            Qind[1].append(yind)
            Qval.append(val)
        elif mode == 'A':
            line = line.split(' ')
            yind = int(line[0])-1
            xind = int(line[1])-1
            val = float(line[2])
            # A[xind,yind] = val
            # nnz.append((xind,yind,val))
            Aind[0].append(xind)
            Aind[1].append(yind)
            Aval.append(val)
        elif mode == 'c':
            line = line.split(' ')[0]
            line = float(line)
            c[counter] = line
            counter+=1
        elif mode == 'b':
            line = line.split(' ')[0]
            line = float(line)
            b[counter] = line
            counter+=1
        elif mode == 'vscale':
            line = line.split(' ')[0]
            line = float(line)
            vscale[counter] = line
            counter+=1
        elif mode == 'cscale':
            line = line.split(' ')[0]
            line = float(line)
            cscale[counter] = line
            counter+=1
        elif mode == 'constscale':
            line = line.split(' ')[0]
            line = float(line)
            constscale[counter] = line
            counter+=1
        elif mode == 'l':
            line = line.split(' ')[0]
            if 'Inf' in line:
                counter+=1
                continue
            line = float(line)
            var_lb[counter] = line
            vars_ident_l[counter] = 1.0
            counter+=1
        elif mode == 'u':
            line = line.split(' ')[0]
            if 'Inf' in line:
                counter+=1
                continue
            line = float(line)
            var_ub[counter] = line
            vars_ident_u[counter] = 1.0
            counter+=1
        elif mode == 'numEquation':
            line = line.split(' ')[0]
            nneq = int(line)
            for i in range(nneq,m):
                cons_ident[i] = 1.0

                     
    ff.close()    
    
    print('!!!!!!!!!!!!!!!!!!!!')
    # for i in range(A.shape[0]):
    #     b[i] = -b[i]
    #     if cons_ident[i] > 0.5:
    #         for j in range(A.shape[1]):
    #             A[i,j] = -A[i,j]
    cons_ident_set = set()
    for i in range(m):
        if cons_ident[i] > 0.5:
            cons_ident_set.add(i)
    A = torch.sparse_coo_tensor(Aind,Aval,[m,n])
    Q = torch.sparse_coo_tensor(Qind,Qval,[n,n])
    # for i in range(A.shape[0]):
    # for i in cons_ident_set:
        # b[i] = -b[i]
        # b[i] = b[i]
    # print(A.shape)
    # for ent in A[:,0]:
    #     if ent!=0.0:
    #         print(ent)
    # quit()
    #     if cons_ident[i] > 0.5:
    #         for j in range(A.shape[1]):
    #             A[i,j] = -A[i,j]
    
    v_feat = torch.zeros((n,2))
    c_feat = torch.zeros((m,3))
    x = torch.zeros((n,1))
    y = torch.zeros((m,1))

    log_fnm = fnm.split('.')
    log_fnm = [x for x in log_fnm if x!=''][0]
    log_fnm = log_fnm.split('/')[-1]
    xsol_file = f'../logs/{log_fnm}_primal_scaled.txt'
    ysol_file = f'../logs/{log_fnm}_dual_scaled.txt'

    if os.path.isfile(xsol_file):
        ff = open(xsol_file,'r')
        ll = 0
        for line in ff:
            x[ll] = float(line)
            ll+=1
        ff.close()
    
    if os.path.isfile(ysol_file):
        ff = open(ysol_file,'r')
        ll = 0
        for line in ff:
            y[ll] = float(line)
            ll+=1
        ff.close()


    return v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident






def extract_solfile_unscaled(fnm):
    # return params
    v_feat = None
    c_feat = None
    Q = None
    A = None
    c = None
    b = None
    vscale = None
    cscale = None
    constscale = None

    var_ub = None
    var_lb = None
    vars_ident_l = None
    vars_ident_u = None
    cons_ident = None

    print(fnm,'???????????')
    fnm = fnm.replace('/ins','/ori_ins')
    fnm = fnm.replace('/gen_train_cont','/train')
    tsp = fnm.split('/')
    fnm='/'.join([tsp[0],tsp[1],'train',tsp[3]])
    print(fnm,'!!!!!!!!!!')
    
    
    n=0
    m=0
    
    print(fnm)
    ff = open(fnm,'r')
    print('    File Opened!!!!!!!!!!')
    mode = 'N'
    counter = 0

    nnz = []
    for line in ff:
        line = line.replace('\n','')
        if line == 'Q':
            print(line)
            mode = 'Q'
            Q = np.zeros((n,n))
            continue
        elif line == 'A':
            print(line)
            mode = 'A'
            A = np.zeros((m,n))
            continue
        elif line == 'c':
            print(line)
            mode = 'c'
            c = np.zeros((n,))
            counter = 0
            continue
        elif line == 'b':
            print(line)
            mode = 'b'
            b = np.zeros((m,))
            counter = 0
            continue
        elif line == 'vscale':
            print(line)
            mode = 'vscale'
            vscale = np.zeros((n,))
            counter = 0
            continue
        elif line == 'cscale':
            print(line)
            mode = 'cscale'
            cscale = np.zeros((m,))
            counter = 0
            continue
        elif line == 'constscale':
            print(line)
            mode = 'constscale'
            constscale = np.zeros((1,))
            counter = 0
            continue
        elif line == 'l':
            print(line)
            mode = 'l'
            var_lb = np.zeros((n,))
            counter = 0
            continue
        elif line == 'u':
            print(line)
            mode = 'u'
            var_ub = np.zeros((n,))
            counter = 0
            continue
        elif line == 'numEquation':
            print(line)
            mode = 'numEquation'
            cons_ident = np.zeros((m,))
            continue
        
        if mode =='N':
            line = line.split(' ')
            m = int(line[0])
            n = int(line[1])
            vars_ident_l = np.zeros((n,))
            vars_ident_u = np.zeros((n,))
            print(n,m)
        elif mode == 'Q':
            line = line.split(' ')
            yind = int(line[0])-1
            xind = int(line[1])-1
            val = float(line[2])
            Q[xind,yind] = val
        elif mode == 'A':
            line = line.split(' ')
            yind = int(line[0])-1
            xind = int(line[1])-1
            val = float(line[2])
            # A[xind,yind] = val
            nnz.append((xind,yind,val))
        elif mode == 'c':
            line = line.split(' ')[0]
            line = float(line)
            c[counter] = line
            counter+=1
        elif mode == 'b':
            line = line.split(' ')[0]
            line = float(line)
            b[counter] = line
            counter+=1
        elif mode == 'vscale':
            line = line.split(' ')[0]
            line = float(line)
            vscale[counter] = line
            counter+=1
        elif mode == 'cscale':
            line = line.split(' ')[0]
            line = float(line)
            cscale[counter] = line
            counter+=1
        elif mode == 'constscale':
            line = line.split(' ')[0]
            line = float(line)
            constscale[counter] = line
            counter+=1
        elif mode == 'l':
            line = line.split(' ')[0]
            if 'Inf' in line:
                counter+=1
                continue
            line = float(line)
            var_lb[counter] = line
            vars_ident_l[counter] = 1.0
            counter+=1
        elif mode == 'u':
            line = line.split(' ')[0]
            if 'Inf' in line:
                counter+=1
                continue
            line = float(line)
            var_ub[counter] = line
            vars_ident_u[counter] = 1.0
            counter+=1
        elif mode == 'numEquation':
            line = line.split(' ')[0]
            nneq = int(line)
            for i in range(nneq,m):
                cons_ident[i] = 1.0

                     
    ff.close()    
    
    print('!!!!!!!!!!!!!!!!!!!!')
    # for i in range(A.shape[0]):
    #     b[i] = -b[i]
    #     if cons_ident[i] > 0.5:
    #         for j in range(A.shape[1]):
    #             A[i,j] = -A[i,j]
    cons_ident_set = set()
    for i in range(A.shape[0]):
        if cons_ident[i] > 0.5:
            cons_ident_set.add(i)
    for ele in nnz:
        xind,yind,val = ele
        if xind in cons_ident_set:
            # A[xind,yind] = -val
            A[xind,yind] = val
        else:
            A[xind,yind] = val
    # for i in range(A.shape[0]):
    for i in cons_ident_set:
        # b[i] = -b[i]
        b[i] = b[i]
    # print(A.shape)
    # for ent in A[:,0]:
    #     if ent!=0.0:
    #         print(ent)
    # quit()
    #     if cons_ident[i] > 0.5:
    #         for j in range(A.shape[1]):
    #             A[i,j] = -A[i,j]
    
    v_feat = torch.zeros((n,2))
    c_feat = torch.zeros((m,3))
    x = torch.zeros((n,1))
    y = torch.zeros((m,1))

    log_fnm = fnm.split('.')
    log_fnm = [x for x in log_fnm if x!=''][0]
    log_fnm = log_fnm.split('/')[-1]
    xsol_file = f'../logs/{log_fnm}_primal.txt'
    ysol_file = f'../logs/{log_fnm}_dual.txt'

    if os.path.isfile(xsol_file):

        ff = open(xsol_file,'r')
        ll = 0
        for line in ff:
            x[ll] = float(line)
            ll+=1
        ff.close()
    
    if os.path.isfile(ysol_file):
        ff = open(ysol_file,'r')
        ll = 0
        for line in ff:
            y[ll] = float(line)
            ll+=1
        ff.close()


    Q = torch.as_tensor(Q)
    A = torch.as_tensor(A)
    Q = Q.to_sparse()
    A = A.to_sparse()
    
    # print(x)
        
    return v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident



def extract_solfile_unscaled_sparse(fnm):
    # return params
    v_feat = None
    c_feat = None
    Q = None
    A = None
    c = None
    b = None
    vscale = None
    cscale = None
    constscale = None

    var_ub = None
    var_lb = None
    vars_ident_l = None
    vars_ident_u = None
    cons_ident = None

    print(fnm,'???????????')
    fnm = fnm.replace('/ins','/ori_ins')
    fnm = fnm.replace('/gen_train_cont','/train')
    tsp = fnm.split('/')
    fnm='/'.join([tsp[0],tsp[1],'train',tsp[3]])
    print(fnm,'!!!!!!!!!!')
    
    n=0
    m=0
    
    ff = open(fnm,'r')
    print('    File Opened!!!!!!!!!!')
    mode = 'N'
    counter = 0

    nnz = []
    Qind = [[],[]]
    Qval = []
    Aind = [[],[]]
    Aval = []
    for line in ff:
        line = line.replace('\n','')
        if line == 'Q':
            print(line,'SS')
            mode = 'Q'
            continue
        elif line == 'A':
            print(line)
            mode = 'A'
            continue
        elif line == 'c':
            print(line)
            mode = 'c'
            c = np.zeros((n,))
            counter = 0
            continue
        elif line == 'b':
            print(line)
            mode = 'b'
            b = np.zeros((m,))
            counter = 0
            continue
        elif line == 'vscale':
            print(line)
            mode = 'vscale'
            vscale = np.zeros((n,))
            counter = 0
            continue
        elif line == 'cscale':
            print(line)
            mode = 'cscale'
            cscale = np.zeros((m,))
            counter = 0
            continue
        elif line == 'constscale':
            print(line)
            mode = 'constscale'
            constscale = np.zeros((1,))
            counter = 0
            continue
        elif line == 'l':
            print(line)
            mode = 'l'
            var_lb = np.zeros((n,))
            counter = 0
            continue
        elif line == 'u':
            print(line)
            mode = 'u'
            var_ub = np.zeros((n,))
            counter = 0
            continue
        elif line == 'numEquation':
            print(line)
            mode = 'numEquation'
            cons_ident = np.zeros((m,))
            continue
        
        if mode =='N':
            line = line.split(' ')
            m = int(line[0])
            n = int(line[1])
            vars_ident_l = np.zeros((n,))
            vars_ident_u = np.zeros((n,))
        elif mode == 'Q':
            line = line.split(' ')
            yind = int(line[0])-1
            xind = int(line[1])-1
            val = float(line[2])
            Qind[0].append(xind)
            Qind[1].append(yind)
            Qval.append(val)
        elif mode == 'A':
            line = line.split(' ')
            yind = int(line[0])-1
            xind = int(line[1])-1
            val = float(line[2])
            # A[xind,yind] = val
            # nnz.append((xind,yind,val))
            Aind[0].append(xind)
            Aind[1].append(yind)
            Aval.append(val)
        elif mode == 'c':
            line = line.split(' ')[0]
            line = float(line)
            c[counter] = line
            counter+=1
        elif mode == 'b':
            line = line.split(' ')[0]
            line = float(line)
            b[counter] = line
            counter+=1
        elif mode == 'vscale':
            line = line.split(' ')[0]
            line = float(line)
            vscale[counter] = line
            counter+=1
        elif mode == 'cscale':
            line = line.split(' ')[0]
            line = float(line)
            cscale[counter] = line
            counter+=1
        elif mode == 'constscale':
            line = line.split(' ')[0]
            line = float(line)
            constscale[counter] = line
            counter+=1
        elif mode == 'l':
            line = line.split(' ')[0]
            if 'Inf' in line:
                counter+=1
                continue
            line = float(line)
            var_lb[counter] = line
            vars_ident_l[counter] = 1.0
            counter+=1
        elif mode == 'u':
            line = line.split(' ')[0]
            if 'Inf' in line:
                counter+=1
                continue
            line = float(line)
            var_ub[counter] = line
            vars_ident_u[counter] = 1.0
            counter+=1
        elif mode == 'numEquation':
            line = line.split(' ')[0]
            nneq = int(line)
            for i in range(nneq,m):
                cons_ident[i] = 1.0

                     
    ff.close()    
    
    print('!!!!!!!!!!!!!!!!!!!!')
    # for i in range(A.shape[0]):
    #     b[i] = -b[i]
    #     if cons_ident[i] > 0.5:
    #         for j in range(A.shape[1]):
    #             A[i,j] = -A[i,j]
    cons_ident_set = set()
    for i in range(m):
        if cons_ident[i] > 0.5:
            cons_ident_set.add(i)
    A = torch.sparse_coo_tensor(Aind,Aval,[m,n])
    Q = torch.sparse_coo_tensor(Qind,Qval,[n,n])
    # for i in range(A.shape[0]):
    for i in cons_ident_set:
        # b[i] = -b[i]
        b[i] = b[i]
    # print(A.shape)
    # for ent in A[:,0]:
    #     if ent!=0.0:
    #         print(ent)
    # quit()
    #     if cons_ident[i] > 0.5:
    #         for j in range(A.shape[1]):
    #             A[i,j] = -A[i,j]
    
    v_feat = torch.zeros((n,2))
    c_feat = torch.zeros((m,3))
    x = torch.zeros((n,1))
    y = torch.zeros((m,1))

    log_fnm = fnm.split('.')
    log_fnm = [x for x in log_fnm if x!=''][0]
    log_fnm = log_fnm.split('/')[-1]
    xsol_file = f'../logs/{log_fnm}_primal.txt'
    ysol_file = f'../logs/{log_fnm}_dual.txt'

    if os.path.isfile(xsol_file):
        ff = open(xsol_file,'r')
        ll = 0
        for line in ff:
            x[ll] = float(line)
            ll+=1
        ff.close()
    
    if os.path.isfile(ysol_file):
        ff = open(ysol_file,'r')
        ll = 0
        for line in ff:
            y[ll] = float(line)
            ll+=1
        ff.close()


    return v_feat, c_feat, Q, A, c, b, x, y, vscale, cscale, constscale, var_lb, var_ub, vars_ident_l, vars_ident_u, cons_ident









# v_feat, c_feat, Q, A, c, b, x, y, xs,cs, consc = extract_solfile_scaled('/home/lxyang/git/pdqpnet/ins/train/QPCBOEI1.QPS')
# quit()

def getConfig(ident=''):
    f = open('setting')
    res_dic = {}
    reading = True
    if ident!='':
        reading=False
    for line in f:
        if ident!='':
            if ident in line:
                if  'end' in line:
                    break
                else:
                    reading = True
        if reading:
            if '=' not in line or line[0]=='#':
                continue
            line = line.replace('\n','').replace(' ','').split('=')
            res_dic[line[0]] = line[1]
    f.close()
    return res_dic


def process(m,files,epoch,tar_dir,pareto,device,optimizer,choose_weight=False,autoregression_iteration=1,training=True,accu_loss = True):
    if not training:
        return valid(m,files,epoch,tar_dir,pareto,device,optimizer,autoregression_iteration)
    else:
        return train(m,files,epoch,tar_dir,pareto,device,optimizer,choose_weight,autoregression_iteration,accu_loss = accu_loss)




def valid(m,valid_files,epoch,valid_tar_dir,pareto,device,modf,autoregression_iteration):

    avg_valid_loss = [0.0]*autoregression_iteration
    avg_sc = [0.0]*autoregression_iteration
    avg_scprimal = [0.0]*autoregression_iteration
    avg_scdual = [0.0]*autoregression_iteration
    avg_scgap = [0.0]*autoregression_iteration
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

                Q_ori = to_pack['Q_ori'].to(device)
                A_ori = to_pack['A_ori'].to(device)
                AT_ori = torch.transpose(A_ori,0,1)
                c_ori = to_pack['c_ori'].to(device)
                b_ori = to_pack['b_ori'].to(device)
                c_ori = torch.unsqueeze(c_ori,-1)
                b_ori = torch.unsqueeze(b_ori,-1)
                var_lb_ori = torch.as_tensor(to_pack['var_lb_ori'], dtype=torch.float32).to(device)
                var_ub_ori = torch.as_tensor(to_pack['var_ub_ori'], dtype=torch.float32).to(device)

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
                if var_lb_ori.shape[-1]!=1:
                    var_lb_ori = var_lb_ori.unsqueeze(-1)
                if var_ub_ori.shape[-1]!=1:
                    var_ub_ori = var_ub_ori.unsqueeze(-1)
                    
                    
                v_feat = torch.zeros((v_feat.shape[0],1),dtype=torch.float32).to(device)
                c_feat = torch.zeros((c_feat.shape[0],1),dtype=torch.float32).to(device)



                for itr in range(autoregression_iteration):
                    x_pred,y_pred,scs_all,mult = m(AT,A,Q,b,c,v_feat,c_feat,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub, 
                                                   AT_ori,A_ori,Q_ori,b_ori,c_ori,vscale,cscale,constscale,var_lb_ori,var_ub_ori)

                    if type(scs_all) == type((1,2)):
                        scs = scs_all[0]
                    else:
                        scs = scs_all

                    bqual_ori = b_ori.squeeze(-1).to(device)
                    cqual_ori = c_ori.squeeze(-1).to(device)
                    ttloss, prim_res, dual_res, gaps = modf(Q_ori,A_ori,AT_ori,bqual_ori,cqual_ori,x_pred,y_pred,cons_ident,vars_ident_l,vars_ident_u,var_lb_ori,var_ub_ori, vscale,cscale,constscale)
                    print(f'primal_res: {prim_res.item()}   dual_res: {dual_res.item()}   gaps: {gaps.item()}')
                    real_sc = torch.max(prim_res,torch.max(dual_res,gaps))
                    real_sc_num = real_sc.item()
                    avg_sc[itr] += real_sc_num

                    avg_scprimal[itr] += prim_res.item()
                    avg_scdual[itr] += dual_res.item()
                    avg_scgap[itr] += gaps.item()

                    loss = scs
                    # loss = (itr*loss)/autoregression_iteration
                    loss_num = scs.item()
                    avg_valid_loss[itr] += loss_num


                    v_feat = x_pred.detach().clone()
                    c_feat = y_pred.detach().clone()

                    

                    print(f'Auto-regression on {fnm}, iteration {itr} loss:{loss_num}     real sc:{real_sc_num}')
                bar()
                
    return avg_valid_loss, avg_sc, avg_scprimal, avg_scdual, avg_scgap


def compute_obj(Q,c,x,y):

    c = c.squeeze(-1)

    print(Q.shape,x.shape)
    r1 = torch.sparse.mm(Q,x).squeeze(-1)
    print(f'r1 shape: {r1.shape}')
    r1 = torch.matmul(r1,x.squeeze(-1))
    print(f'r1 shape: {r1.shape}')

    r2 = torch.matmul(c,x.squeeze(-1))
    print(r2.shape)

    print(r1+r2)

    print(x,torch.max(x))
    print(y,torch.max(y))


    quit()



def inference(m,fnm,epoch,valid_tar_dir,pareto,device,modf,autoregression_iteration):
    f_tar = gzip.open(f'{valid_tar_dir}/{fnm}','rb')
    to_pack = pickle.load(f_tar)
    v_feat = to_pack['vf'].to(device)
    c_feat = to_pack['cf'].to(device)
    Q = to_pack['Q'].to(device)
    A = to_pack['A'].to(device)
    AT = torch.transpose(A,0,1)
    c = to_pack['c'].to(device)
    b = to_pack['b'].to(device)
    x = to_pack['x'].to(device)
    y = to_pack['y'].to(device)
    c = torch.unsqueeze(c,-1)
    b = torch.unsqueeze(b,-1)

    Q_ori = to_pack['Q_ori'].to(device)
    A_ori = to_pack['A_ori'].to(device)
    AT_ori = torch.transpose(A_ori,0,1)
    c_ori = to_pack['c_ori'].to(device)
    b_ori = to_pack['b_ori'].to(device)
    c_ori = torch.unsqueeze(c_ori,-1)
    b_ori = torch.unsqueeze(b_ori,-1)
    var_lb_ori = torch.as_tensor(to_pack['var_lb_ori'], dtype=torch.float32).to(device)
    var_ub_ori = torch.as_tensor(to_pack['var_ub_ori'], dtype=torch.float32).to(device)

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
    if var_lb_ori.shape[-1]!=1:
        var_lb_ori = var_lb_ori.unsqueeze(-1)
    if var_ub_ori.shape[-1]!=1:
        var_ub_ori = var_ub_ori.unsqueeze(-1)
    # in this version, use all 0 start
    v_feat = torch.zeros((v_feat.shape[0],1),dtype=torch.float32).to(device)
    c_feat = torch.zeros((c_feat.shape[0],1),dtype=torch.float32).to(device)

        

    for itr in range(autoregression_iteration):
        x_pred,y_pred,scs,mult = m(AT,A,Q,b,c,v_feat,c_feat,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub,
                                                   AT_ori,A_ori,Q_ori,b_ori,c_ori,vscale,cscale,constscale,var_lb_ori,var_ub_ori)

        bqual = b.squeeze(-1)
        cqual = c.squeeze(-1)

        bqual_ori = b_ori.squeeze(-1).to(device)
        cqual_ori = c_ori.squeeze(-1).to(device)

        # ttloss, prim_res, dual_res, gaps = modf(Q,A,AT,bqual,cqual,x_pred,y_pred,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub,
        #                                            AT_ori,A_ori,Q_ori,b_ori,c_ori,vscale,cscale,constscale,var_lb_ori,var_ub_ori)
        ttloss, prim_res, dual_res, gaps = modf(Q_ori,A_ori,AT_ori,bqual_ori,cqual_ori,x_pred,y_pred,cons_ident,vars_ident_l,vars_ident_u,var_lb_ori,var_ub_ori, vscale,cscale,constscale)
        print(fnm) 
        print(f'primal_res: {prim_res.item()}_{itr}   dual_res: {dual_res.item()}   gaps: {gaps.item()}')
        azv = torch.zeros(x_pred.shape).to(device)
        print(f'    l2 norm err: {torch.norm(x_pred-x,2)}      0\'s l2 norm err: {torch.norm(azv-x,2)}\n\n')

        v_feat = x_pred.detach().clone()
        c_feat = y_pred.detach().clone()



    # compute primal obj
    # compute_obj(Q,c,x_pred,y_pred)

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



def sol_check(fdir,device,modf):
    f_tar = gzip.open(f'{fdir}','rb')
    to_pack = pickle.load(f_tar)
    v_feat = to_pack['vf'].to(device)
    c_feat = to_pack['cf'].to(device)
    Q = to_pack['Q'].to(device)
    A = to_pack['A'].to(device)
    AT = torch.transpose(A,0,1)
    c = to_pack['c'].to(device)
    b = to_pack['b'].to(device)
    x = to_pack['x'].to(device)
    y = to_pack['y'].to(device)
    c = torch.unsqueeze(c,-1)
    b = torch.unsqueeze(b,-1)

    Q_ori = to_pack['Q_ori'].to(device)
    A_ori = to_pack['A_ori'].to(device)
    AT_ori = torch.transpose(A_ori,0,1)
    c_ori = to_pack['c_ori'].to(device)
    b_ori = to_pack['b_ori'].to(device)
    c_ori = torch.unsqueeze(c_ori,-1)
    b_ori = torch.unsqueeze(b_ori,-1)
    var_lb_ori = torch.as_tensor(to_pack['var_lb_ori'], dtype=torch.float32).to(device)
    var_ub_ori = torch.as_tensor(to_pack['var_ub_ori'], dtype=torch.float32).to(device)

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
    if var_lb_ori.shape[-1]!=1:
        var_lb_ori = var_lb_ori.unsqueeze(-1)
    if var_ub_ori.shape[-1]!=1:
        var_ub_ori = var_ub_ori.unsqueeze(-1)
    # in this version, use all 0 start
    v_feat = torch.zeros((v_feat.shape[0],1),dtype=torch.float32).to(device)
    c_feat = torch.zeros((c_feat.shape[0],1),dtype=torch.float32).to(device)


    bqual_ori = b_ori.squeeze(-1).to(device)
    cqual_ori = c_ori.squeeze(-1).to(device)
    ttloss, prim_res, dual_res, gaps = modf(Q_ori,A_ori,AT_ori,bqual_ori,cqual_ori,x,y,cons_ident,vars_ident_l,vars_ident_u,var_lb_ori,var_ub_ori, vscale,cscale,constscale)
    # print(x)
    # print(y)
    print('primal_res',prim_res)
    print('dual_res',dual_res)
    print('gaps',gaps)
    print()


check_grad=False
# check_grad=True
def train(m,train_files,epoch,train_tar_dir,pareto,device,optimizer,choose_weight,autoregression_iteration,accu_loss):
    avg_train_loss = [0.0]*autoregression_iteration
    random.shuffle(train_files)
    with alive_bar(len(train_files),title=f"Training epoch {epoch}") as bar:
        for fnm in train_files:
            # input()
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



            Q_ori = to_pack['Q_ori'].to(device)
            A_ori = to_pack['A_ori'].to(device)
            AT_ori = torch.transpose(A_ori,0,1)
            c_ori = to_pack['c_ori'].to(device)
            b_ori = to_pack['b_ori'].to(device)
            c_ori = torch.unsqueeze(c_ori,-1)
            # print(c_ori)
            # quit()
            b_ori = torch.unsqueeze(b_ori,-1)
            var_lb_ori = torch.as_tensor(to_pack['var_lb_ori'], dtype=torch.float32).to(device)
            var_ub_ori = torch.as_tensor(to_pack['var_ub_ori'], dtype=torch.float32).to(device)


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
            if var_lb_ori.shape[-1]!=1:
                var_lb_ori = var_lb_ori.unsqueeze(-1)
            if var_ub_ori.shape[-1]!=1:
                var_ub_ori = var_ub_ori.unsqueeze(-1)

            
            # in this version, use all 0 start
            var_feat = torch.zeros((v_feat.shape[0],1),dtype=torch.float32).to(device)
            con_feat = torch.zeros((c_feat.shape[0],1),dtype=torch.float32).to(device)
            
            
            if accu_loss:
                net_loss = None
                optimizer.zero_grad()

            for itr in range(autoregression_iteration):
                if not accu_loss:
                    optimizer.zero_grad()
                x_pred,y_pred,scs_all,mult = m(AT,A,Q,b,c,var_feat,con_feat,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub,
                                                   AT_ori,A_ori,Q_ori,b_ori,c_ori,vscale,cscale,constscale,var_lb_ori,var_ub_ori)
                # print(x_pred)
                pr_it = scs_all[1].item()
                du_it = scs_all[2].item()
                gp_it = scs_all[3].item()

                loss = None
                # pareto front
                eps = torch.tensor(1e-6)
                if pareto:
                    relkkt1 = scs_all[1]
                    relkkt2 = scs_all[2]
                    relkkt3 = scs_all[3]
                    optimizer.zero_grad()
                    relkkt1.backward(retain_graph=True)
                    grd1 = compute_weight_grad(m)
                    mag_1 = torch.norm(grd1)+eps
                    optimizer.zero_grad()
                    relkkt2.backward(retain_graph=True)
                    grd2 = compute_weight_grad(m)
                    mag_2 = torch.norm(grd2)+eps
                    optimizer.zero_grad()
                    relkkt3.backward(retain_graph=True)
                    grd3 = compute_weight_grad(m)
                    mag_3 = torch.norm(grd3)+eps
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


                # loss = (itr*loss)/autoregression_iteration
                loss_num = scs.item()
                avg_train_loss[itr] += loss_num

                
                if choose_weight:
                    print(x_pred)
                    print(x)
                    print(y_pred)
                    print(y)
                    input('wait')

                if accu_loss:
                    if net_loss is None:
                        net_loss = loss*((itr+1)/autoregression_iteration)
                    else:
                        net_loss = net_loss+loss*((itr+1)/autoregression_iteration)
                # else:
                #     loss.backward()
                #     # optimizer.step()

                var_feat = x_pred.detach().clone()
                con_feat = y_pred.detach().clone()

                

                # print(f'Auto-regression on {fnm}, iteration {itr} loss:{loss_num}')
            nm=torch.norm(x_pred,float('inf'))
            print(f'x linf norm: {nm}')
            nm=torch.norm(y_pred,float('inf'))
            print(f'y linf norm: {nm}')
            if accu_loss:
                print(f'{fnm}   avg_loss: {round(net_loss.item()/autoregression_iteration,4)}         {round(pr_it,4)}   ---   {round(du_it,4)}   ---   {round(gp_it,4)}')
                net_loss.backward()
                if check_grad:
                    for name, param in m.named_parameters():
                        print(param.grad,name)
                        input()
                    quit()
                optimizer.step()
            else:
                print(f'{fnm}   avg_loss: {round(loss.item(),4)}        {round(pr_it,4)}   ---   {round(du_it,4)}   ---   {round(gp_it,4)}')
                loss.backward()
                if check_grad:
                    for name, param in m.named_parameters():
                        print(param.grad,name)
                        input()
                    quit()
                optimizer.step()

            # print(f'Auto regression finished')
            # input()

            
            # try:
            #     print(compute_weight_grad(m))
            #     # input('f8i')
            # except:
            #     print('no grad')
            # print('!!!!!!!!!   min/max',torch.min(x_pred).item(),torch.max(x_pred).item())


            # f=open('tmp.bounds','w')
            # for i in range(x_pred.shape[0]):
            #     if x_pred[i].item()<var_lb[i].item() or x_pred[i].item()>var_ub[i].item():
            #         st=f'{var_lb[i].item()} {x_pred[i].item()} {var_ub[i].item()}\n'
            #         f.write(st)
            # f.close()
            bar()

    return avg_train_loss


def draw_plot(x=None,y=None,ident=''):
    plt.clf()
    if x is not None:
        plt.plot(x,y)
    else:
        plt.plot(y)
    plt.title(f'{ident}')
    plt.savefig(f'../plots/plt_{ident}.png')






def process_supervised(m,files,epoch,tar_dir,device,optimizer,choose_weight=False,autoregression_iteration=1,training=True,accu_loss = True):
    if not training:
        return valid_supervised(m,files,epoch,tar_dir,device,optimizer,autoregression_iteration)
    else:
        return train_supervised(m,files,epoch,tar_dir,device,optimizer,choose_weight,autoregression_iteration,accu_loss = accu_loss)




def valid_supervised(m,valid_files,epoch,valid_tar_dir,device,modf,autoregression_iteration):

    loss_func = torch.nn.MSELoss()
    avg_valid_loss = [0.0]*autoregression_iteration
    avg_sc = [0.0]*autoregression_iteration
    avg_scprimal = [0.0]*autoregression_iteration
    avg_scdual = [0.0]*autoregression_iteration
    avg_scgap = [0.0]*autoregression_iteration
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
                x = to_pack['x'].to(device)
                y = to_pack['y'].to(device)
                c = torch.unsqueeze(c,-1)
                b = torch.unsqueeze(b,-1)
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



                for itr in range(autoregression_iteration):
                    x_pred,y_pred,scs_all,mult = m(AT,A,Q,b,c,v_feat,c_feat,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub)

                    bqual = b.squeeze(-1).to(device)
                    cqual = c.squeeze(-1).to(device)
                    ttloss, prim_res, dual_res, gaps = modf(Q,A,AT,bqual,cqual,x_pred,y_pred,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub)
                    print(f'primal_res: {prim_res.item()}   dual_res: {dual_res.item()}   gaps: {gaps.item()}')
                    real_sc = torch.max(prim_res,torch.max(dual_res,gaps))
                    real_sc_num = real_sc.item()
                    avg_sc[itr] += real_sc_num

                    avg_scprimal[itr] += prim_res.item()
                    avg_scdual[itr] += dual_res.item()
                    avg_scgap[itr] += gaps.item()

                    loss = loss_func(x_pred, x)
                    loss_num = loss.item()
                    avg_valid_loss[itr] += loss_num


                    v_feat = x_pred.detach().clone()
                    c_feat = y_pred.detach().clone()

                    

                    print(f'Auto-regression on {fnm}, iteration {itr} loss:{loss_num}     real sc:{real_sc_num}')
                bar()
    return avg_valid_loss, avg_sc, avg_scprimal, avg_scdual, avg_scgap

def train_supervised(m,train_files,epoch,train_tar_dir,device,optimizer,choose_weight,autoregression_iteration,accu_loss):
    avg_train_loss = [0.0]*autoregression_iteration

    loss_func = torch.nn.MSELoss()

    random.shuffle(train_files)
    with alive_bar(len(train_files),title=f"Training epoch {epoch}") as bar:
        for fnm in train_files:
            # input()
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
            x = to_pack['x'].to(device)
            y = to_pack['y'].to(device)
            c = torch.unsqueeze(c,-1)
            b = torch.unsqueeze(b,-1)
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

            
            # in this version, use all 0 start
            var_feat = torch.zeros((v_feat.shape[0],1),dtype=torch.float32).to(device)
            con_feat = torch.zeros((c_feat.shape[0],1),dtype=torch.float32).to(device)
            
            
            if accu_loss:
                net_loss = None
                optimizer.zero_grad()

            for itr in range(autoregression_iteration):
                if not accu_loss:
                    optimizer.zero_grad()
                x_pred,y_pred,scs_all,mult = m(AT,A,Q,b,c,var_feat,con_feat,cons_ident,vars_ident_l,vars_ident_u,var_lb,var_ub)
                # print(x_pred)
                pr_it = scs_all[1].item()
                du_it = scs_all[2].item()
                gp_it = scs_all[3].item()
                loss = loss_func(x_pred, x)

                loss_num = loss.item()
                avg_train_loss[itr] += loss_num

                
                if choose_weight:
                    print(x_pred)
                    print(x)
                    print(y_pred)
                    print(y)
                    input('wait')

                if accu_loss:
                    if net_loss is None:
                        net_loss = loss
                    else:
                        net_loss = net_loss+loss
                # else:
                #     loss.backward()
                #     # optimizer.step()

                var_feat = x_pred.detach().clone()
                con_feat = y_pred.detach().clone()


            if accu_loss:
                print(f'{fnm}   avg_loss: {round(net_loss.item()/autoregression_iteration,4)}         {round(pr_it,4)}   ---   {round(du_it,4)}   ---   {round(gp_it,4)}')
                net_loss.backward()
                if check_grad:
                    for name, param in m.named_parameters():
                        print(param.grad,name)
                        input()
                    quit()
                optimizer.step()
            else:
                print(f'{fnm}   avg_loss: {round(loss.item(),4)}        {round(pr_it,4)}   ---   {round(du_it,4)}   ---   {round(gp_it,4)}')
                loss.backward()
                if check_grad:
                    for name, param in m.named_parameters():
                        print(param.grad,name)
                        input()
                    quit()
                optimizer.step()

            bar()

    return avg_train_loss
