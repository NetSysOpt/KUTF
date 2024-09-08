import torch
import torch.nn as nn
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def print_parameters(model):
    z = [p for p in model.parameters() if p.requires_grad]
    for zz in z:
        print(zz.numel(),zz.names)
        
def compute_weight_grad(model):
    z = []
    k = 0 
    for a,grad in model.named_parameters():
        if "weight" in a:
            #print(grad.grad)
            if grad.grad != None:
                z.append(torch.norm(grad.grad,dim=None).item())
                k=k+1
    return torch.tensor(z)

        
def divide_weights(model,div=4.0,div_bias=True):
    params = model.named_parameters()
    for name,parm in params:
        if "weight" in name or div_bias:
            parm.data = parm.data/div

def init_weights(m):
    a = 0.0001
    b = a+0.0002
    if isinstance(m, nn.Linear):
        # torch.nn.init.xavier_uniform(m.weight)
        # torch.nn.init.uniform_(m.weight,a=0.013,b=0.0145)
        torch.nn.init.xavier_uniform(m.weight)
        # torch.nn.init.uniform_(m.weight,a=a,b=b)
        if m.bias is not None:
            torch.nn.init.uniform_(m.bias,a=a,b=b)
        # m.bias.data.fill_(0.001)




class step_size_pred(torch.nn.Module):
    
    def __init__(self,feat_size):
        super(step_size_pred,self).__init__()
        self.feat_size = feat_size
        
        self.lin_1 = nn.Sequential(
            nn.Linear(feat_size,1,bias=False),
        )
        self.lin_2 = nn.Sequential(
            nn.Linear(feat_size,1,bias=False),
        )
        self.act = nn.LeakyReLU()
        
    def forward(self,x,y):
        return self.act(torch.sum(self.lin_1(x)) + torch.sum(self.lin_2(y)))


class inner_loop_geq(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size):
        super(inner_loop_geq,self).__init__()
        self.feat_size = feat_size
        self.emu_gamma = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))
        self.emu_eta = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))
        self.emu_beta = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))
        self.emu_theta = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))

        # TO be changed
        self.yproj = proj_y(feat_size)
        self.xproj = proj_x(feat_size)

        self.lin_1 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=False),
        )
        self.lin_3 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=False),
        )
        self.lin_4 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=False),
        )
        
        
    def forward(self,x,x_bar,y,Q,A,AT,c,b,indicator_y,indicator_x_l,indicator_x_u,l,u,cmat,bmat):

        # start updating
        x_md = (1.0-self.emu_beta)*x_bar + (self.emu_beta)*x
        
        # update x
        x_new = x - self.emu_eta * (self.lin_3(torch.matmul(Q,x_md)) + cmat - self.lin_4(torch.matmul(AT,y)))
        x_new = self.xproj(x_new, indicator_x_l, indicator_x_u, l, u)


        # need to check if we want to wrap this with a linear layer
        #       current setting seems better for the theoretical part
        #  !Y update
        x_delta = self.emu_theta*(x_new - x) + x_new
        x_delta = self.emu_gamma * (bmat - self.lin_1(torch.matmul(A,x_delta))  )
        y_new = y + x_delta
        y_new = self.yproj(y_new, indicator_y)

        # print('step sizes')
        # print(self.emu_eta,self.emu_theta,self.emu_gamma,self.emu_beta)
        # quit()

        x_bar = (1.0-self.emu_beta)*x_bar + (self.emu_beta)*x_new

        return x_new,x_bar,y_new


class inner_loop_geq_stepsize(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size):
        super(inner_loop_geq_stepsize,self).__init__()
        self.feat_size = feat_size
        self.emu_gamma = step_size_pred(feat_size)
        self.emu_eta = step_size_pred(feat_size)
        self.emu_beta = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))
        self.emu_theta = step_size_pred(feat_size)

        # TO be changed
        self.yproj = proj_y(feat_size)
        self.xproj = proj_x(feat_size)

        self.lin_1 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=False),
        )
        self.lin_3 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=False),
        )
        self.lin_4 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=False),
        )
        
        
    def forward(self,x,x_bar,y,Q,A,AT,c,b,indicator_y,indicator_x_l,indicator_x_u,l,u,cmat,bmat):

        # start updating
        x_md = (1.0-self.emu_beta)*x_bar + (self.emu_beta)*x
        
        # update x
        x_new = x - self.emu_eta(x,y) * (self.lin_3(torch.matmul(Q,x_md)) + cmat - self.lin_4(torch.matmul(AT,y)))
        x_new = self.xproj(x_new, indicator_x_l, indicator_x_u, l, u)

        # need to check if we want to wrap this with a linear layer
        #       current setting seems better for the theoretical part
        #  !Y update
        x_delta = self.emu_theta(x_new,y)*(x_new - x) + x_new
        x_delta = self.emu_gamma(x_new,y) * (bmat - self.lin_1(torch.matmul(A,x_delta))  )
        y_new = self.yproj(y + x_delta, indicator_y)


        x_bar = (1.0-self.emu_beta)*x_bar + (self.emu_beta)*x_new

        return x_new,x_bar,y_new


class PDQP_Net_geq(torch.nn.Module):
    def __init__(self,x_size,y_size,feat_size,nlayer=8):
        super(PDQP_Net_geq,self).__init__()

        self.feat_size = feat_size

        # self.init_x = nn.Sequential(
        #     nn.Linear(x_size,feat_size,bias=False),
        #     # nn.LeakyReLU(),
        # )
        # self.init_y = nn.Sequential(
        #     nn.Linear(y_size,feat_size,bias=False),
        #     # nn.LeakyReLU(),
        # )
        self.init = nn.Sequential(
            nn.Linear(y_size,feat_size,bias=False),
            # nn.LeakyReLU(),
        )

        self.updates = nn.ModuleList()
        for indx in range(nlayer):
            self.updates.append(inner_loop_geq(feat_size,feat_size,feat_size))
            # self.updates.append(inner_loop_geq_stepsize(feat_size,feat_size,feat_size))

        # self.out_x = nn.Sequential(
        #     nn.Linear(feat_size,1,bias=False),
        # )
        # self.out_y = nn.Sequential(
        #     nn.Linear(feat_size,1,bias=False),
        # )
        self.out = nn.Sequential(
            nn.Linear(feat_size,1,bias=False),
        )
        
    def forward(self,A,AT,Q,b,c,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u):

        # initial encoding
        # x = self.init_x(x)
        # y = self.init_y(y)
        x = self.init(x)
        y = self.init(y)
        
        x_bar = x
        cmat = torch.matmul(c,torch.ones((1,self.feat_size),dtype = torch.float32).to(c.device))
        bmat = torch.matmul(b,torch.ones((1,self.feat_size),dtype = torch.float32).to(b.device))
        for index, layer in enumerate(self.updates):
            x,x_bar,y = layer(x,x_bar,y,Q,A,AT,c,b,indicator_y,indicator_x_l,indicator_x_u,l,u,cmat,bmat)
        # x = self.out_x(x)
        # y = self.out_y(y)
        x = self.out(x)
        y = self.out(y)
        return x,y



class PDQP_Net_AR_geq(torch.nn.Module):
    def __init__(self,x_size,y_size,feat_size,max_k = 20, threshold = 1e-8,nlayer=1, 
                 tfype='linf', use_dual=True, eta_opt = 1e+6, div=4.0):
        super(PDQP_Net_AR_geq,self).__init__()
        self.max_k = max_k
        self.threshold = threshold
        
        self.net = PDQP_Net_geq(x_size,y_size,feat_size,nlayer=nlayer)
        self.net.apply(init_weights)
        divide_weights(self.net,div=div,div_bias=True)

        self.qual_func = relKKT_general(tfype,eta_opt)
            
        self.final_out = proj_x_no_mlp(1)

    def forward(self,AT,A,Q,b,c,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u,
                                AT_ori=None,A_ori=None,Q_ori=None,b_ori=None,c_ori=None,vscale=None,cscale=None,constscale=None,var_lb_ori=None,var_ub_ori=None):
        bqual = b.squeeze(-1)
        cqual = c.squeeze(-1)
        bqual_ori = b_ori.squeeze(-1)
        cqual_ori = c_ori.squeeze(-1)

        scs = None
        mult = 0.0
        for iter in range(self.max_k):
            x,y = self.net(A,AT,Q,b,c,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u)
            # x = self.final_out(x, indicator_x_l, indicator_x_u, l, u)
            sc = self.qual_func(Q_ori,A_ori,AT_ori,bqual_ori,cqual_ori,x,y,indicator_y,indicator_x_l,indicator_x_u,var_lb_ori,var_ub_ori,
                                vscale,cscale,constscale)

            scs = sc
            if sc[0].item() <= self.threshold:
                break
        else:   
            mult = 1.0
        # x = self.final_out(x, indicator_x_l, indicator_x_u, l, u)
        # scs = self.qual_func(Q,A,AT,bqual,cqual,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u)

        return x,y,scs,mult
        








class PDQP_Net(torch.nn.Module):
    def __init__(self,x_size,y_size,feat_sizes):
        super(PDQP_Net,self).__init__()


        self.updates = nn.ModuleList()
        # self.updates.append(inner_loop(x_size,y_size,feat_sizes[0],x_size))
        # self.updates.append(inner_loop(feat_sizes[0],feat_sizes[0],feat_sizes[1],x_size))
        # for indx in range(2,len(feat_sizes)):
        #     # print(indx,indx-2)
        #     self.updates.append(inner_loop(feat_sizes[indx-1],feat_sizes[indx-1],feat_sizes[indx],feat_sizes[indx-2]))



        self.updates.append(inner_loop_restart(x_size,y_size,feat_sizes[0],x_size))
        self.updates.append(inner_loop_restart(feat_sizes[0],feat_sizes[0],feat_sizes[1],x_size))
        for indx in range(2,len(feat_sizes)):
            # print(indx,indx-2)
            self.updates.append(inner_loop_restart(feat_sizes[indx-1],feat_sizes[indx-1],feat_sizes[indx],feat_sizes[indx-2]))
        
        # self.out_x = nn.Sequential(
        #     nn.Linear(feat_sizes[-1],1,bias=True),
        #     nn.LeakyReLU(),
        # )
        # self.out_y = nn.Sequential(
        #     nn.Linear(feat_sizes[-1],1,bias=True),
        #     nn.LeakyReLU(),
        # )

    def forward(self,A,Q,b,c,x,y):
        c = torch.unsqueeze(c,-1)
        b = torch.unsqueeze(b,-1)
        AT = torch.transpose(A,0,1)
        x_bar = x
        y_bar = y
        x_hist = [x,x]
        # print(x)
        for index, layer in enumerate(self.updates):
            x_past = x_hist[-2]
            # print(x_past.shape, x.shape)
            x,x_bar,y,y_bar = layer(x_past,x,x_bar,y,y_bar,Q,A,AT,c,b)
            x_hist.append(x)
        # quit()

        return x,y


# torch.transpose(A,0,1)
class inner_loop_restart(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size,past_size):
        super(inner_loop_restart,self).__init__()
        self.feat_size = feat_size
        self.emu_gamma = torch.nn.Parameter(torch.randn(size=(1, ),requires_grad=True))
        self.emu_eta = torch.nn.Parameter(torch.randn(size=(1, ),requires_grad=True))
        self.emu_beta = torch.nn.Parameter(torch.randn(size=(1, ),requires_grad=True))
        self.emu_theta = torch.nn.Parameter(torch.randn(size=(1, ),requires_grad=True))

        self.exp_x_past = nn.Sequential(
            nn.Linear(past_size,feat_size,bias=True),
            # nn.ReLU(),
            # nn.LayerNorm(feat_size)
        )
        self.exp_x_bar = nn.Sequential(
            nn.Linear(x_size,feat_size,bias=True),
            # nn.ReLU(),
            # nn.LayerNorm(feat_size)
        )
        self.exp_x = nn.Sequential(
            nn.Linear(x_size,feat_size,bias=True),
            # nn.ReLU(),
            # nn.LayerNorm(feat_size)
        )
        self.exp_y_bar = nn.Sequential(
            nn.Linear(y_size,feat_size,bias=True),
            # nn.ReLU(),
            # nn.LayerNorm(feat_size)
        )
        self.exp_y = nn.Sequential(
            nn.Linear(y_size,feat_size,bias=True),
            # nn.ReLU(),
            # nn.LayerNorm(feat_size)
        )


        self.lin_1 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            # nn.LayerNorm(feat_size)
            # nn.ReLU(),
        )

        self.lin_2 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            # nn.LayerNorm(feat_size)
            # nn.ReLU(),
        )

        self.lin_3 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            # nn.LayerNorm(feat_size)
            # nn.ReLU(),
        )
        
        self.shift_x1 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            nn.ReLU(),
        )
        self.shift_x2 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            nn.ReLU(),
        )
        self.shift_x3 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
        )
        
        self.shift_y1 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            nn.ReLU(),
        )
        self.shift_y2 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            nn.ReLU(),
        )
        self.shift_y3 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
        )

        self.shift_y = nn.Sequential(
            # nn.Linear(feat_size,feat_size,bias=True),
            nn.ReLU(),
        )
        # self.proj_pos = nn.ReLU()
        
    def forward(self,x_past,x,x_bar,y,y_bar,Q,A,AT,c,b):
        # encoding to another dimension:
        x_past = self.exp_x_past(x_past)
        x_bar = self.exp_x_bar(x_bar)
        x = self.exp_x(x)
        y = self.exp_y(y)
        y_bar = self.exp_y_bar(y_bar)
        c = torch.matmul(c,torch.ones((1,self.feat_size),dtype = torch.float32))
        b = torch.matmul(b,torch.ones((1,self.feat_size),dtype = torch.float32))

        # print(x,y)
        # input()

        # start updating
        x_md = (1.0-self.emu_beta)*x_bar + self.emu_beta*x
        # print(x_md, self.emu_beta)
        # input()

        # need to check if we want to wrap this with a linear layer
        #       current setting seems better for the theoretical part
        x_delta = self.emu_theta*(x - x_past) + x

        x_delta = self.emu_gamma * self.lin_1(torch.matmul(A,x_delta))
        # y_new = self.proj_pos(y + x_delta - b)
        # print('GHGGG', y.shape,x_delta.shape, b.shape)
        y_new = y + x_delta - b


        x_new = x - self.emu_eta * (self.lin_2(torch.matmul(Q,x_md)) + c + self.lin_3(torch.matmul(AT,y))) 

        # relu to emulate proj to l<x<u, looks weird but aligns exactly
        x_new = self.shift_x3(self.shift_x1(x_new) - self.shift_x2(x_new))
        # y_new = self.shift_y(y_new)
        y_new = self.shift_y3(self.shift_y1(y_new) - self.shift_y2(y_new))

        x_bar = (1.0-self.emu_beta)*x_bar + self.emu_beta*x_new
        y_bar = (1.0-self.emu_beta)*y_bar + self.emu_beta*y_new


        return x_new,x_bar,y_new,y_bar



class PDQP_Net_new(torch.nn.Module):
    def __init__(self,x_size,y_size,feat_size,nlayer=8):
        super(PDQP_Net_new,self).__init__()

        self.feat_size = feat_size

        self.init_x = nn.Sequential(
            nn.Linear(x_size,feat_size,bias=True),
            # nn.LeakyReLU(),
        )
        self.init_y = nn.Sequential(
            nn.Linear(y_size,feat_size,bias=True),
            # nn.LeakyReLU(),
        )

        self.updates = nn.ModuleList()
        for indx in range(nlayer):
            self.updates.append(inner_loop_new(feat_size,feat_size,feat_size,feat_size))

        self.out_x = nn.Sequential(
            nn.Linear(feat_size,1,bias=True),
        )
        self.out_y = nn.Sequential(
            nn.Linear(feat_size,1,bias=True),
        )
        
    def forward(self,A,AT,Q,b,c,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u):

        # initial encoding
        x = self.init_x(x)
        y = self.init_y(y)
        

        x_bar = x
        x_hist = [x,x]
        # print(x)
        cmat = torch.matmul(c,torch.ones((1,self.feat_size),dtype = torch.float32).to(c.device))
        bmat = torch.matmul(b,torch.ones((1,self.feat_size),dtype = torch.float32).to(b.device))
        for index, layer in enumerate(self.updates):
            x_past = x_hist[-2]
            x,x_bar,y = layer(x_past,x,x_bar,y,Q,A,AT,c,b,indicator_y,indicator_x_l,indicator_x_u,l,u,cmat,bmat)
            x_hist.append(x)
            # if len(x_hist)>3:
            #     x_hist[-3]=None
        x = self.out_x(x)
        y = self.out_y(y)
        return x,y


class PDQP_Net_AR(torch.nn.Module):
    def __init__(self,x_size,y_size,feat_size,max_k = 20, threshold = 1e-8,nlayer=1, type='linf', use_dual=True):
        super(PDQP_Net_AR,self).__init__()
        self.max_k = max_k
        self.threshold = threshold
        
        self.net = PDQP_Net_new(x_size,y_size,feat_size,nlayer=nlayer)
        self.net.apply(init_weights)

        # self.qual_func = relKKT_l2([1.0,0.2,0.9])
        self.qual_func=None
        # if type=='l2':  
        #     self.qual_func = relKKT_l2()
        # if type=='l1':  
        #     self.qual_func = relKKT_l1()
        # if type=='linf':  
        #     self.qual_func = relKKT(use_dual)
        self.qual_func = relKKT_general(tfype)
            
        self.final_out = proj_x_no_mlp(1)

    def forward(self,AT,A,Q,b,c,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u):
        bqual = b.squeeze(-1)
        cqual = c.squeeze(-1)
        
        # if debug:
        #     print(f'Problem size: {A.shape}')

        # x_hist = []
        # y_hist = []
        # l_hist = []
        scs = None
        mult = 0.0
        for iter in range(self.max_k):
            x,y = self.net(A,AT,Q,b,c,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u)
            # x = self.final_out(x, indicator_x_l, indicator_x_u, l, u)
            # x_hist.append(x)
            # y_hist.append(y)
            sc = self.qual_func(Q,A,AT,bqual,cqual,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u)

            # l_hist.append(sc)
            # if scs is None:
            #     scs = sc
            # else:
            #     scs += sc
            scs = sc
            if sc[0].item() <= self.threshold:
                break
        else:   
            mult = 1.0
        # x = self.final_out(x, indicator_x_l, indicator_x_u, l, u)
        # scs = self.qual_func(Q,A,AT,bqual,cqual,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u)

        # if debug:
        #     print(f'Prediction generated within {iter} iterations')

        


        return x,y,scs,mult
        # return x,y,l_hist,mult
        
        
        
class PDQP_layer_shared(torch.nn.Module):
    def __init__(self,x_size,y_size,feat_size,nlayer=8):
        super(PDQP_layer_shared,self).__init__()

        self.feat_size = feat_size

        self.init = nn.Sequential(
            nn.Linear(x_size,feat_size,bias=False),
            # nn.LeakyReLU(),
        )

        self.updates = nn.ModuleList()
        for indx in range(nlayer):
            self.updates.append(inner_loop_shared(feat_size,feat_size,feat_size,feat_size))

        self.out = nn.Sequential(
            nn.Linear(feat_size,1,bias=False),
        )
        self.final_out = proj_x_no_mlp(1)
        
    def forward(self,A,AT,Q,b,c,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u):

        # initial encoding
        x = self.init(x)
        y = self.init(y)
        

        x_bar = x
        x_hist = [x,x]
        # print(x)
        cmat = torch.matmul(c,torch.ones((1,self.feat_size),dtype = torch.float32).to(c.device))
        bmat = torch.matmul(b,torch.ones((1,self.feat_size),dtype = torch.float32).to(b.device))
        for index, layer in enumerate(self.updates):
            x_past = x_hist[-2]
            x,x_bar,y = layer(x_past,x,x_bar,y,Q,A,AT,c,b,indicator_y,indicator_x_l,indicator_x_u,l,u,cmat,bmat)
            # input('gpo')
            x_hist.append(x)
        x = self.out(x)
        y = self.out(y)
        x = self.final_out(x, indicator_x_l, indicator_x_u, l, u)

        return x,y

        
class inner_loop_shared(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size,past_size):
        super(inner_loop_shared,self).__init__()
        self.feat_size = feat_size
        self.emu_gamma = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))
        self.emu_eta = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))
        self.emu_beta = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))
        self.emu_theta = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))

        self.yproj = proj_y(feat_size)
        self.xproj = proj_x(feat_size)

        self.lin = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=False),
        )
        
        
    def forward(self,x_past,x,x_bar,y,Q,A,AT,c,b,indicator_y,indicator_x_l,indicator_x_u,l,u,cmat,bmat):
        # print(self.emu_beta)
        # print(self.emu_eta)
        # print(self.emu_gamma)
        # print(self.emu_theta)
        # quit()

        # start updating
        x_md = (1.0-self.emu_beta)*x_bar + self.emu_beta*x
        

        # need to check if we want to wrap this with a linear layer
        #       current setting seems better for the theoretical part
        #  !Y update
        x_delta = self.emu_theta*(x - x_past) + x
        x_delta = self.emu_gamma * self.lin(torch.matmul(A,x_delta))
        y_new = self.yproj(y + x_delta - bmat, indicator_y)


        #  !X update
        x_new = self.lin(x) - self.emu_eta * (self.lin(torch.matmul(Q,x_md)) + cmat + self.lin(torch.matmul(AT,y))) 
        x_new = self.xproj(x_new, indicator_x_l, indicator_x_u, l, u)

        x_bar = (1.0-self.emu_beta)*x_bar + self.emu_beta*x_new

        return x_new,x_bar,y_new
        
        
class PDQP_Net_shared(torch.nn.Module):
    def __init__(self,x_size,y_size,feat_size,max_k = 20, threshold = 1e-4,nlayer=1,type='l2'):
        super(PDQP_Net_shared,self).__init__()
        self.max_k = max_k
        self.threshold = threshold
        
        self.net = PDQP_layer_shared(x_size,y_size,feat_size,nlayer=nlayer)
        self.net.apply(init_weights)

        # self.qual_func = relKKT_l2([1.0,0.2,0.9])
        # self.qual_func = relKKT_l2()
        # self.qual_func = relKKT_l1()
        # self.qual_func = relKKT()
        # self.qual_func=None
        # if type=='l2':  
        #     self.qual_func = relKKT_l2()
        # if type=='l1':  
        #     self.qual_func = relKKT_l1()
        # if type=='linf':  
        #     self.qual_func = relKKT()
        self.qual_func = relKKT_general(tfype)

    def forward(self,AT,A,Q,b,c,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u):
        bqual = b.squeeze(-1)
        cqual = c.squeeze(-1)
        
        # if debug:
        #     print(f'Problem size: {A.shape}')

        x_hist = []
        y_hist = []
        # l_hist = []
        scs = None
        mult = 0.0
        for iter in range(self.max_k):
            x,y = self.net(A,AT,Q,b,c,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u)
            x_hist.append(x)
            y_hist.append(y)
            sc = self.qual_func(Q,A,AT,bqual,cqual,x,y,indicator_y,indicator_x_l,indicator_x_u,l,u)
            # l_hist.append(sc)
            # if scs is None:
            #     scs = sc
            # else:
            #     scs += sc
            scs = sc
            if sc[0].item() <= self.threshold:
                break
        else:
            mult = 1.0

            


        return x,y,scs,mult
        # return x,y,l_hist,mult

        
        
        

class proj_y(torch.nn.Module):
    def __init__(self, feat_size):
        super(proj_y,self).__init__()
        self.rr = nn.ReLU()

    def forward(self,y,indicator_y):
        if indicator_y.shape[-1]!=1:
            indicator_y = indicator_y.unsqueeze(-1)
        res = y + indicator_y * self.rr(-y)
        return res

class proj_x(torch.nn.Module):
    def __init__(self, feat_size):
        super(proj_x,self).__init__()
        self.rr = nn.LeakyReLU()
        self.feat_size = feat_size
        self.lin_1 = nn.Sequential(
            nn.Linear(feat_size+1,feat_size,bias=False),
        )

        self.lin_2 = nn.Sequential(
            nn.Linear(feat_size+1,feat_size,bias=False),
        )


    def forward(self,x,indicator_x_l,indicator_x_u, l, u):
        if indicator_x_l.shape[-1]!=1:
            indicator_x_l = indicator_x_l.unsqueeze(-1)
        if indicator_x_u.shape[-1]!=1:
            indicator_x_u = indicator_x_u.unsqueeze(-1)
        # p1 = torch.cat((x,u),-1)
        xuu = x - indicator_x_u * self.rr(self.lin_1(torch.cat((x,u),-1)))
        # p2 = torch.cat((xuu,l),-1)
        x = xuu + indicator_x_l * self.rr(self.lin_2(torch.cat((xuu,l),-1)))


        # u = u.repeat([1,self.feat_size])
        # l = l.repeat([1,self.feat_size])
        # x = x - indicator_x_u * self.rr(x-u) + indicator_x_l * self.rr(l-x)
        return x




# class proj_x(torch.nn.Module):
#     def __init__(self, feat_size):
#         super(proj_x,self).__init__()
#         self.rr = nn.ReLU()
#         self.feat_size = feat_size
#         self.lin_1 = nn.Sequential(
#             nn.Linear(feat_size+1,feat_size,bias=False),
#         )

#         self.lin_2 = nn.Sequential(
#             nn.Linear(feat_size+1,feat_size,bias=False),
#         )


#     def forward(self,x,indicator_x_l,indicator_x_u, l, u):
#         if indicator_x_l.shape[-1]!=1:
#             indicator_x_l = indicator_x_l.unsqueeze(-1)
#         if indicator_x_u.shape[-1]!=1:
#             indicator_x_u = indicator_x_u.unsqueeze(-1)
#         p1 = torch.cat((x,u),-1)
#         # u = u.repeat([1,self.feat_size])
#         # l = l.repeat([1,self.feat_size])
#         # print(self.lin_1.device)
#         # print(p1.device)
#         # xuu = x - indicator_x_u * self.rr(x-u)
#         xuu = x - indicator_x_u * self.rr(self.lin_1(p1))
#         p2 = torch.cat((xuu,l),-1)
#         # x = xuu + indicator_x_l * self.rr(l-x)
#         x = xuu + indicator_x_l * self.rr(self.lin_2(p2))
#         return x
    
    
class proj_x_no_mlp(torch.nn.Module):
    def __init__(self, feat_size):
        super(proj_x_no_mlp,self).__init__()
        self.act = nn.ReLU()


    def forward(self,x,indicator_x_l,indicator_x_u, l, u):
        # if indicator_x_l.shape[-1]!=1:
        #     indicator_x_l = indicator_x_l.unsqueeze(-1)
        # if indicator_x_u.shape[-1]!=1:
        #     indicator_x_u = indicator_x_u.unsqueeze(-1)
        # p1 = torch.cat((x,u),-1)
        # print(self.lin_1.device)
        # print(p1.device)
        
        x = x + torch.mul(self.act(l-x), indicator_x_l) - torch.mul(self.act(x-u), indicator_x_u)
        return x

class inner_loop_new(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size,past_size):
        super(inner_loop_new,self).__init__()
        self.feat_size = feat_size
        self.emu_gamma = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))
        self.emu_eta = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))
        self.emu_beta = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))
        self.emu_theta = torch.nn.Parameter(torch.ones(size=(1, ),requires_grad=True))

        # TO be changed
        self.yproj = proj_y(feat_size)
        self.xproj = proj_x(feat_size)

        self.lin_1 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=False),
        )
        self.lin_2 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=False),
        )
        self.lin_3 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=False),
        )
        self.lin_4 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=False),
        )
        
        
    def forward(self,x_past,x,x_bar,y,Q,A,AT,c,b,indicator_y,indicator_x_l,indicator_x_u,l,u,cmat,bmat):
        # print(self.emu_beta)
        # print(self.emu_eta)
        # print(self.emu_gamma)
        # print(self.emu_theta)
        # quit()

        # start updating
        x_md = (1.0-self.emu_beta)*x_bar + self.emu_beta*x
        

        # need to check if we want to wrap this with a linear layer
        #       current setting seems better for the theoretical part
        #  !Y update
        x_delta = self.emu_theta*(x - x_past) + x
        x_delta = self.emu_gamma * self.lin_1(torch.matmul(A,x_delta))
        y_new = self.yproj(y + x_delta - bmat, indicator_y)


        #  !X update
        x_new = self.lin_2(x) - self.emu_eta * (self.lin_3(torch.matmul(Q,x_md)) + cmat + self.lin_4(torch.matmul(AT,y))) 
        x_new = self.xproj(x_new, indicator_x_l, indicator_x_u, l, u)

        x_bar = (1.0-self.emu_beta)*x_bar + self.emu_beta*x_new

        return x_new,x_bar,y_new






class inner_loop(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size,past_size):
        super(inner_loop,self).__init__()
        self.feat_size = feat_size
        self.emu_gamma = torch.nn.Parameter(torch.randn(size=(1, ),requires_grad=True))
        self.emu_eta = torch.nn.Parameter(torch.randn(size=(1, ),requires_grad=True))
        self.emu_beta = torch.nn.Parameter(torch.randn(size=(1, ),requires_grad=True))
        self.emu_theta = torch.nn.Parameter(torch.randn(size=(1, ),requires_grad=True))

        self.exp_x_past = nn.Sequential(
            nn.Linear(past_size,feat_size,bias=True),
            # nn.ReLU(),
            # nn.LayerNorm(feat_size)
        )
        self.exp_x_bar = nn.Sequential(
            nn.Linear(x_size,feat_size,bias=True),
            # nn.ReLU(),
        )
        self.exp_x = nn.Sequential(
            nn.Linear(x_size,feat_size,bias=True),
            nn.LayerNorm(feat_size)
            # nn.ReLU(),
        )
        self.exp_y_bar = nn.Sequential(
            nn.Linear(y_size,feat_size,bias=True),
            # nn.ReLU(),
        )
        self.exp_y = nn.Sequential(
            nn.Linear(y_size,feat_size,bias=True),
            nn.LayerNorm(feat_size)
            # nn.ReLU(),
        )

        self.lin_1 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            # nn.ReLU(),
        )

        self.lin_2 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            # nn.ReLU(),
        )

        self.lin_3 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            # nn.ReLU(),
        )
        
        self.shift_x = nn.Sequential(
            # nn.Linear(feat_size,feat_size,bias=True),
            nn.ReLU(),
        )
        self.shift_y = nn.Sequential(
            # nn.Linear(feat_size,feat_size,bias=True),
            nn.ReLU(),
        )
        # self.proj_pos = nn.ReLU()
        
    def forward(self,x_past,x,x_bar,y,y_bar,Q,A,AT,c,b):
        # encoding to another dimension:
        x_past = self.exp_x_past(x_past)
        # x_bar = self.exp_x_bar(x_bar)
        x = self.exp_x(x)
        y = self.exp_y(y)
        # y_bar = self.exp_y_bar(y_bar)

        # start updating
        # x_md = (1.0-self.emu_beta)*x_bar + self.emu_beta*x
        x_md = x
        # print(x_md, self.emu_beta)
        # input()

        # need to check if we want to wrap this with a linear layer
        #       current setting seems better for the theoretical part
        x_delta = self.emu_theta*(x - x_past) + x
        x_delta = self.emu_gamma * self.lin_1(torch.matmul(A,x_delta))
        # y_new = self.proj_pos(y + x_delta - b)
        y_new = y + x_delta - b 
        y_new = self.shift_y(y_new)


        x_new = x - self.emu_eta * (self.lin_2(torch.matmul(Q,x_md)) + c + self.lin_3(torch.matmul(AT,y)) )

        # relu to emulate proj to l<x<u, looks weird but aligns exactly
        x_new = self.shift_x(x_new)

        # x_bar = (1.0-self.emu_beta)*x_bar + self.emu_beta*x_new
        # y_bar = (1.0-self.emu_beta)*y_bar + self.emu_beta*y_new


        return x_new,x_bar,y_new,y_bar

class r_primal(torch.nn.Module):
    
    def __init__(self):
        super(r_primal,self).__init__()
        # self.act = proj_y(1)
        self.relu = nn.ReLU()
        
    def forward(self,A,b,c,x,Iy, il, iu, l, u):
        Ax = torch.sparse.mm(A,x)
        # lower_variable_violation[tx] = max(variable_lower_bound[tx] - primal_vec[tx], 0.0)
        # upper_variable_violation[tx] = max(primal_vec[tx] - variable_upper_bound[tx], 0.0)
        #     constraint_violation[tx] = right_hand_side[tx] - activities[tx]
        #     constraint_violation[tx] = max(right_hand_side[tx] - activities[tx], 0.0)
        # cons_vio = Ax-b.unsqueeze(-1)
        cons_vio = Ax - b.unsqueeze(-1)
        cons_vio = cons_vio + torch.mul(self.relu(-cons_vio),Iy)
        var_vio = torch.mul(self.relu(l-x), il) + torch.mul(self.relu(x-u), iu)
        part_2 = torch.linalg.vector_norm(torch.cat((var_vio,cons_vio),0),float('inf'))
        # part_2 = torch.linalg.vector_norm(cons_vio,float('inf'))

        # part_2 = torch.linalg.vector_norm(self.act(Ax-b.unsqueeze(-1),Iy),float('inf'))
        
        # part_3 = 1.0 + torch.max(torch.linalg.vector_norm(Ax,float('inf')),torch.linalg.vector_norm(b,float('inf')))
        part_3 = 1.0 + torch.linalg.vector_norm(b,float('inf'))
        # torch.linalg.vector_norm(a,float('inf'))
        return part_2/part_3


class r_primal_old(torch.nn.Module):
    
    def __init__(self):
        super(r_primal,self).__init__()
        self.act = proj_y(1)
        # self.act = nn.ReLU()
        
    def forward(self,A,b,c,x,Iy):
        Ax = torch.sparse.mm(A,x)
        part_2 = torch.linalg.vector_norm(self.act(Ax-b.unsqueeze(-1),Iy),float('inf'))
        
        # part_3 = 1.0 + torch.max(torch.linalg.vector_norm(part_1,float('inf')),torch.linalg.vector_norm(b,float('inf')))
        part_3 = 1.0 + torch.linalg.vector_norm(b,float('inf'))
        return part_2/part_3


class r_dual(torch.nn.Module):
    
    def __init__(self):
        super(r_dual,self).__init__()
        self.yproj = proj_y(1)
        
    def forward(self,Q,AT,b,c,x,y,Iy, il, iu, l, u):
        Qx = torch.sparse.mm(Q,x) 
        ATy = torch.sparse.mm(AT,y) 
        
        primal_grad = Qx + ATy + c.unsqueeze(-1)
        
        # primal_grad = torch.abs(primal_grad - self.xproj(primal_grad, il, iu, l, u))
        # primal_grad = primal_grad - torch.mul(torch.max(primal_grad, l),il) + torch.mul(torch.min(primal_grad, u), iu)
        # dual_vio = self.yproj(-y, Iy)
        # quit()
        # print(top_part)
        # part_1 = torch.sparse.mm(Q,x) + torch.sparse.mm(AT,y) + c
        # top_part = torch.max(torch.linalg.vector_norm(primal_grad,float('inf')), torch.linalg.vector_norm(dual_vio,float('inf')))
        top_part = torch.linalg.vector_norm(primal_grad,float('inf'))
        # quit()
        
        # bot_part = 1.0 + torch.max(torch.linalg.vector_norm(Qx,float('inf')),torch.max(torch.linalg.vector_norm(ATy,float('inf')),torch.linalg.vector_norm(c,float('inf'))))
        bot_part = 1.0 + torch.linalg.vector_norm(c,float('inf'))
        # torch.linalg.vector_norm(a,float('inf'))
        return top_part/bot_part



class r_dual_inf_pdqp(torch.nn.Module):
    
    def __init__(self):
        super(r_dual_inf_pdqp,self).__init__()
        self.yproj = proj_y(1)
        self.act = nn.ReLU()
        
    def forward(self,Q,AT,b,c,x,y,Iy, il, iu, l, u):
        # print(x[0])
        # print(y[0])

        Qx = torch.sparse.mm(Q,x) 
        ATy = torch.sparse.mm(AT,y) 
        
        primal_grad = c.unsqueeze(-1) + ATy + Qx
        
        RCV = primal_grad - torch.mul(self.act(primal_grad), il) + torch.mul(-self.act(-primal_grad), iu)
        DR = torch.mul(self.act(-y), Iy)
        
        # primal_grad = torch.abs(primal_grad - self.xproj(primal_grad, il, iu, l, u))
        # primal_grad = primal_grad - torch.mul(torch.max(primal_grad, l),il) + torch.mul(torch.min(primal_grad, u), iu)
        # dual_vio = self.yproj(-y, Iy)
        # quit()
        # print(top_part)
        # part_1 = torch.sparse.mm(Q,x) + torch.sparse.mm(AT,y) + c
        # top_part = torch.max(torch.linalg.vector_norm(primal_grad,float('inf')), torch.linalg.vector_norm(dual_vio,float('inf')))
        top_part = torch.linalg.vector_norm(torch.cat((RCV, DR),0),float('inf'))
        
        # bot_part = 1.0 + torch.max(torch.linalg.vector_norm(Qx,float('inf')),torch.max(torch.linalg.vector_norm(ATy,float('inf')),torch.linalg.vector_norm(c,float('inf'))))
        bot_part = 1.0 + torch.linalg.vector_norm(c,float('inf'))
        res = top_part/bot_part
        # print(RCV[0].item(),DR[0].item(),' ---   ',y[0].item(),x[0].item(),Qx[0].item(),ATy[0].item(),primal_grad[0].item())
        # print('!!!!!!!!!!!!!!!!1        ',res)
        return res

class r_gap(torch.nn.Module):
    
    def __init__(self):
        super(r_gap,self).__init__()
        self.act = nn.ReLU()
        
    def forward(self,Q,A,AT,b,c,x,y, il, iu, l, u):
        
        xt = torch.transpose(x,0,1)
        qx = torch.matmul(Q,x)
        quad_term = torch.matmul(xt,qx)
        # print(c.shape,x.shape)
        lin_term = torch.matmul(c,x)
        vio_term = torch.matmul(b,y)

        # compute RC
        ATy = torch.sparse.mm(AT,y) 
        primal_grad = c.unsqueeze(-1) - ATy + qx
        
        RC = torch.mul(self.act(primal_grad), il) + torch.mul(-self.act(-primal_grad), iu)
        
        rc_contribution_lower = torch.where(RC>0,l,0.0)
        rc_contribution_lower = torch.mul(RC,rc_contribution_lower)
        rc_contribution_lower = torch.sum(rc_contribution_lower)
        
        rc_contribution_upper = torch.where(RC<0,u,0.0)
        rc_contribution_upper = torch.mul(RC,rc_contribution_upper)
        rc_contribution_upper = torch.sum(rc_contribution_upper)
        
        rc_contribution = rc_contribution_upper + rc_contribution_lower
        rc_contribution_upper = None
        rc_contribution_lower = None
        
        top_part = torch.abs(quad_term + lin_term - vio_term - rc_contribution)

        # top_part = torch.abs(quad_term + lin_term + vio_term)
        print('Obj abs gap: ',top_part)
        bot_part = 1.0 + torch.max(torch.abs(vio_term - 0.5*quad_term ),torch.abs(0.5*quad_term + lin_term))
        print(f'Primal: {lin_term.item()+0.5*quad_term.item()} - Dual: {vio_term.item()-0.5*quad_term.item()}=  {(quad_term + lin_term - vio_term).item()}       ------  RCC: {rc_contribution}    1/2xTQx: {0.5*quad_term.item()}')
        
        
        
        # rc_contribution = torch.where(RC>0,l,u)
        # rc_contribution = torch.mul(RC,rc_contribution)
        # rc_contribution = torch.sum(rc_contribution)
        # Axb = torch.mul(y,torch.sparse.mm(A,x)- b.unsqueeze(-1))
        # Axb = torch.norm(Axb,1)
        # top_part = torch.abs(quad_term + lin_term - vio_term - rc_contribution) 
        # print(f'abs gap: {top_part}')
        
        
        
        
        # bot_part=torch.tensor(8176.4)
        # bot_part = 1.0 + torch.max(torch.norm(Q,float('inf')) , torch.max(torch.linalg.vector_norm(c,float('inf')) + torch.linalg.vector_norm(b,float('inf'))))
        # bot_part = torch.tensor(1e+4)
        # torch.linalg.vector_norm(a,float('inf'))
        
        # print(f'Primal: {lin_term.item()+0.5*quad_term.item()} - Dual: {vio_term.item()-0.5*quad_term.item()}=  {(quad_term + lin_term - vio_term).item()}       ------  RCC: {rc_contribution} ')
        
        
        return top_part/bot_part
        # return top_part

class relKKT(torch.nn.Module):
    
    def __init__(self,use_dual = True):
        super(relKKT,self).__init__()
        self.rpm = r_primal()
        # self.rdl = r_dual()
        self.rdl = r_dual_inf_pdqp()
        self.rgp = r_gap()
        self.use_dual = use_dual

    def forward(self,Q,A,AT,b,c,x,y,Iy, il, iu, l, u):
        # return torch.max(torch.max(self.rpm(A,b,c,x),self.rdl(Q,AT,b,c,x,y)),self.rgp(Q,A,AT,b,c,x,y))
        # t1 = self.rpm(A,b,c,x,Iy)
        t1 = self.rpm(A,b,c,x,Iy, il, iu, l, u)
        t3 = self.rgp(Q,A,AT,b,c,x,y, il, iu, l, u)

        if self.use_dual:
            t2 = self.rdl(Q,AT,b,c,x,y,Iy, il, iu, l, u)
            # if abs(t2.item())>1e-4: 
            #     res = torch.max(t1,torch.max(t2,t3))
            # else:
            #     print(f'0 dual res, skip it ')
            #     res = torch.max(t1,t3)
            # res = torch.max(t1,torch.max(t2,t3))
        else:
            t2 = torch.tensor(0.0)
            # res = torch.max(t1,t3)
        res = t1+t2+t3
        # res = t3
        # print(t1.item())
        # print(t2.item())
        # print(t3.item())
        return res,t1,t2,t3











class r_primal_l1(torch.nn.Module):
    
    def __init__(self):
        super(r_primal_l1,self).__init__()
        self.act = proj_y(1)
        
    def forward(self,A,b,c,x,Iy):
        part_1 = torch.sparse.mm(A,x)-b.unsqueeze(-1)
        part_1 = self.act(part_1,Iy)
        part_1 = torch.linalg.vector_norm(part_1,1)
        part_3 = 1.0 + torch.linalg.vector_norm(b,1)
        return part_1/part_3

class r_dual_l1(torch.nn.Module):
    
    def __init__(self):
        super(r_dual_l1,self).__init__()
        
    def forward(self,Q,AT,b,c,x,y):
        Qx = torch.sparse.mm(Q,x) 
        ATy = torch.sparse.mm(AT,y) 
        
        
        top_part = torch.linalg.vector_norm(Qx + ATy + c.unsqueeze(-1),1)
        
        # bot_part = 1.0 + torch.linalg.vector_norm(c,1)
        bot_part = 1E+4 + torch.linalg.vector_norm(c,1)
        return top_part/bot_part

class r_gap_l1(torch.nn.Module):
    
    def __init__(self):
        super(r_gap_l1,self).__init__()
        
    def forward(self,Q,A,AT,b,c,x,y):
        
        xt = torch.transpose(x,0,1)
        qx = torch.matmul(Q,x)
        quad_term = torch.matmul(xt,qx)
        lin_term = torch.matmul(c,x)
        vio_term = torch.matmul(b,y)
        top_part = torch.abs(quad_term + lin_term + vio_term)
        
        return top_part

class relKKT_l1(torch.nn.Module):
    
    def __init__(self):
        super(relKKT_l1,self).__init__()
        self.rpm = r_primal_l1()
        self.rdl = r_dual_l1()
        self.rgp = r_gap_l1()

    def forward(self,Q,A,AT,b,c,x,y,Iy):
        # return torch.max(torch.max(self.rpm(A,b,c,x),self.rdl(Q,AT,b,c,x,y)),self.rgp(Q,A,AT,b,c,x,y))
        t1 = self.rpm(A,b,c,x,Iy)
        t2 = self.rdl(Q,AT,b,c,x,y)
        t3 =self.rgp(Q,A,AT,b,c,x,y)

        # print(t1.item(),end=', ')
        # print(t2.item(),end=', ')
        # print(t3.item())
        res = t1 + t2 + t3
        # res = torch.max(t1,torch.max(t2,t3))
        # res = t1
        # res = self.rpm(A,b,c,x,Iy)+self.rdl(Q,AT,b,c,x,y)
        return res













class r_primal_real(torch.nn.Module):
    
    def __init__(self):
        super(r_primal_real,self).__init__()
        self.act = nn.ReLU()
        
    def forward(self,A,b,c,x,Iy, il, iu, l, u):
        Ax = torch.sparse.mm(A,x)
        var_vio = torch.mul(self.act(l-x), il) + torch.mul(self.act(x-u), iu)
        cons_vio = b.unsqueeze(-1) - Ax
        cons_vio = cons_vio + torch.mul(self.act(-cons_vio),Iy)
        part_2 = torch.linalg.vector_norm(torch.cat((var_vio,cons_vio),0),float('inf'))
        part_3 = 1.0 + torch.max(torch.linalg.vector_norm(Ax,float('inf')),torch.linalg.vector_norm(b,float('inf')))
        return part_2/part_3

class r_dual_inf_real(torch.nn.Module):
    
    def __init__(self):
        super(r_dual_inf_real,self).__init__()
        self.yproj = proj_y(1)
        self.act = nn.ReLU()
        
    def forward(self,Q,AT,b,c,x,y,Iy, il, iu, l, u):

        Qx = torch.sparse.mm(Q,x) 
        ATy = torch.sparse.mm(AT,y) 
        
        primal_grad = c.unsqueeze(-1) - ATy + Qx
        
        RCV = primal_grad - torch.mul(self.act(primal_grad), il) - torch.mul(-self.act(-primal_grad), iu)
        DR = torch.mul(self.act(-y), Iy)
        
        top_part = torch.linalg.vector_norm(torch.cat((RCV, DR),0),float('inf'))
        
        bot_part = 1.0 + torch.max(torch.linalg.vector_norm(Qx,float('inf')),torch.max(torch.linalg.vector_norm(ATy,float('inf')),torch.linalg.vector_norm(c,float('inf'))))
        res = top_part/bot_part
        return res

class relKKT_real(torch.nn.Module):
    
    def __init__(self):
        super(relKKT_real,self).__init__()
        self.rpm = r_primal_real()
        self.rdl = r_dual_inf_real()
        self.rgp = r_gap()

    # def forward(self,Q,A,AT,b,c,x,y,Iy, il, iu, l, u):

    def forward(self,Q,A,AT,b,c,x,y,Iy, il, iu, l, u, vscale,cscale,cons_scale):
        
        x_unscaled = torch.mm(torch.div(x,vscale),cons_scale)
        y_unscaled = torch.mm(torch.div(y,cscale),cons_scale)

        t1 = self.rpm(A,b,c,x_unscaled,Iy, il, iu, l, u)
        t2 = self.rdl(Q,AT,b,c,x_unscaled,y_unscaled,Iy, il, iu, l, u)
        t3 = self.rgp(Q,A,AT,b,c,x_unscaled,y_unscaled, il, iu, l, u)
        res = torch.max(t1,torch.max(t2,t3))
        return res,t1,t2,t3











class r_primal_general(torch.nn.Module):
    
    def __init__(self,mode = 2,norm=False):
        super(r_primal_general,self).__init__()
        # self.act = proj_y(1)
        self.relu = nn.ReLU()
        self.mode = mode
        self.norm = norm
        
    def forward(self,A,b,c,x,Iy, il, iu, l, u):
        # diff_l = self.relu(l-x)
        # print(diff_l)
        
        # diff_l = torch.mul(self.relu(l-x), il) + torch.mul(self.relu(x-u), iu)
        
        # for i in range(x.shape[0]):
        #     if (diff_l[i]>0):
        #         print(f'l:{l[i].item()}      x:{x[i].item()}      u:{u[i].item()}       vio:{diff_l[i].item()}')
        #         input()
        # input()

        Ax = torch.sparse.mm(A,x)
        cons_vio = b.unsqueeze(-1) - Ax
        cons_vio = cons_vio + torch.mul(self.relu(-cons_vio),Iy)
        var_vio = torch.mul(self.relu(l-x), il) + torch.mul(self.relu(x-u), iu)
        part_2 = torch.linalg.vector_norm(torch.cat((var_vio,cons_vio),0),self.mode)
        if self.norm:
            part_3 = 1.0 + torch.max(torch.linalg.vector_norm(Ax,self.mode),torch.linalg.vector_norm(b,self.mode))
        else:
            part_3 = 1.0 + torch.linalg.vector_norm(b,self.mode)
            # part_3 = 1.0 
        res = part_2/part_3
        print(f'var_vio: {torch.norm(var_vio,2).item()}   cons_vio: {torch.norm(cons_vio,2).item()}')
        # print(part_2/(1.0 + torch.max(torch.linalg.vector_norm(Ax,self.mode),torch.linalg.vector_norm(b,self.mode))))
        return res

class r_dual_general(torch.nn.Module):
    
    def __init__(self,mode=2,norm=False):
        super(r_dual_general,self).__init__()
        self.yproj = proj_y(1)
        self.act = nn.ReLU()
        self.mode = mode
        self.norm = norm
        
    def forward(self,Q,AT,b,c,x,y,Iy, il, iu, l, u):

        Qx = torch.sparse.mm(Q,x) 
        ATy = torch.sparse.mm(AT,y) 
        

        primal_grad = c.unsqueeze(-1) - ATy + Qx
        
        RCV = primal_grad - torch.mul(self.act(primal_grad), il) - torch.mul(-self.act(-primal_grad), iu)
        DR = torch.mul(self.act(-y), Iy)
        
        RCV_norm= torch.norm(RCV,self.mode)
        DR_norm= torch.norm(DR,self.mode)
        print(f'RCV norm: {RCV_norm}     DR norm: {DR_norm}')

        top_part = torch.linalg.vector_norm(torch.cat((RCV, DR),0),self.mode)
        
        if self.norm:
            bot_part = 1.0 + torch.max(torch.linalg.vector_norm(Qx,self.mode),torch.max(torch.linalg.vector_norm(ATy,self.mode),torch.linalg.vector_norm(c,2)))
        else:
            bot_part = 1.0 + torch.linalg.vector_norm(c,self.mode)
            # bot_part = 1.0 
        res = top_part/bot_part
        return res
        
        
        

        
class r_gap_general(torch.nn.Module):
    
    def __init__(self,mode=2,eta_opt=1e+6,norm=False):
        super(r_gap_general,self).__init__()
        self.mode = mode
        self.act = nn.ReLU()
        self.eta_opt = eta_opt
        self.norm = norm
        
        
    def forward(self,Q,A,AT,b,c,x,y,Iy, il, iu,l,u):
        xt = torch.transpose(x,0,1)
        qx = torch.matmul(Q,x)
        quad_term = torch.matmul(xt,qx)
        lin_term = torch.matmul(c,x)
        vio_term = torch.matmul(b,y)
        
        # compute RC
        ATy = torch.sparse.mm(AT,y) 
        primal_grad = c.unsqueeze(-1) - ATy + qx
        
        
        RC = torch.mul(self.act(primal_grad), il) + torch.mul(-self.act(-primal_grad), iu)
        rc_contribution = torch.where(RC>0,l,u)
        # for i in range(rc_contribution.shape[0]):
        #     if rc_contribution[i].item()<0:
        #         print(rc_contribution[i].item(),RC[i].item(),l[i].item(),u[i].item())
        # quit()
        rc_contribution = torch.mul(RC,rc_contribution)
        rc_contribution = torch.sum(rc_contribution)
        # rc_contribution = torch.norm(rc_contribution,1)
        

        Axb = torch.mul(y,torch.sparse.mm(A,x)- b.unsqueeze(-1))
        # Axb = torch.mul(Axb,(1.0-Iy))
        
        # print(y)
        # input()
        # print(torch.sparse.mm(A,x)- b.unsqueeze(-1))
        # input()
        Axb = torch.norm(Axb,1)



        # Ayc = torch.mul(x,ATy - c.unsqueeze(-1))
        # Ayc = torch.norm(Ayc,1)




        # top_part = torch.abs(quad_term + lin_term - vio_term - rc_contribution)
        # top_part = torch.abs(quad_term + lin_term - vio_term )+ rc_contribution + Axb
        top_part = torch.abs(quad_term + lin_term - vio_term - rc_contribution) 
        print(f'abs gap: {top_part}')



        
        print(f'Primal: {lin_term.item()+0.5*quad_term.item()} - Dual: {vio_term.item()-0.5*quad_term.item()}=  {(quad_term + lin_term - vio_term).item()}       ------  RCC: {rc_contribution}    1/2xTQx: {0.5*quad_term.item()}\n           yT(Ax-b): {Axb.item()}')
        # for i in range(y.shape[0]):
        #     if y[i].item()*b[i].item()<0:
        #         print(y[i].item()*b[i].item(),y[i].item(),b[i].item(),i)
        #         input()
        # quit()
        # return top_part/self.eta_opt
        

        # bot_part = 1.0 + torch.norm(Q,self.mode)
        # bot_part = 1.0 + torch.max(torch.abs(vio_term - 0.5*quad_term ),torch.abs(0.5*quad_term + lin_term))
        bot_part = self.eta_opt
        if self.eta_opt is None:
            bot_part = 1.0 + torch.max(torch.abs(vio_term - 0.5*quad_term ),torch.abs(0.5*quad_term + lin_term))
        
        

        return top_part/bot_part
    






        
class r_CS_general(torch.nn.Module):
    
    def __init__(self,mode=2,eta_opt=1e+6):
        super(r_CS_general,self).__init__()
        self.act = nn.ReLU()
        self.eta_opt = eta_opt
        
    def forward(self,Q,A,AT,b,c,x,y,Iy, il, iu,l,u):
        # consider ocnstraint violation
        Axb = torch.mul(y,torch.sparse.mm(A,x)- b.unsqueeze(-1))
        Axb = torch.mul(Axb,Iy)
        Axb = torch.norm(Axb,1)
        # consider bound violations
        ATy = torch.sparse.mm(AT,y) 
        qx = torch.matmul(Q,x)
        primal_grad = c.unsqueeze(-1) - ATy + qx


        # RC = torch.mul(self.act(primal_grad), il) + torch.mul(-self.act(-primal_grad), iu)
        # rc_contribution = torch.where(RC>0,l,u)
        # rc_contribution = torch.mul(RC,rc_contribution)
        # rc_contribution = torch.norm(rc_contribution,1)


        # var_vio = torch.mul(self.act(l-x), il) + torch.mul(self.act(x-u), iu)
        # uz = torch.where(iu!=0,u,99999)
        # vv = torch.where(iu!=0,x-u,0)
        # vz = torch.where(il!=0,l-x,0)
        # vio_info = torch.cat((l,x,uz,vv,vz),1)
        # for i in vio_info:
        #     if i[-1]>0 or i[-2]>0:
        #         print(i)
        #         input()
        lb = torch.mul(self.act(primal_grad), il)
        lb = torch.norm(torch.mul((x-l),lb),1)
        ub = torch.mul(self.act(-primal_grad), iu)
        ub = torch.norm(torch.mul((u-x),ub),1)
        # print(f'primal grad: {primal_grad}')
        # print(f'LB: {lb}')
        # print(f'UB: {ub}')
        # input()
        rc_contribution = lb+ub
        # add relu
        # return Axb
        return (Axb+rc_contribution)/self.eta_opt


class relKKT_general(torch.nn.Module):
    
    def __init__(self,mode=2,eta_opt = 1e+6,norm=False):
        print(f'!!!  relKKT using {mode} norm, with estimated optimum of {eta_opt}  !!!')
        super(relKKT_general,self).__init__()
        if mode == 'linf':
            mode = float('inf')
        elif '2' in mode:
            mode = 2
        elif '1' in mode:
            mode = 1
        self.rpm = r_primal_general(mode,norm)
        self.rdl = r_dual_general(mode,norm)
        self.rgp = r_gap_general(mode,eta_opt,norm)
        # self.rgp = r_CS_general(mode,eta_opt)
        

    def forward(self,Q,A,AT,b,c,x,y,Iy, il, iu, l, u, vscale,cscale,cons_scale):
        
        # # Unscale iterates. 
        # x = x./variable_rescaling
        # x = x.*const_scale

        x_unscaled = torch.mm(torch.div(x,vscale),cons_scale)
        y_unscaled = torch.mm(torch.div(y,cscale),cons_scale)



        t1 = self.rpm(A,b,c,x_unscaled,Iy, il, iu, l, u)
        t2 = self.rdl(Q,AT,b,c,x_unscaled,y_unscaled,Iy, il, iu, l, u)
        t3 =self.rgp(Q,A,AT,b,c,x_unscaled,y_unscaled,Iy, il, iu,l,u)

        # res = t1+t2+t3
        # res = t1+t2
        # res = t2+t3
        res = torch.max(t3,torch.max(t2,t1))
        return res,t1,t2,t3