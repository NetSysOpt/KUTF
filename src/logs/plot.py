import matplotlib.pyplot as plt
import gzip
import json

def read_json(fnm):
    # Opening JSON file
    f = gzip.open(fnm)
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    # Closing file
    f.close()
    return data



fnm_fx = 'CONT-200_full_log'
fnm_fx = 'QPLIB_8785_489_full_log'
fnm_fx = 'QPLIB_8906_213_full_log'
temp_folder = 'warmstart'

def plot_primal(fnm):
    tname = f'/home/lxyang/git/pdqpnet/src/logs/{temp_folder}/{fnm}.json.gz'

    logs = read_json(tname)
    for key in logs:
        print(key)
    iters = logs['iteration_stats']
    x=[]
    y=[]
    for it in iters:
        # print(it)
        it_num = int(it['iteration_number'])
        primal = float(it['convergence_information'][0]['primal_objective'])
        x.append(it_num)
        y.append(primal)

    plt.plot(x,y,label='ws')



    # tname = f'/home/lxyang/git/pdqpnet/src/logs/ori/{fnm}.json.gz'
    tname = f'/home/lxyang/git/pdqpnet/logs/{fnm}.json.gz'

    logs = read_json(tname)
    for key in logs:
        print(key)
    iters = logs['iteration_stats']
    x=[]
    y=[]
    for it in iters:
        # print(it)
        it_num = int(it['iteration_number'])
        primal = float(it['convergence_information'][0]['primal_objective'])
        x.append(it_num)
        y.append(primal)

    plt.plot(x,y,label='ori')
    plt.legend()
    plt.savefig('./plots/primal.png')




def plot_res(fnm):
    x=[]
    y=[]
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    tname = f'/home/lxyang/git/pdqpnet/src/logs/{temp_folder}/{fnm}.json.gz'
    # tname = f'/home/lxyang/git/pdqpnet/logs/{fnm}.json.gz'

    logs = read_json(tname)
    for key in logs:
        print(key)
    iters = logs['iteration_stats']
    for it in iters:
        # print(it)
        it_num = int(it['iteration_number'])
        primal = float(it['convergence_information'][0]['relative_l_inf_primal_residual'])
        x.append(it_num)
        y.append(primal)
        it_num = int(it['iteration_number'])
        primal = float(it['convergence_information'][0]['relative_l_inf_dual_residual'])
        x1.append(it_num)
        y1.append(primal)
        it_num = int(it['iteration_number'])
        primal = float(it['convergence_information'][0]['relative_optimality_gap'])
        x2.append(it_num)
        y2.append(primal)




    qx=[]
    qy=[]
    qx1=[]
    qy1=[]
    qx2=[]
    qy2=[]
    # tname = f'/home/lxyang/git/pdqpnet/src/logs/ori/{fnm}.json.gz'
    tname = f'/home/lxyang/git/pdqpnet/logs/{fnm}.json.gz'


    logs = read_json(tname)
    for key in logs:
        print(key)
    iters = logs['iteration_stats']
    for it in iters:
        it_num = int(it['iteration_number'])
        primal = float(it['convergence_information'][0]['relative_l_inf_primal_residual'])
        qx.append(it_num)
        qy.append(primal)
        it_num = int(it['iteration_number'])
        primal = float(it['convergence_information'][0]['relative_l_inf_dual_residual'])
        qx1.append(it_num)
        qy1.append(primal)
        it_num = int(it['iteration_number'])
        primal = float(it['convergence_information'][0]['relative_optimality_gap'])
        qx2.append(it_num)
        qy2.append(primal)

    plt.clf()
    plt.plot(x,y,label='ws')
    plt.plot(qx,qy,label='ori')
    plt.legend()
    plt.ylim(0.0,1.0)
    plt.savefig('./plots/primal_res.png')
    plt.clf()
    
    plt.plot(x1,y1,label='ws')
    plt.plot(qx1,qy1,label='ori')
    plt.legend()
    plt.savefig('./plots/dual_res.png')
    plt.clf()

    plt.plot(x2,y2,label='ws')
    plt.plot(qx2,qy2,label='ori')
    plt.legend()
    plt.savefig('./plots/gap.png')
    plt.clf()

plot_primal(fnm_fx)
plot_res(fnm_fx)