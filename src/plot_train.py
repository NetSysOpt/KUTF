import matplotlib.pyplot as plt


us_loss=[[],[]]
sl_loss=[[],[]]

mode = 'qplib_8845'
loss_log = open(f'../logs/train_{mode}_supervised.log','r')
# loss_log = open(f'../logs/train_{mode}.log','r')
for line in loss_log:
    lst = line.replace('\n','').split(' ')
    lst = [float(x) for x in lst]
    print(lst)
    sl_loss[0].append(lst[1])
    sl_loss[1].append(lst[2])
loss_log.close()
plt.plot(us_loss[0],label='unsupervised trianing', linestyle='-',marker=',',c='red')
plt.plot(us_loss[1],label='unsupervised valid', linestyle='--',marker=',',c='red')

if False:
    loss_log = open(f'../logs/train_{mode}.log','r')
    for line in loss_log:
        lst = line.replace('\n','').split(' ')
        lst = [float(x) for x in lst]
        print(lst)
        us_loss[0].append(lst[1])
        us_loss[1].append(lst[2])
    loss_log.close()

    plt.plot(sl_loss[0],label='supervised trianing', linestyle='-',marker=',',c='blue')
    plt.plot(sl_loss[1],label='supervised valid', linestyle='--',marker=',',c='blue')
plt.legend()
# plt.ylim(0.,1.)
plt.ylabel('KKT err')
plt.xlabel('epoch')
plt.yscale('log')
plt.savefig('../plots/training/scs.png')


if False:

    # plot distances
    import os
    ident = '3913'
    logs = os.listdir('../plots/distance/logs')
    logs=[x for x in logs if ident in x]
    print(logs)

    ori_stat = [[],[],[],[],[],[],[]]
    us_stat = [[],[],[],[],[],[],[]]
    for l in logs:
        if 'ori' in l:
            # supervised
            f=open(f'../plots/distance/logs/{l}','r')
            for line in f:
                lst = line.replace('\n','').split(' ')
                lst = [float(x) for x in lst]
                print(lst)
                ori_stat[0].append(lst[1])
                ori_stat[1].append(lst[2])
                ori_stat[2].append(lst[3])
                ori_stat[3].append(lst[4])
                ori_stat[4].append(lst[5])
                ori_stat[5].append(lst[6])
                ori_stat[6].append(lst[6]+lst[5])
            f.close()
        else:
            # supervised
            f=open(f'../plots/distance/logs/{l}','r')
            for line in f:
                lst = line.replace('\n','').split(' ')
                lst = [float(x) for x in lst]
                print(lst)
                us_stat[0].append(lst[1])
                us_stat[1].append(lst[2])
                us_stat[2].append(lst[3])
                us_stat[3].append(lst[4])
                us_stat[4].append(lst[5])
                us_stat[5].append(lst[6])
                us_stat[6].append(lst[6]+lst[5])
            f.close()

    us_stat[4] = [x/max(us_stat[4]) for x in us_stat[4]]
    us_stat[5] = [x/max(us_stat[5]) for x in us_stat[5]]
    us_stat[6] = [x/max(us_stat[6]) for x in us_stat[6]]
    ori_stat[4] = [x/max(ori_stat[4]) for x in ori_stat[4]]
    ori_stat[5] = [x/max(ori_stat[5]) for x in ori_stat[5]]
    ori_stat[6] = [x/max(ori_stat[6]) for x in ori_stat[6]]

    plt.clf()
    plt.scatter(ori_stat[6],ori_stat[0],label='supervised trianing',c='blue')
    plt.scatter(us_stat[6],us_stat[0],label='unsupervised trianing',c='red')
    plt.legend()
    plt.ylabel('KKT err')
    plt.xlabel('distance')
    plt.yscale('log')
    plt.savefig(f'../plots/training/{ident}_total_err.png')

    plt.clf()
    plt.scatter(ori_stat[6],ori_stat[1],label='supervised trianing',c='blue')
    plt.scatter(us_stat[6],us_stat[1],label='unsupervised trianing',c='red')
    plt.legend()
    plt.ylabel('KKT err')
    plt.xlabel('distance')
    # plt.yscale('log')
    plt.savefig(f'../plots/training/{ident}_pres.png')

    plt.clf()
    plt.scatter(ori_stat[6],ori_stat[2],label='supervised trianing',c='blue')
    plt.scatter(us_stat[6],us_stat[2],label='unsupervised trianing',c='red')
    plt.legend()
    plt.ylabel('KKT err')
    plt.xlabel('distance')
    # plt.yscale('log')
    plt.savefig(f'../plots/training/{ident}_dres.png')

    plt.clf()
    plt.scatter(ori_stat[6],ori_stat[3],label='supervised trianing',c='blue')
    plt.scatter(us_stat[6],us_stat[3],label='unsupervised trianing',c='red')
    plt.legend()
    plt.ylabel('KKT err')
    plt.xlabel('distance')
    # plt.yscale('log')
    plt.savefig(f'../plots/training/{ident}_gap.png')