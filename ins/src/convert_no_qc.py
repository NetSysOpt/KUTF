import gurobipy as gp

base = '/home/lxyang/git/pdqpnet/ins/qplib/'
flist = ['QPLIB_3861.lp','QPLIB_3871.lp']

for fdir in flist:
    fdir = base+fdir
    ftar = fdir.split('/')[-1].replace('.lp','.log')

    model = gp.read(fdir)
    model.Params.LogFile = ftar
    savetar = base.replace('qplib','qplib_qps')+fdir.split('/')[-1].replace('.lp','.mps')
    vs = model.getVars()
    cs = model.getQConstrs()

    removed=0
    for v in vs:
        if v.vType=='B':
            v.vType=gp.GRB.CONTINUOUS
    removed=0
    for c in cs:
        model.remove(c)
        removed+=1
    print(f'removed {removed} QCs')
    model.update()
    model.write(savetar)
    model.optimize()