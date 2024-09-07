import gurobipy as gp

base = '/home/lxyang/git/pdqpnet/ins/qplib_qps/'
flist = ['co5_1.mps','cont5.mps','dist3.mps','dtoc.mps','rqp1.mps']
for fdir in flist:
    fdir = base+fdir
    ftar = fdir.split('/')[-1].replace('.mps','.log')

    model = gp.read(fdir)
    model.Params.LogFile = ftar
    savetar = base+fdir.split('/')[-1].replace('.mps','_nqc.mps')
    vs = model.getVars()
    cs = model.getQConstrs()

    for c in cs:
        model.remove(c)
    model.update()
    model.write(savetar)
    # model.optimize()