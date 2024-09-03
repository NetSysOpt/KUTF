import gurobipy as gp

base = '/home/lxyang/git/pdqpnet/ins/qplib_qps/'
flist = ['co5_1_nqc.mps','cont5_nqc.mps','dist3_nqc.mps','dtoc_nqc.mps','rqp1_nqc.mps']
for fdir in flist:
    fdir = base+fdir
    ftar = fdir.split('/')[-1].replace('.mps','.log')

    model = gp.read(fdir)
    model.Params.LogFile = ftar
    # savetar = base+fdir.split('/')[-1].replace('.mps','_nqc.mps')
    # model.write(savetar)
    # vs = model.getVars()
    # cs = model.getQConstrs()

    # for c in cs:
    #     model.remove(c)

    model.optimize()