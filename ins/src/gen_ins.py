import random
random.seed(0)
import os
import gurobipy as gp
import numpy as np
import random

def gen_cont_uniform_pert(pert_range=0.1, ori_ins = '../train/CONT-201.QPS', indx = 0):
    uni_pert = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
    print(uni_pert)
    ff = open(ori_ins,'r')
    mode = ''
    tar_ins = ori_ins.split('/')[-1].replace('.QPS',f'_{indx}.QPS')
    out_file = f'../gen_train_cont/{tar_ins}'
    fout = open(out_file,'w')
    for line in ff:
        if ' '!=line[0]:
            fout.write(line)
            if 'ROWS' in line:
                mode = 'ROWS'
            elif 'COLUMNS' in line:
                mode = 'COLUMNS'
            elif 'RHS' in line:
                mode = 'RHS'
            elif 'RANGES' in line:
                mode = 'RANGES'
            elif 'BOUNDS' in line:
                mode = 'BOUNDS'
            elif 'QUADOBJ' in line:
                mode = 'QUADOBJ'
            else:
                continue
        else:
            if 'ROWS' == mode:
                fout.write(line)
            elif 'COLUMNS' == mode:
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                coeff1 = lst[2]
                coeff2 = None
                if len(lst)>3:
                    coeff2 = lst[4]
                new_coeff1 = float(coeff1)*uni_pert
                line = line.replace(coeff1,str(new_coeff1))
                if coeff2 is not None and coeff2!=coeff1:
                    new_coeff2 = float(coeff2)*uni_pert
                    line = line.replace(coeff2,str(new_coeff2))
                fout.write(line)
            elif 'RHS' == mode:
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                rhs = lst[2]
                if len(lst)>3:
                    print('invalid input, quit',lst)
                    quit()
                new_rhs = float(rhs)*uni_pert
                line=line.replace(rhs,str(new_rhs))
                fout.write(line)
            elif 'RANGES' == mode:
                print(line)
            elif 'BOUNDS' == mode:
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                rhs = lst[3]
                if len(lst)>4:
                    print('invalid input, quit',lst)
                    quit()
                new_rhs = float(rhs)*uni_pert
                line=line.replace(rhs,str(new_rhs))
                fout.write(line)
            elif 'QUADOBJ' == mode:
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                rhs = lst[2]
                if len(lst)>3:
                    print('invalid input, quit',lst)
                    quit()
                new_rhs = float(rhs)*uni_pert*uni_pert
                line=line.replace(rhs,str(new_rhs))
                fout.write(line)
    ff.close()
    fout.close()


def gen_8938(pert_range=0.1, ori_ins = '../qplib_qps/QPLIB_8938.mps', indx = 0):
    uni_pert = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
    print(uni_pert)

    ff = open(ori_ins,'r')
    mode = ''
    tar_ins = ori_ins.split('/')[-1].replace('.mps',f'_{indx}.mps')
    if not os.path.exists(f'../gen_train_8938'):
        os.mkdir(f'../gen_train_8938')
    out_file = f'../gen_train_8938/{tar_ins}'
    fout = open(out_file,'w')
    for line in ff:
        if ' '!=line[0]:
            fout.write(line)
            if 'ROWS' in line:
                mode = 'ROWS'
            elif 'COLUMNS' in line:
                mode = 'COLUMNS'
            elif 'RHS' in line:
                mode = 'RHS'
            elif 'RANGES' in line:
                mode = 'RANGES'
            elif 'BOUNDS' in line:
                mode = 'BOUNDS'
            elif 'QUADOBJ' in line:
                mode = 'QUADOBJ'
            else:
                continue
        else:
            if 'ROWS' == mode:
                fout.write(line)
            elif 'COLUMNS' == mode:
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                coeff1 = lst[2]
                coeff2 = None
                if len(lst)>3:
                    coeff2 = lst[4]
                new_line = f' {lst[0]}'
                if 'OBJ' in lst[1]:
                    pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                    new_coeff1 = float(coeff1)*pert1
                    new_line = new_line + f' {lst[1]} {new_coeff1}'
                else: 
                    new_line = new_line + f' {lst[1]} {lst[2]}'

                if coeff2 is not None:
                    if 'OBJ' in lst[3]:
                        pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                        new_coeff2 = float(coeff2)*uni_pert
                        new_line = new_line + f' {lst[3]} {new_coeff2}'
                    else:
                        new_line = new_line + f' {lst[3]} {lst[4]}'
                new_line = new_line+'\n'
                fout.write(new_line)
            elif 'RHS' == mode:
                # lst = line.replace('\n','').split(' ')
                # lst = [x for x in lst if x!='']
                # rhs = lst[2]
                # if len(lst)>3:
                #     print('invalid input, quit',lst)
                #     quit()
                # new_rhs = float(rhs)*uni_pert
                # line=line.replace(rhs,str(new_rhs))
                fout.write(line)
            elif 'RANGES' == mode:
                print(line)
                fout.write(line)
            elif 'BOUNDS' == mode:
                # lst = line.replace('\n','').split(' ')
                # lst = [x for x in lst if x!='']
                # rhs = lst[3]
                # if len(lst)>4:
                #     print('invalid input, quit',lst)
                #     quit()
                # new_rhs = float(rhs)*uni_pert
                # line=line.replace(rhs,str(new_rhs))
                fout.write(line)
                
            # Q
            elif 'QUADOBJ' == mode:
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                rhs = lst[2]
                if len(lst)>3:
                    print('invalid input, quit',lst)
                    quit()
                new_rhs = float(rhs)*uni_pert
                line=line.replace(rhs,str(new_rhs))
                fout.write(line)
    ff.close()
    fout.close()

def gen_8906(pert_range=0.1, ori_ins = '../qplib_qps/QPLIB_8906.mps', indx = 0):
    uni_pert = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
    print(uni_pert)

    ff = open(ori_ins,'r')
    mode = ''
    tar_ins = ori_ins.split('/')[-1].replace('.mps',f'_{indx}.mps')
    if not os.path.exists(f'../gen_train_8906'):
        os.mkdir(f'../gen_train_8906')
    out_file = f'../gen_train_8906/{tar_ins}'
    fout = open(out_file,'w')
    for line in ff:
        if ' '!=line[0]:
            fout.write(line)
            if 'ROWS' in line:
                mode = 'ROWS'
            elif 'COLUMNS' in line:
                mode = 'COLUMNS'
            elif 'RHS' in line:
                mode = 'RHS'
            elif 'RANGES' in line:
                mode = 'RANGES'
            elif 'BOUNDS' in line:
                mode = 'BOUNDS'
            elif 'QUADOBJ' in line:
                mode = 'QUADOBJ'
            else:
                continue
        else:
            if 'ROWS' == mode:
                fout.write(line)
            elif 'COLUMNS' == mode:
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                coeff1 = lst[2]
                coeff2 = None
                if len(lst)>3:
                    coeff2 = lst[4]
                new_line = f' {lst[0]}'
                if 'OBJ' in lst[1] :
                    pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                    new_coeff1 = float(coeff1)*pert1
                    new_line = new_line + f' {lst[1]} {new_coeff1}'
                else: 
                    new_line = new_line + f' {lst[1]} {lst[2]}'

                if coeff2 is not None:
                    if 'OBJ' in lst[3]:
                        pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                        new_coeff2 = float(coeff2)*pert1
                        new_line = new_line + f' {lst[3]} {new_coeff2}'
                    else:
                        new_line = new_line + f' {lst[3]} {lst[4]}'
                new_line = new_line+'\n'
                fout.write(new_line)
            elif 'RHS' == mode:
                # lst = line.replace('\n','').split(' ')
                # lst = [x for x in lst if x!='']
                # rhs = lst[2]
                # if len(lst)>3:
                #     print('invalid input, quit',lst)
                #     quit()
                # new_rhs = float(rhs)*uni_pert
                # line=line.replace(rhs,str(new_rhs))
                fout.write(line)
            elif 'RANGES' == mode:
                print(line)
                fout.write(line)
            elif 'BOUNDS' == mode:
                # lst = line.replace('\n','').split(' ')
                # lst = [x for x in lst if x!='']
                # rhs = lst[3]
                # if len(lst)>4:
                #     print('invalid input, quit',lst)
                #     quit()
                # new_rhs = float(rhs)*uni_pert
                # line=line.replace(rhs,str(new_rhs))
                fout.write(line)
                
            # Q
            elif 'QUADOBJ' == mode:
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                rhs = lst[2]
                if len(lst)>3:
                    print('invalid input, quit',lst)
                    quit()
                pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                new_rhs = float(rhs)*pert1
                # new_rhs = float(rhs)*uni_pert
                line=line.replace(rhs,str(new_rhs))
                fout.write(line)
    ff.close()
    fout.close()


def gen_8785(pert_range=0.1, ori_ins = '../qplib_qps/QPLIB_8785.mps', indx = 0):
    uni_pert = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
    print(uni_pert)

    ff = open(ori_ins,'r')
    mode = ''
    tar_ins = ori_ins.split('/')[-1].replace('.mps',f'_{indx}.mps')
    if not os.path.exists(f'../gen_train_8785'):
        os.mkdir(f'../gen_train_8785')
    out_file = f'../gen_train_8785/{tar_ins}'
    fout = open(out_file,'w')
    for line in ff:
        if ' '!=line[0]:
            fout.write(line)
            if 'ROWS' in line:
                mode = 'ROWS'
            elif 'COLUMNS' in line:
                mode = 'COLUMNS'
            elif 'RHS' in line:
                mode = 'RHS'
            elif 'RANGES' in line:
                mode = 'RANGES'
            elif 'BOUNDS' in line:
                mode = 'BOUNDS'
            elif 'QUADOBJ' in line:
                mode = 'QUADOBJ'
            else:
                continue
        else:
            if 'ROWS' == mode:
                fout.write(line)
            elif 'COLUMNS' == mode:
                # lst = line.replace('\n','').split(' ')
                # lst = [x for x in lst if x!='']
                # coeff1 = lst[2]
                # coeff2 = None
                # if len(lst)>3:
                #     coeff2 = lst[4]
                # new_line = f' {lst[0]}'
                # if 'OBJ' in lst[1]:
                #     pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                #     new_coeff1 = float(coeff1)*pert1
                #     new_line = new_line + f' {lst[1]} {new_coeff1}'
                # else: 
                #     new_line = new_line + f' {lst[1]} {lst[2]}'

                # if coeff2 is not None:
                #     if 'OBJ' in lst[3]:
                #         pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                #         new_coeff2 = float(coeff2)*pert1
                #         new_line = new_line + f' {lst[3]} {new_coeff2}'
                #     else:
                #         new_line = new_line + f' {lst[3]} {lst[4]}'
                # new_line = new_line+'\n'
                fout.write(line)
            elif 'RHS' == mode:
                # lst = line.replace('\n','').split(' ')
                # lst = [x for x in lst if x!='']
                # rhs = lst[2]
                # if len(lst)>3:
                #     print('invalid input, quit',lst)
                #     quit()
                # new_rhs = float(rhs)*uni_pert
                # line=line.replace(rhs,str(new_rhs))
                fout.write(line)
            elif 'RANGES' == mode:
                fout.write(line)
            elif 'BOUNDS' == mode:
                pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                rhs = lst[3]
                if len(lst)>4:
                    print('invalid input, quit',lst)
                    quit()
                new_rhs = float(rhs)*pert1
                lst = line.split(' ')
                lst = [x for x in lst if x!='']
                lst[-1] = str(int(round(new_rhs)))
                new_line = ' ' + ' '.join(lst) + '\n'
                # print(lst)
                # print(line)
                # print(new_line)
                # quit()
                # line=line.replace(rhs,str(new_rhs))
                # print(line)
                fout.write(new_line)
                
            # Q
            elif 'QUADOBJ' == mode:
                pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                rhs = lst[2]
                if len(lst)>3:
                    print('invalid input, quit',lst)
                    quit()
                new_rhs = float(rhs)*pert1
                lst = line.split(' ')
                lst = [x for x in lst if x!='']
                lst[-1] = str(new_rhs)
                new_line = ' ' + ' '.join(lst)+ '\n'
                fout.write(new_line)
    ff.close()
    fout.close()

def gen_8602(pert_range=0.1, ori_ins = '../qplib_qps/QPLIB_8602.mps', indx = 0):
    uni_pert = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
    print(uni_pert)

    ff = open(ori_ins,'r')
    mode = ''
    tar_ins = ori_ins.split('/')[-1].replace('.mps',f'_{indx}.mps')
    if not os.path.exists(f'../gen_train_8602'):
        os.mkdir(f'../gen_train_8602')
    out_file = f'../gen_train_8602/{tar_ins}'
    fout = open(out_file,'w')
    for line in ff:
        if ' '!=line[0]:
            fout.write(line)
            if 'ROWS' in line:
                mode = 'ROWS'
            elif 'COLUMNS' in line:
                mode = 'COLUMNS'
            elif 'RHS' in line:
                mode = 'RHS'
            elif 'RANGES' in line:
                mode = 'RANGES'
            elif 'BOUNDS' in line:
                mode = 'BOUNDS'
            elif 'QUADOBJ' in line:
                mode = 'QUADOBJ'
            else:
                continue
        else:
            if 'ROWS' == mode:
                fout.write(line)
            elif 'COLUMNS' == mode:
                # lst = line.replace('\n','').split(' ')
                # lst = [x for x in lst if x!='']
                # coeff1 = lst[2]
                # coeff2 = None
                # if len(lst)>3:
                #     coeff2 = lst[4]
                # new_line = f' {lst[0]}'
                # if 'OBJ' in lst[1]:
                #     pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                #     new_coeff1 = float(coeff1)*pert1
                #     new_line = new_line + f' {lst[1]} {new_coeff1}'
                # else: 
                #     new_line = new_line + f' {lst[1]} {lst[2]}'

                # if coeff2 is not None:
                #     if 'OBJ' in lst[3]:
                #         pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                #         new_coeff2 = float(coeff2)*pert1
                #         new_line = new_line + f' {lst[3]} {new_coeff2}'
                #     else:
                #         new_line = new_line + f' {lst[3]} {lst[4]}'
                # new_line = new_line+'\n'
                fout.write(line)
            elif 'RHS' == mode:
                # lst = line.replace('\n','').split(' ')
                # lst = [x for x in lst if x!='']
                # rhs = lst[2]
                # if len(lst)>3:
                #     print('invalid input, quit',lst)
                #     quit()
                # new_rhs = float(rhs)*uni_pert
                # line=line.replace(rhs,str(new_rhs))
                fout.write(line)
            elif 'RANGES' == mode:
                fout.write(line)
            elif 'BOUNDS' == mode:
                pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                rhs = lst[3]
                if len(lst)>4:
                    print('invalid input, quit',lst)
                    quit()
                new_rhs = float(rhs)*pert1
                lst = line.split(' ')
                lst = [x for x in lst if x!='']
                lst[-1] = str(int(round(new_rhs)))
                new_line = ' ' + ' '.join(lst) + '\n'
                # print(lst)
                # print(line)
                # print(new_line)
                # quit()
                # line=line.replace(rhs,str(new_rhs))
                # print(line)
                fout.write(new_line)
                
            # Q
            elif 'QUADOBJ' == mode:
                pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                rhs = lst[2]
                if len(lst)>3:
                    print('invalid input, quit',lst)
                    quit()
                new_rhs = float(rhs)*pert1
                lst = line.split(' ')
                lst = [x for x in lst if x!='']
                lst[-1] = str(new_rhs)
                new_line = ' ' + ' '.join(lst)+ '\n'
                fout.write(new_line)
    ff.close()
    fout.close()


def gen_8906(pert_range=0.1, ori_ins = '../qplib_qps/QPLIB_8906.mps', indx = 0):
    uni_pert = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
    print(uni_pert)

    ff = open(ori_ins,'r')
    mode = ''
    tar_ins = ori_ins.split('/')[-1].replace('.mps',f'_{indx}.mps')
    if not os.path.exists(f'../gen_train_8906'):
        os.mkdir(f'../gen_train_8906')
    out_file = f'../gen_train_8906/{tar_ins}'
    fout = open(out_file,'w')
    for line in ff:
        if ' '!=line[0]:
            fout.write(line)
            if 'ROWS' in line:
                mode = 'ROWS'
            elif 'COLUMNS' in line:
                mode = 'COLUMNS'
            elif 'RHS' in line:
                mode = 'RHS'
            elif 'RANGES' in line:
                mode = 'RANGES'
            elif 'BOUNDS' in line:
                mode = 'BOUNDS'
            elif 'QUADOBJ' in line:
                mode = 'QUADOBJ'
            else:
                continue
        else:
            if 'ROWS' == mode:
                fout.write(line)
            elif 'COLUMNS' == mode:
                pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                rhs = lst[2]
                new_rhs = float(rhs)*pert1
                lst[2] = str(int(round(new_rhs)))
                if len(lst)>4:
                    rhs = lst[4]
                    pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                    new_rhs = float(rhs)*pert1
                    lst[4] = str(int(round(new_rhs)))
                new_line = ' ' + ' '.join(lst) + '\n'
                fout.write(new_line)
                fout.write(line)
            elif 'RHS' == mode:
                pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                rhs = lst[2]
                new_rhs = float(rhs)*pert1
                lst[2] = str(int(round(new_rhs)))
                new_line = ' ' + ' '.join(lst) + '\n'

                fout.write(line)
            elif 'RANGES' == mode:
                fout.write(line)
            elif 'BOUNDS' == mode:
                # pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                # lst = line.replace('\n','').split(' ')
                # lst = [x for x in lst if x!='']
                # rhs = lst[3]
                # if len(lst)>4:
                #     print('invalid input, quit',lst)
                #     quit()
                # new_rhs = float(rhs)*pert1
                # lst = line.split(' ')
                # lst = [x for x in lst if x!='']
                # lst[-1] = str(int(round(new_rhs)))
                # new_line = ' ' + ' '.join(lst) + '\n'
                # fout.write(new_line)
                fout.write(line)
                
            # Q
            elif 'QUADOBJ' == mode:
                pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                lst = line.replace('\n','').split(' ')
                lst = [x for x in lst if x!='']
                rhs = lst[2]
                if len(lst)>3:
                    print('invalid input, quit',lst)
                    quit()
                new_rhs = float(rhs)*pert1
                lst = line.split(' ')
                lst = [x for x in lst if x!='']
                lst[-1] = str(new_rhs)
                new_line = ' ' + ' '.join(lst)+ '\n'
                fout.write(new_line)
    ff.close()
    fout.close()


def pert_ins(pert_range=0.1, ori_ins = '../qplib_qps/QPLIB_8906.mps', indx = 0, 
                identifier ='8906',
                per_Q = True,
                per_c = True,
                per_A = False,
                per_b = False,
            ):
    uni_pert = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
    print(uni_pert)

    ff = open(ori_ins,'r')
    mode = ''
    tar_ins = ori_ins.split('/')[-1].replace('.mps',f'_{indx}.mps')
    if not os.path.exists(f'../gen_train_{identifier}'):
        os.mkdir(f'../gen_train_{identifier}')
    out_file = f'../gen_train_{identifier}/{tar_ins}'
    fout = open(out_file,'w')
    for line in ff:
        if ' '!=line[0]:
            fout.write(line)
            if 'ROWS' in line:
                mode = 'ROWS'
            elif 'COLUMNS' in line:
                mode = 'COLUMNS'
            elif 'RHS' in line:
                mode = 'RHS'
            elif 'RANGES' in line:
                mode = 'RANGES'
            elif 'BOUNDS' in line:
                mode = 'BOUNDS'
            elif 'QUADOBJ' in line:
                mode = 'QUADOBJ'
            else:
                continue
        else:
            if 'ROWS' == mode:
                fout.write(line)
            elif 'COLUMNS' == mode:
                if per_A and not per_c:
                    pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                    lst = line.replace('\n','').split(' ')
                    lst = [x for x in lst if x!='']
                    row_name = lst[1]
                    if 'obj' not in row_name.lower():
                        rhs = lst[2]
                        new_rhs = float(rhs)*pert1
                        lst[2] = str(new_rhs)
                    if len(lst)>4:
                        row_name = lst[3]
                        if 'obj' not in row_name.lower():
                            rhs = lst[4]
                            pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                            new_rhs = float(rhs)*pert1
                            lst[4] = str(new_rhs)
                    new_line = ' ' + ' '.join(lst) + '\n'
                    fout.write(new_line)
                elif per_c and not per_A:
                    pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                    lst = line.replace('\n','').split(' ')
                    lst = [x for x in lst if x!='']
                    row_name = lst[1]
                    if 'obj' in row_name.lower():
                        rhs = lst[2]
                        new_rhs = float(rhs)*pert1
                        lst[2] = str(new_rhs)
                        # print(line)
                        # print(lst[2])
                        # input()
                    if len(lst)>4:
                        row_name = lst[3]
                        if 'obj' in row_name.lower():
                            rhs = lst[4]
                            pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                            new_rhs = float(rhs)*pert1
                            lst[4] = str(new_rhs)
                    new_line = ' ' + ' '.join(lst) + '\n'
                    fout.write(new_line)
                elif per_c and per_A:
                    pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                    lst = line.replace('\n','').split(' ')
                    lst = [x for x in lst if x!='']
                    rhs = lst[2]
                    new_rhs = float(rhs)*pert1
                    lst[2] = str(new_rhs)
                    if len(lst)>4:
                        rhs = lst[4]
                        pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                        new_rhs = float(rhs)*pert1
                        lst[4] = str(new_rhs)
                    new_line = ' ' + ' '.join(lst) + '\n'
                    fout.write(new_line)
                else:
                    fout.write(line)
            elif 'RHS' == mode:
                if per_b:
                    pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                    lst = line.replace('\n','').split(' ')
                    lst = [x for x in lst if x!='']
                    rhs = lst[2]
                    new_rhs = float(rhs)*pert1
                    lst[2] = str(int(round(new_rhs)))
                    new_line = ' ' + ' '.join(lst) + '\n'
                    fout.write(line)
                else:
                    fout.write(line)
            elif 'RANGES' == mode:
                fout.write(line)
            elif 'BOUNDS' == mode:
                fout.write(line)
                
            # Q
            elif 'QUADOBJ' == mode:
                if per_Q:
                    pert1 = 1.0+random.random()*pert_range* (1-random.randint(0, 1)*2)
                    lst = line.replace('\n','').split(' ')
                    lst = [x for x in lst if x!='']
                    rhs = lst[2]
                    if len(lst)>3:
                        print('invalid input, quit',lst)
                        quit()
                    new_rhs = float(rhs)*pert1
                    lst = line.split(' ')
                    lst = [x for x in lst if x!='']
                    lst[-1] = str(new_rhs)
                    new_line = ' ' + ' '.join(lst)+ '\n'
                    fout.write(new_line)
                else:
                    fout.write(line)

    ff.close()
    fout.close()


import time



def gen_ins(indx = 0, identifier='syn',n=5000,m=5000,sat=0.9,
            density=0.1,ub=1.0,seed=0,use_q=True
            ):
    np.random.seed(seed)
    random.seed(seed)
    tar_ins = f'{identifier}_{indx}.mps'
    if not os.path.exists(f'../gen_train_{identifier}'):
        os.mkdir(f'../gen_train_{identifier}')
    out_file = f'../gen_train_{identifier}/{tar_ins}'
    # construct model
    model = gp.Model(tar_ins)
    model.Params.OutputFlag=0
    # add vars
    v_time = time.time()
    l_coeff = np.random.normal(size=(n),loc=3,scale=1).tolist()
    if use_q:
        q_coeff = np.random.normal(size=(n),loc=4,scale=2).tolist()
        for i in range(n):
            if q_coeff[i]<0.1:
                q_coeff[i] = 0.1
        
    variables = []
    ubs_map = {}
    for i in range(n):
        # ub_local = ub*random.random()
        ub_local = ub
        ubs_map[i] = ub_local
        tmp = model.addVar(lb=0, ub=ub_local, name=f"x_{i}")
        variables.append(tmp)
    print(f'var added, time: {time.time()-v_time}')
    # construct objective
    v_time = time.time()

    if use_q:
        model.setObjective(gp.quicksum(l_coeff[i]*variables[i]+q_coeff[i]*variables[i]*variables[i] for i in range(n)), gp.GRB.MINIMIZE)
    else:
        model.setObjective(gp.quicksum(l_coeff[i]*variables[i] for i in range(n)), gp.GRB.MINIMIZE)

    print(f'objective added, time: {time.time()-v_time}')
    # construct A
    v_time = time.time()
    
    rhs=[]
    cos=[]
    for i in range(m):
        sums = 0.
        expr = gp.LinExpr()
        idf = np.random.rand(n)
        co = np.random.normal(size=(n),loc=2,scale=1)
        for j in range(n):
            if idf[j]<=density:
                v = variables[j]
                expr += v*co[j]
                sums += ubs_map[ubs_map[j]] * co[j]
        cos.append(expr)
        rhs.append(max(sums*sat,0.0))
    
    model.addConstrs(cos[i]>=rhs[i] for i in range(m))
    print(f'constraint added, time: {time.time()-v_time}')
    
    # model.optimize()
    # if model.Status!=2:
    #     print('Infeasible model')
    # else:
    #     print('OK,save')
    model.write(out_file)


def gen_ins_mat(indx = 0, identifier='syn',n=5000,m=5000,sat=0.9,
            density=0.1,ub=1.0,seed=0,scale_ratio = 0.1,loc=2,scale=1
            ):
    np.random.seed(seed)
    random.seed(seed)
    tar_ins = f'{identifier}_{indx}.mps'
    if not os.path.exists(f'../gen_train_{identifier}'):
        os.mkdir(f'../gen_train_{identifier}')
    out_file = f'../gen_train_{identifier}/{tar_ins}'
    # construct model
    model = gp.Model(tar_ins)
    model.Params.OutputFlag=0
    # add vars
    v_time = time.time()
    l_coeff = np.random.normal(size=(n),loc=3,scale=1).tolist()
    q_coeff = np.random.normal(size=(n),loc=4,scale=2).tolist()
    for i in range(n):
        if q_coeff[i]<0.1:
            q_coeff[i] = 0.1
    variables = []
    ubs_map = {}
    for i in range(n):
        ub_local = ub*random.random()
        ubs_map[i] = ub_local
        tmp = model.addVar(lb=0, ub=ub_local, name=f"x_{i}")
        variables.append(tmp)
    print(f'var added, time: {time.time()-v_time}')
    # construct objective
    v_time = time.time()
    model.setObjective(gp.quicksum(l_coeff[i]*variables[i]+q_coeff[i]*variables[i]*variables[i] for i in range(n)), gp.GRB.MINIMIZE)
    print(f'objective added, time: {time.time()-v_time}')
    # construct A
    rhs=[]
    cos=[]
    v_time = time.time()
    bbr = range(n)
    for i in range(m):
        nnz_row = int(np.random.normal(loc=int(n*density), scale=n*scale_ratio*density))
        nz_coe = np.random.normal(size=(nnz_row),loc=loc,scale=scale)
        c_indx = np.random.choice(bbr, size=(nnz_row),replace=False)
        rhs = gp.quicksum(ubs_map[j]*nz_coe[edx] for edx,j in enumerate(c_indx))
        cos.append(model.addConstr(gp.quicksum(variables[j]*nz_coe[edx] for edx,j in enumerate(c_indx))>=rhs))
        # if i % 1000 == 0:
        #     print(i)


    print(f'constraint added, time: {time.time()-v_time}')
    
    # model.optimize()
    # if model.Status!=2:
    #     print('Infeasible model')
    # else:
    #     print('OK,save')
    model.write(out_file)


def gen_svm(indx = 0, identifier='small',n=1000,m=1000,
            density=0.15, lbd=0.1
            ):
    tar_ins = f'svm{identifier}_{indx}.mps'
    if not os.path.exists(f'../gen_train_svm{identifier}'):
        os.mkdir(f'../gen_train_svm{identifier}')
    out_file = f'../gen_train_svm{identifier}/{tar_ins}'
    # construct model
    model = gp.Model(tar_ins)
    model.Params.OutputFlag=0
    # add vars
    v_time = time.time()
    variables = []
    variables_t = []
    # add x
    for i in range(n):
        tmp = model.addVar(lb=-float('inf'), ub=float('inf'), name=f"x_{i}")
        variables.append(tmp)
    # add t
    for i in range(m):
        tmp = model.addVar(lb=0, ub=float('inf'), name=f"t_{i}")
        variables_t.append(tmp)
    print(f'var added, time: {time.time()-v_time}')
    # construct objective
    v_time = time.time()
    model.setObjective(gp.quicksum(lbd*variables_t[j] for j in range(m))+gp.quicksum(variables[i]*variables[i] for i in range(n)), gp.GRB.MINIMIZE)
    print(f'objective added, time: {time.time()-v_time}')
    # construct A
    v_time = time.time()
    
    rhs=[]
    cos=[]
    for i in range(m):
        expr = gp.LinExpr()
        bi = -1.0
        if i<=m//2:
            bi = 1.0

        idf = np.random.rand(n)
        if i<=m//2:
            co = np.random.normal(size=(n),loc=1./n,scale=1./n)
        else:
            co = np.random.normal(size=(n),loc=-1./n,scale=1./n)
        for j in range(n):
            if idf[j]<=density:
                expr -= variables[j]*co[j]
        expr += variables_t[i]
        cos.append(expr)
        rhs.append(1.0)
    
    model.addConstrs(cos[i]>=rhs[i] for i in range(m))
    print(f'constraint added, time: {time.time()-v_time}')
    
    # model.optimize()
    # if model.Status!=2:
    #     print('Infeasible model')
    # else:
    #     print('OK,save')
    model.write(out_file)


    
    
    
import multiprocessing


# for i in range(1):
    # gen_cont_uniform_pert(indx = i)
    # gen_8938(indx = i)


    # gen_8602(indx = i)
    # gen_8906(indx = i)
    # gen_8785(indx = i)
    # pert_ins(indx=i,ori_ins = '../qplib_qps/QPLIB_8785.mps',identifier ='8785',)
    # pert_ins(pert_range=0,indx=i,ori_ins = '../qplib_qps/twod.mps',identifier ='twod',)
    # pert_ins(indx=i,ori_ins = '../qplib_qps/QPLIB_3913.mps',identifier ='3913',)
    # pert_ins(indx=i,ori_ins = '../qplib_qps/QPLIB_8845.mps',identifier ='8845',)
    # pert_ins(indx=i,ori_ins = '../qplib_qps/rqp1_nqc.mps',identifier ='rqp1',)
    # gen_ins(indx=i)
    
# p = gen_ins_mat(1,'synxlarge',10000,10000,0.96,0.01,1.0,1,)
# quit()

nworker=6
pool = multiprocessing.Pool(nworker)
    
for i in range(200):
    # p = pool.apply_async(gen_ins, (i,'synsmall',1000,1000,0.8,0.3,))  
    # p = pool.apply_async(gen_ins, (i,'synmid',5000,5000,0.8,0.1,))  
    # p = pool.apply_async(gen_ins, (i,'synxlarge',100000,100000,0.96,0.01,1.0,i,))  
    # p = pool.apply_async(gen_ins, (i,'synxlarge',10000,10000,0.96,0.1,1.0,i,))  
    p = pool.apply_async(gen_ins, (i,'lplarge',10000,10000,0.96,0.1,1.0,i,False))  
    # p = pool.apply_async(gen_svm, (i,'small',256,10000,0.15,))  
    # p = pool.apply_async(gen_svm, (i,'mid',512,50000,0.15,))  
    # p = pool.apply_async(gen_svm, (i,'large',1024,100000,0.15,))  
    # p = pool.apply_async(pert_ins, (0.1, '../qplib_qps/QPLIB_8602.mps', i, '8602',True,True,False,False))  
    # p = pool.apply_async(pert_ins, (0.1, '../qplib_qps/QPLIB_5527.mps', i, '5527',True,True,False,False))  
    # p = pool.apply_async(pert_ins, (0.1, '../qplib_qps/QPLIB_8845.mps', i, '8845',True,True,False,False))  
    # p = pool.apply_async(pert_ins, (0.1, '../qplib_qps/QPLIB_8559.mps', i, '8559',True,True,False,False))  
    # p = pool.apply_async(pert_ins, (0.1, '../qplib_qps/QPLIB_5543.mps', i, '5543',True,True,False,False))  
    # p = pool.apply_async(pert_ins, (0.1, '../qplib_qps/QPLIB_5924.mps', i, '5924',True,True,False,False))  
    # p = pool.apply_async(pert_ins, (0.1, '../qplib_qps/QPLIB_3547.mps', i, '3547',True,True,False,False))  
    # p = pool.apply_async(pert_ins, (0.1, '../qplib_qps/QPLIB_3698.mps', i, '3698',True,True,False,False))  
    # p = pool.apply_async(pert_ins, (0.1, '../qplib_qps/QPLIB_3708.mps', i, '3708',True,True,False,False))  
    # p = pool.apply_async(pert_ins, (0.1, '../qplib_qps/QPLIB_3913.mps', i, '3913',True,True,False,False))  
pool.close()
pool.join()