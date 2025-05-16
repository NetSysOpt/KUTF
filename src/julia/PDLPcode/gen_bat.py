# julia --project scripts/solve.jl --instance_path=/home/lxyang/git/pdqpnet/ins/train/AUG2D.QPS --output_directory=/home/lxyang/git/pdqpnet/logs

# julia scripts/solve_warmstart.jl --instance_path=../../../ins/train/YAO.QPS --output_directory=../../logs

import os
import torch
mode = 'ori'
mode = 'qplib'
mode = 'cont201'
# mode = 'cont201_test'
mode = 'qplib8938'
mode = 'qplib8938_test'
mode = 'mm_test'
mode = '8785_test'
# mode = 'qplib8602'
# mode = 'qplib8785'
# mode = 'qplib8906'
# mode = 'qplib8845'
mode = '8906_test'
# mode = '8602_test'
mode = 'synlarge'

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--type','-t', type=str, default='8785_test')
parser.add_argument('--timelim','-m', type=int, default=500)
parser.add_argument('--ori','-o', type=int, default=0)
parser.add_argument('--choose','-c', type=int, default=0)
parser.add_argument('--ex','-e', type=str, default='')

args = parser.parse_args()


def get_pw(fnm,choose=0):
    fnm = fnm.replace('.mps','.mps.pkl').replace('.QPS','.QPS.pkl')
    f=open(f'../../../predictions/primalweight_{fnm}.sol','r')
    for line in f:
        res = line.replace('\n','').split(' ')
        res=[float(x) for x in res if x!=''][choose]
        res = float(res)
        print(res)
        if res!=res:
            res = 10.0
            print(fnm)
    f.close()
    return res

mode = args.type

if mode == 'ori':
    f=open('run_test.bat','w')
    flist = os.listdir('../../../ins/train/')
    for fnm in flist:
        fdir = f'../../../ins/train/{fnm}'
        st = 'julia scripts/solve_warmstart.jl'
        st = f'{st} --instance_path={fdir} --output_directory=../../logs/warmstart'
        print(st)
        f.write(st+'\n')



    flist = os.listdir('../../../ins/valid/')
    for fnm in flist:
        fdir = f'../../../ins/valid/{fnm}'
        st = 'julia scripts/solve_warmstart.jl'
        st = f'{st} --instance_path={fdir} --output_directory=../../logs/warmstart'
        print(st)
        f.write(st+'\n')


    flist = os.listdir('../../../ins/train/')
    for fnm in flist:
        fdir = f'../../../ins/train/{fnm}'
        st = 'julia scripts/solve.jl'
        st = f'{st} --instance_path={fdir} --output_directory=../../logs/ori'
        print(st)
        f.write(st+'\n')
        
        
    flist = os.listdir('../../../ins/valid/')
    for fnm in flist:
        fdir = f'../../../ins/valid/{fnm}'
        st = 'julia scripts/solve.jl'
        st = f'{st} --instance_path={fdir} --output_directory=../../logs/ori'
        print(st)
        f.write(st+'\n')
    f.close()
        
elif mode == 'qplib':
    f=open('run_test_qplib.bat','w')
    flist = os.listdir('../../../ins/qplib_qps/')
    for fnm in flist:
        fdir = f'../../../ins/qplib_qps/{fnm}'
        st = 'julia scripts/solve.jl'
        st = f'{st} --instance_path={fdir} --output_directory=../../qplib_log'
        print(st)
        f.write(st+'\n')
    
    
    
    f.close()

else:
    # julia ../pdlp/FirstOrderLp.jl/scripts/solve_qp.jl --instance_path ./mps/prob_0.mps --time_sec_limit 15000 --method pdhg --output_dir ./res/0
    if 'test' not in mode:
        modifier = '--termination_evaluation_frequency 10  --relative_optimality_tol 1e-5'
        print('!!!!!! free mode for data')
        f=open(f'run_data.bat','w')
        flist = os.listdir(f'../../../ins/gen_train_{mode}/')
        # flist = os.listdir('../../../pkl/8906_valid/')
        # flist = [x.replace('.pkl','') for x in flist]
        for fnm in flist:
            fdir = f'../../../ins/gen_train_{mode}/{fnm}'
            st = 'julia ./scripts/solve_qp.jl'
            # st = f'{st} --instance_path={fdir} --output_directory=../../../logs --time_sec_limit=3600 --step_size=0.01'
            st = f'{st} --instance_path {fdir} --output_dir ../../../logs_lp --time_sec_limit=3600 {modifier} --method pdhg --writemodel true'
            # st = f'{st} --instance_path={fdir} --output_directory=../../../logs --time_sec_limit=3600 --primal_weight=220.0'
            print(st)
            f.write(st+'\n')
        f.close()

    else:
        # modifier = ''

        # modifier2 = ''
        tl = args.timelim
        choose=args.choose
        # if 'netlib' in mode:
        #     choose = 1
        modifier = '--termination_evaluation_frequency 5 --relative_optimality_tol 1e-4 --verbosity 3'
        modifier2 = '--termination_evaluation_frequency 5 --relative_optimality_tol 1e-4 --verbosity 3'
        # julia ../pdlp/FirstOrderLp.jl/scripts/solve_qp.jl --instance_path ./mps/prob_0.mps --time_sec_limit 15000 --method pdhg --output_dir ./res/0
        mode = mode.replace('test_','').replace('_test','')
        f=open(f'run_test.bat','w')
        flist = os.listdir(f'../../../pkl/{mode}_test/')
        flist = [x.replace('.pkl','') for x in flist]
        for fnm in flist:
            if True:
            # if False:
                if choose<0:
                    pw=' --primal_weight 0.1'
                else:
                    pw = get_pw(fnm,choose)
                    pw = min(pw,10.0)
                    pw = max(pw,0.01)
                    # pw = pw*1.5
                    # pw = 1.8
                    pw =f' --primal_weight {pw}'
                # pw =f' --primal_weight=0.4'
                # pw=''
                fdir = f'../../../ins/gen_train_{mode}/{fnm}'
                st = 'julia ./scripts/solve_qp_warmstart.jl'
                st = f'{st} --instance_path {fdir} --output_dir ../../logs/warmstart{args.ex} --time_sec_limit {tl} {modifier}{pw} --method pdhg'
                # st = f'{st} --instance_path={fdir} --output_directory=../../logs/warmstart --time_sec_limit=3600 --checkiter=100{modifier}'
                print(st)
                f.write(st+'\n')
            # if True:
            # if False:
            if args.ori==1:
                fdir = f'../../../ins/gen_train_{mode}/{fnm}'
                st = 'julia scripts/solve_qp.jl'
                st = f'{st} --instance_path {fdir}  --output_dir ../../../logs --time_sec_limit={tl} {modifier2} --method pdhg'
                print(st)
                f.write(st+'\n')
        f.close()
        

