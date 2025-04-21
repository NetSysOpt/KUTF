
# Python program to read
# json file
import os
import json
import gzip
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--type','-t', type=str, default='')
args = parser.parse_args()


# filters='8938'
filters='mm'
filters='8785'
filters='8906'
filters=args.type.split(',')
filters = [x for x in filters if x!='']


def read_json(fnm):
    try:
        # Opening JSON file
        if '.gz' in fnm:
            f = gzip.open(fnm)
        else:
            f = open(fnm)
            
        # returns JSON object as 
        # a dictionary
        data = json.load(f)
        # Closing file
        f.close()
        return data
    except:
        return None

file_map = {}

ori_log = './ori'
ori_log = '../../logs'

ori_files = os.listdir(ori_log)
ori_files = [x for x in ori_files if '.json' in x and '.gz' not in x]
for fnm in ori_files:
    if fnm not in file_map:
        file_map[fnm] = []
    file_map[fnm].append([-1,-1,-1,-1,-1,-1])
    print(f'{ori_log}/{fnm}')
    logs = read_json(f'{ori_log}/{fnm}')

    if logs is None:
        continue
    itnm = logs['iteration_count']
    tttm = logs['solve_time_sec']
    tprimal = logs['solution_stats']['convergence_information'][0]['primal_objective']
    # print(logs)
    # print(itnm)
    # print(tttm)
    file_map[fnm][0][0] = itnm
    file_map[fnm][0][1] = tttm
    file_map[fnm][0][2] = tprimal

wms_files1 = os.listdir('./warmstart')

to_process = [x for x in wms_files1 if os.path.isdir(f'./warmstart/{x}')]

folder_names = ['warmstart']

ntypes = 2

wms_files = [x for x in wms_files1 if '.gz'  in x]
for fnm in wms_files:
    print(fnm)
    fnm = fnm.replace('.gz','')
    fnm2 = fnm+'.gz'
    fnm = fnm.replace('_full_log','_summary')
    if fnm not in file_map:
        file_map[fnm] = []
    file_map[fnm].append([-1,-1,-1,-1,-1,-1])
    logs = read_json(f'./warmstart/{fnm2}')
    if logs is None:
        continue
    # print(logs.keys())
    # for ll in logs['iteration_stats']:
    #     print(ll)
    #     input()
    itnm = logs['iteration_count']
    tttm = logs['solve_time_sec']
    tprimal = logs['solution_stats']['convergence_information'][0]['primal_objective']
    # print(logs)
    # print(itnm)
    # print(tttm)
    ress = logs['iteration_stats'][0]['convergence_information'][0]
    pres = ress['relative_l_inf_primal_residual']
    dres = ress['relative_l_inf_dual_residual']
    gap = ress['relative_optimality_gap']
    file_map[fnm][-1][0] = itnm
    file_map[fnm][-1][1] = tttm
    file_map[fnm][-1][2] = tprimal
    file_map[fnm][-1][3] = pres
    file_map[fnm][-1][4] = dres
    file_map[fnm][-1][5] = gap
    

# for pro_folder in to_process:
#     wms_files1 = os.listdir(f'./warmstart/{pro_folder}')
#     folder_names.append(pro_folder)
#     wms_files = [x for x in wms_files1 if '.json' in x and '.gz' not in x]

#     for fnm in wms_files:
#         if fnm not in file_map:
#             file_map[fnm] = []
#         file_map[fnm].append([-1,-1])
#         logs = read_json(f'./warmstart/{pro_folder}/{fnm}')
#         itnm = logs['iteration_count']
#         tttm = logs['solve_time_sec']
#         # print(logs)
#         # print(itnm)
#         # print(tttm)
#         file_map[fnm][-1][0] = itnm
#         file_map[fnm][-1][1] = tttm

ntypes = 2


f = open('res.csv','w')
st = f'ins ori_time '
# st = f'ins ori_time ori_iter '
for ff in folder_names:
    # st+=f'{ff}_time ratio {ff}_iter ratio PDQP_primal ws_primal pres dres rgap'
    st+=f'{ff}_time T_ratio '
    st+=f'ori_iter {ff}_iter iter_ratio '
    st+=f'PDQP_primal ws_primal pres dres rgap'
st+='\n'
print(st)
f.write(st)
# all_rat1 = 0.0
# all_rat2 = 0.0
all_rats1=[0]*ntypes
all_rats2=[0]*ntypes



keys = []
for fnm in file_map:
    keys.append(fnm)
keys.sort()

if filters=='mm':
    tars_sums = os.listdir('/home/lxyang/git/pdqpnet/pkl/mm_test')
    keys = [x.replace('.QPS.pkl','_summary.json').replace('.mps.pkl','_summary.json') for x in tars_sums]
elif len(filters)!=0:
    keys_new = []
    for ft in filters:
        keys_new += [x for x in keys if ft in x]
    keys = keys_new


ops=[0,0,0]
n_ent = -1
for fnm in keys:
    n_ent = max(len(file_map[fnm]), n_ent)
avg_time = [0]*n_ent


processed = 0
for fnm in keys:

    print(fnm,file_map[fnm])
    if len(file_map[fnm])<2:
        continue
    st = f'{fnm} {file_map[fnm][0][1]} '
    # st += f'{file_map[fnm][0][0]} '
    flag = False
    for idx in range(1,len(file_map[fnm])):
        ent = file_map[fnm][idx]
        # time
        if file_map[fnm][idx][1] < 0:
            flag = True
            break
        ratio1 = (file_map[fnm][0][1] - file_map[fnm][idx][1])/(file_map[fnm][0][1]+1e-8)
        all_rats1[idx] +=ratio1
        st += f'{file_map[fnm][idx][1]} {round(ratio1*100,2)}% '

        # iteration
        ratio2 = (file_map[fnm][0][0] - file_map[fnm][idx][0])/(file_map[fnm][0][0]+1e-8)
        st += f'{file_map[fnm][0][0]} {file_map[fnm][idx][0]} {round(ratio2*100,2)}% '

        print(file_map[fnm][idx])
        avg_time[idx] += file_map[fnm][idx][1]

        ratio1 = (file_map[fnm][0][0] - file_map[fnm][idx][0])/(file_map[fnm][0][0]+1e-8)
        all_rats2[idx] +=ratio1
        # st += f'{file_map[fnm][idx][0]} {round(ratio1*100,2)}% '
        st += f'{file_map[fnm][0][2]} {file_map[fnm][idx][2]}'
        processed+=1
        
        st+=f' {file_map[fnm][idx][3]} {file_map[fnm][idx][4]} {file_map[fnm][idx][5]}'
        ops[0]+=file_map[fnm][idx][3]
        ops[1]+=file_map[fnm][idx][4]
        ops[2]+=file_map[fnm][idx][5]


    if not flag:
        avg_time[0] += file_map[fnm][0][1]
        st+='\n'
        f.write(st)

# ops[0]=ops[0]/processed
# ops[1]=ops[1]/processed
# ops[2]=ops[2]/processed

# for idx in range(1,len(all_rats2)):
#     all_rats1[idx] /=processed
#     all_rats2[idx] /=processed
# st = f'avg / / '
# for idx in range(1,len(all_rats2)):
#     st += f'/ {all_rats1[idx]} / {all_rats2[idx]} / / / {ops[0]} {ops[1]} {ops[2]}'
# st+='\n'

st = f'totalTime '
for idx in range(len(all_rats2)):
    st += f'{avg_time[idx]} '
st+='\n'
f.write(st)

st = f'Improv. '
for idx in range(len(all_rats2)):
    val = (avg_time[0] - avg_time[idx])/avg_time[0]
    val = round(val * 100.0, 4)
    st += f'{val}% '
st+='\n'
f.write(st)

f.close()