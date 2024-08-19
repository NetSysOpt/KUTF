
# Python program to read
# json file
import os
import json





def read_json(fnm):
    # Opening JSON file
    f = open(fnm)
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    # Closing file
    f.close()
    return data

file_map = {}

ori_files = os.listdir('../qplib_log')
ori_files = [x for x in ori_files if '.json' in x and '.gz' not in x]
for fnm in ori_files:
    if fnm not in file_map:
        file_map[fnm] = [[0,0,0,0,0,0,0],[0,0]]
    logs = read_json(f'../qplib_log/{fnm}')
    itnm = logs['iteration_count']
    tttm = logs['solve_time_sec']
    # primal_objective
    # convergence_information
    # solution_stats
    primal_obj = logs['solution_stats']['convergence_information'][0]['primal_objective']
    dual_obj = logs['solution_stats']['convergence_information'][0]['dual_objective']
    gap = logs['solution_stats']['convergence_information'][0]['relative_optimality_gap']
    pri_res = logs['solution_stats']['convergence_information'][0]['relative_l_inf_primal_residual']
    dua_res = logs['solution_stats']['convergence_information'][0]['relative_l_inf_dual_residual']
    # relative_optimality_gap
    # relative_l_inf_dual_residual
    # relative_l_inf_primal_residual
    # print(logs)
    # print(itnm)
    # print(tttm)
    file_map[fnm][0][0] = itnm
    file_map[fnm][0][1] = tttm
    file_map[fnm][0][2] = primal_obj
    file_map[fnm][0][3] = dual_obj
    file_map[fnm][0][4] = pri_res
    file_map[fnm][0][5] = dua_res
    file_map[fnm][0][6] = gap

    
f = open('qplib_pdqp.csv','w')
st = f'ins time iteration primal_obj dual_obj primal_res dual_res opt_gap\n'
f.write(st)
for fnm in file_map:
    if file_map[fnm][0][1] == -1:
        continue
    if file_map[fnm][0][0] == 0:
        continue
    st = f'{fnm} {file_map[fnm][0][1]} {file_map[fnm][0][0]} {file_map[fnm][0][2]} {file_map[fnm][0][3]} {file_map[fnm][0][4]} {file_map[fnm][0][5]} {file_map[fnm][0][6]} \n'
    f.write(st)
f.write(st)
f.close()