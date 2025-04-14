import subprocess
import os
import tempfile
import json
import argparse
import time
from tabulate import tabulate
import numpy as np



if __name__ == "__main__":
    start_time=time.time()
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)') 
    parser.add_argument('--ds', type=str, default="api_graph", metavar='S',help='dataset name')
    parser.add_argument('--b_m', type=float, default=0.3, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='S',help='learning rate(default: 0.001)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='S',help='weight decay(default: 0.01)')
    parser.add_argument('--label_ratio', type=float, default=0.2, metavar='S',help='labeled ratio (default: 0.1)')
    parser.add_argument('--nps', type=int, metavar='S',default=10000,help='number of projection samples(default: 100)')
    parser.add_argument('--bma', type=float, metavar='S',default=0.9,help='batch minority allocation(default: 0)')
    parser.add_argument('--alpha', type=float, metavar='S',default=1,help='distill loss multiplier(default: 0)')
    parser.add_argument('--lab_samp_in_mem_ratio', type=float, metavar='S',default=1.0,help='Percentage of labeled samples to store in memory(default: 1.0)')
    parser.add_argument('--bool_gpm', type=str, metavar='S',default="True",help='Enables gradient projections(default: True)')
    parser.add_argument('--mem_strat', type=str, metavar='S',default="equal",help='Buffer memory strategy(default: full initialization)')
    parser.add_argument('--training_cutoff', type=int, default=36, metavar='S',help='train the model for first n tasks and test for time decay on the rest')
    parser.add_argument('--bool_closs', type=str, metavar='S',default="False",help='Enables using contrastive loss(default: False)')
    parser.add_argument('--mlps', type=int, metavar='S',default=1,help='Number of learners (MLPs)default: 1)')
    parser.add_argument('--cos_dist', type=float, metavar='S',default=0.13,help='cosine distance for OWL(default: 0.3)')
    parser.add_argument('--mode_val', type=int, metavar='S',default=90,help='Mode value for OWL (default: 99)')
    parser.add_argument('--analyst_labels', type=int, metavar='S',default=50,help='No of labels from analysts (default:50)')
    parser.add_argument('--uncertainity', type=str, metavar='S',default="psuedo-loss",help='Sample selector')
    parser.add_argument('--family_info', type=bool, metavar='S',default=True,help='Include Family Info in HCL')
    parser.add_argument('--cl_method',type=str,metavar='S',default="MIR",help='Method of CL')    
    args = parser.parse_args()
    auc_results = {}
    seed_list = [1,2,3]
    label_results ={}
    curr_dir = os.getcwd()
    for seed_value in seed_list:
        print("seed is",seed_value)
        fd, temp_file_name = tempfile.mkstemp() # create temporary file
        os.close(fd) # close the file
        proc = subprocess.Popen([
                                "python3 "+
                                f"{args.cl_method}.py"+
                                f" --seed={seed_value}"+
                                f" --ds={args.ds}"+
                                f" --lr={args.lr}"+
                                f" --wd={args.wd}"+
                                f" --gpu={args.gpu}"+
                                f" --filename={temp_file_name}"+
                                f" --training_cutoff={args.training_cutoff}"
                            ], shell=True, cwd=curr_dir)
        proc.communicate()
        with open(temp_file_name) as fp:
            result = json.load(fp)
            # auc_results[str(seed_value)] = result#result[str(seed_value)]
            auc_results[str(seed_value)] = result[str(seed_value)]
        os.unlink(temp_file_name)    
    print(label_results)
    print("{:<20}  {:<20}".format('Argument','Value'))
    print("*"*80)
    for arg in vars(args):
        print("{:<20}  {:<20}".format(arg, getattr(args, arg)))
    print("*"*80)    
    # print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20} {:<10}".format('seed','PR-AUC(O)', 'PR-AUC(I)', 'ROC-AUC','grad_norm_mean','grad_norm_variance'))
    print("*"*80)
    aut_results = {}
   
    for key, value in auc_results.items():
        # print("training results for seed value",key)
        prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N = value[0][0],value[0][1],value[0][2],value[0][3],value[0][4],value[0][5],value[0][6]
        aut_results[key] = [prauc_in_aut,prauc_out_aut]
    #     pnt_table = [
    #     # ['task_CI']+ task_CI_pnt, 
    #     # ['test_CI'] + test_CI_pnt,
    #     ['prauc Benign traffic'] + prauc_in_pnt, 
    #     ['prauc Attack traffic'] + prauc_out_pnt
    # ]
    #     print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
    #     print(f'AUT(prauc inliers,{N}) := {prauc_in_aut}')
    #     print(f'AUT(prauc outliers,{N}) := {prauc_out_aut}')
    #     print("testing results for seed value",key)
        prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N = value[1][0],value[1][1],value[1][2],value[1][3],value[1][4],value[1][5],value[1][6]
        # pnt_table = [ # ['task_CI']+ task_CI_pnt, 
        #         # ['test_CI'] + test_CI_pnt,
        #         ['prauc Benign traffic'] + prauc_in_pnt, 
        #         ['prauc Attack traffic'] + prauc_out_pnt
        #         ]
        # print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
        # print(f'AUT(prauc inliers,{N}) := {prauc_in_aut}')
        # print(f'AUT(prauc outliers,{N}) := {prauc_out_aut}')
        aut_results[key].extend([prauc_in_aut,prauc_out_aut])
        prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N = value[2][0],value[2][1],value[2][2],value[2][3],value[2][4],value[2][5],value[2][6]
        aut_results[key].extend([prauc_in_aut,prauc_out_aut])
    print("-"*80)
    aut_results_values = list(aut_results.values())
    # aut_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*aut_results_values)]
    aut_average = (np.mean(np.array(list(aut_results.values())),axis=0)).tolist()
    auc_std = (np.std(np.array(list(aut_results.values())),axis=0)).tolist()

    aut_average = (np.mean(np.array(list(aut_results.values())),axis=0)).tolist()
    aut_std = (np.std(np.array(list(aut_results.values())),axis=0)).tolist()
    print("{:<20}  {:<20}  {:<20}  ".format('Cols','AUT(Benign)-seen','AUT(Attack)-seen'))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20} ".format('Mean',float(str(aut_average[0])[:5]), float(str(aut_average[1]))))    
    print("{:<20}  {:<20}  {:<20} ".format('std',float(str(aut_std[0])[:5]), float(str(aut_std[1]))))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}  ".format('Cols','AUT(Benign)-unseen','AUT(Attack)-unseen'))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20} ".format('Mean',float(str(aut_average[2])[:5]), float(str(aut_average[3]))))    
    print("{:<20}  {:<20}  {:<20} ".format('std',float(str(aut_std[2])[:5]), float(str(aut_std[3]))))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}  ".format('Cols','AUT(Benign)-all','AUT(Attack)-all'))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20} ".format('Mean',float(str(aut_average[4])[:5]), float(str(aut_average[5]))))    
    print("{:<20}  {:<20}  {:<20} ".format('std',float(str(aut_std[4])[:5]), float(str(aut_std[5]))))
    total_time = time.time()-start_time
    print("total execution time is %.3f seconds" % (total_time))
    print("avg execution time %.3f seconds"%(total_time/len(seed_list)))
