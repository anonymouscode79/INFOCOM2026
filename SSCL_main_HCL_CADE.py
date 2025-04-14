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
    parser.add_argument('--label_ratio', type=float, default=1, metavar='S',help='labeled ratio (default: 0.1)')
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
    parser.add_argument('--analyst_label', type=int, metavar='S',default=100,help='No of labels from analysts (default:50)')
    parser.add_argument('--uncertainity', type=str, metavar='S',default="pseudo-loss",help='Sample selector')
    parser.add_argument('--family_info', type=str, metavar='S',default=True,help='Include Family Info in HCL')
    args = parser.parse_args()
    auc_results = {}
    seed_list = [1,2,3]
    label_results ={}
    curr_dir = os.getcwd()
    for seed_value in seed_list:
        print("seed is",seed_value)
        fd, temp_file_name = tempfile.mkstemp() # create temporary file
        os.close(fd) # close the file
        print("Family Info Main:",args.family_info)
        proc = subprocess.Popen(["python SSCL_HCL_cade.py --seed="+str(seed_value)+" --ds="+str(args.ds)+ " --analyst_label="+str(args.analyst_label) +" --family_info=" +str(args.family_info) + " --uncertainity="+str(args.uncertainity)+" --lr="+str(args.lr)+" --wd="+str(args.wd)+" --alpha="+str(args.alpha)+" --gpu="+str(args.gpu)+" --filename="+str(temp_file_name)+" --cos_dist="+str(args.cos_dist)+" --mode_val="+str(args.mode_val)+" --bool_gpm="+str(args.bool_gpm)+" --mem_strat="+str(args.mem_strat)+" --b_m="+str(args.b_m)+" --label_ratio="+str(args.label_ratio)+" --lab_samp_in_mem_ratio="+str(args.lab_samp_in_mem_ratio)+" --nps="+str(args.nps)+" --bma="+str(args.bma)+" --training_cutoff="+str(args.training_cutoff)+" --bool_closs="+str(args.bool_closs)+" --mlps="+str(args.mlps)],shell=True,cwd=curr_dir)
        proc.communicate()
        with open(temp_file_name) as fp:
            result = json.load(fp)
            # auc_results[str(seed_value)] = result#result[str(seed_value)]
            auc_results[str(seed_value)] = result[str(seed_value)]
        with open(temp_file_name+"labels") as fp:
            result = json.load(fp)
            # auc_results[str(seed_value)] = result#result[str(seed_value)]
            label_results[str(seed_value)] = result[str(seed_value)]
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
        prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N = value[1][0],value[1][1],value[1][2],value[1][3],value[1][4],value[1][5],value[1][6]
        aut_results[key].extend([prauc_in_aut,prauc_out_aut,float(value[2]),float(value[3]),float(value[4]),float(value[5])])
        prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N = value[6][0],value[6][1],value[6][2],value[6][3],value[6][4],value[6][5],value[6][6]
        aut_results[key].extend([prauc_in_aut,prauc_out_aut])
    printable_label_results ={"attack":[],"benign":[]}
    for key, value in label_results.items():
        attack_analyst, benign_analyst = value[0],value[1]
        printable_label_results["attack"].extend(attack_analyst)
        printable_label_results["benign"].extend(benign_analyst)
    print(label_results)
    print(printable_label_results)
    print("-"*80)
    aut_results_values = list(aut_results.values())
    # aut_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*aut_results_values)]
    aut_average = (np.mean(np.array(list(aut_results.values())),axis=0)).tolist()
    auc_std = (np.std(np.array(list(aut_results.values())),axis=0)).tolist()

    attack_mean = np.mean(np.array(printable_label_results["attack"]))
    benign_mean = np.mean(np.array(printable_label_results["benign"]))
    attack_std = np.std(np.array(printable_label_results["attack"]))
    benign_std = np.std(np.array(printable_label_results["benign"]))
    print("{:<20}  {:<20}  {:<20}".format('Cols','AUT(Benign)-seen tasks','AUT(Attack)-seen tasks'))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}".format('Mean',float(str(aut_average[0])[:5]), float(str(aut_average[1])[:5])))
    print("{:<20}  {:<20}  {:<20}".format('Variance',float(str(auc_std[0])[:5]), float(str(auc_std[1])[:5])))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}".format('Cols','AUT(Benign)-unseen tasks','AUT(Attack)-unseen tasks'))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}".format('Mean',float(str(aut_average[2])[:5]), float(str(aut_average[3])[:5])))
    print("{:<20}  {:<20}  {:<20}".format('Variance',float(str(auc_std[2])[:5]), float(str(auc_std[3])[:5])))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}".format('Cols','AUT(Benign)-all tasks','AUT(Attack)-all tasks'))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}".format('Mean',float(str(aut_average[8])[:5]), float(str(aut_average[9])[:5])))
    print("{:<20}  {:<20}  {:<20}".format('Variance',float(str(auc_std[8])[:5]), float(str(auc_std[9])[:5])))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format('Cols','Self_labels (Benign)', 'Self_labels (Attack)','Total (self-label)','Analyst_labels (Benign)', 'Analyst_labels (Attack)','Total (analyst-label)'))
    print("-"*80)

    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format('Mean',float(str(aut_average[4])),float(str(aut_average[5])),float(str(aut_average[5]+aut_average[4])),float(benign_mean),float(attack_mean),float(benign_mean)+float(attack_mean)) )
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format('Variance',float(str(auc_std[4])),float(str(auc_std[5])),'---',float(str(benign_std)),float(str(attack_std)),'---'))
    print("-"*80)
    
    print("-"*80)
    total_time = time.time()-start_time
    print("total execution time is %.3f seconds" % (total_time))
    print("avg execution time %.3f seconds"%(total_time/len(seed_list)))
