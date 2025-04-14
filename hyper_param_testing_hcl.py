import subprocess
import os
import tempfile
import json
import argparse
import time
import numpy as np

if __name__ == "__main__":
    start_time=time.time()
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)') 
    parser.add_argument('--ds', type=str, default="api_graph", metavar='S',help='dataset name')
    parser.add_argument('--b_m', type=float, default=0.3, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='S',help='learning rate(default: 0.001)')
    parser.add_argument('--wd', type=float, default=1e-3, metavar='S',help='weight decay(default: 0.01)')
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
    parser.add_argument('--analyst_labels', type=int, metavar='S',default=50,help='No of labels from analysts (default:50)')
    parser.add_argument('--uncertainity', type=str, metavar='S',default="pseudo-loss",help='Sample selector')
    parser.add_argument('--family_info', type=bool, metavar='S',default=True,help='Include Family Info in HCL')
    parser.add_argument('--is_supervised',type=bool,metavar='S',default=True,help='Supervised or semi-supervised')
    args = parser.parse_args()
    auc_results = {}
    seed_list = [1,2,3]
    curr_dir = os.getcwd()
    for seed_value in seed_list:
        print("seed is",seed_value)
    hyperparameter_grid = {
    'learning_rate': [ 1e-6,1e-7],
    'weight_decay': [ 1e-6,1e-7],
    # Add other hyperparameters you want to tune
                               }
                               
    cutoffs= {
        'bodmas':5,
        'androzoo':12,
        'api_graph':35,
    }
    best_pr_auc = 0.0
    best_hyperparameters = {}

    gpu_index = 0  # To alternate between GPUs

    # Open a log file to write results
    dataset = args.ds
    log_file_path = os.path.join(curr_dir, f"hyper_parameter_test_results/{args.uncertainity}_after_fix_train/experiment_results_{dataset}_aut_test_after_1e-6_and_1e-7.log")
    with open(log_file_path, "w",buffering=1) as log_file:
            log_file.write(f"Running experiments for dataset: {dataset}\n")
            best_pr_auc = -float('inf')
            best_hyperparameters = {}
            for lr in hyperparameter_grid['learning_rate']:
                for w_d in hyperparameter_grid['weight_decay']:
                    pr_auc_list = []
                    for seed_value in seed_list:
                        log_file.write(f"LR: {lr} and W-D :{w_d}\n")
                        fd, temp_file_name = tempfile.mkstemp()  # Create temporary file
                        os.close(fd)  # Close the file

                        # Assign GPU alternately
           

                        # Run the subprocess with the current configuration
                        proc = subprocess.Popen([
                            "python3 "+
                            "SSCL_HCL_cade.py"+
                            f" --seed={seed_value}"+
                            f" --ds={dataset}"+
                            f" --family_info={args.family_info}"+
                            f" --is_supervised={args.is_supervised}"+
                            f" --uncertainity={args.uncertainity}"+
                            f" --lr={lr}"+
                            f" --wd={w_d}"+
                            f" --alpha={args.alpha}"+
                            f" --gpu={args.gpu}"+
                            f" --filename={temp_file_name}"+
                            f" --cos_dist={args.cos_dist}"+
                            f" --mode_val={args.mode_val}"+
                            f" --bool_gpm={args.bool_gpm}"+
                            f" --mem_strat={args.mem_strat}"+
                            f" --b_m={args.b_m}"+
                            f" --label_ratio={args.label_ratio}"+
                            f" --lab_samp_in_mem_ratio={args.lab_samp_in_mem_ratio}"+
                            f" --nps={args.nps}"+
                            f" --bma={args.bma}"+
                            f" --training_cutoff={cutoffs[dataset]}"+
                            f" --bool_closs={args.bool_closs}"+
                            f" --mlps={args.mlps}"
                        ], shell=True, cwd=curr_dir)

                        proc.communicate()

                        # Read results from the temporary file
                        with open(temp_file_name) as fp:
                            result = json.load(fp)
                            print(result[str(seed_value)][1][1])
                            pr_auc_outlier = float(result[str(seed_value)][1][3])
                            pr_auc_list.append(pr_auc_outlier)

                        os.unlink(temp_file_name)  # Remove the temporary file
                        log_file.write(f"Seed: {seed_value}, Result: {pr_auc_outlier}\n")

                    # Calculate average PR AUC across seeds
                    avg_pr_auc = np.mean(pr_auc_list)
                    log_file.write(f"Avg PR AUC AUT for lr: {lr}, weight_decay: {w_d} -> {avg_pr_auc}\n")

                    # Update the best hyperparameters if current results are better
                    if avg_pr_auc > best_pr_auc:
                        log_file.write(f"Found better results for lr: {lr} and weight_decay: {w_d} on dataset: {dataset}\n")
                        best_pr_auc = avg_pr_auc
                        best_hyperparameters = {
                            'best_pr_auc_o': best_pr_auc,
                            'learning_rate': lr,
                            'weight_decay': w_d
                        }

            # Store the best results for the current dataset
            log_file.write(json.dumps(best_hyperparameters, indent=4))

    print(f"Experiment results written to {log_file_path}")

    print("{:<20}  {:<20}".format('Argument','Value'))
    print("*"*80)
    for arg in vars(args):
        print("{:<20}  {:<20}".format(arg, getattr(args, arg)))
    print("*"*80)    
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20} {:<10}".format('seed','PR-AUC(O)', 'PR-AUC(I)', 'ROC-AUC','grad_norm_mean','grad_norm_variance'))
    print("*"*80)
    for key, value in auc_results.items():
        # print(key,value)
        pr_auc_o, pr_auc_1,roc_auc,grad_norm_mean,grad_norm_var = value
        print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format(key,pr_auc_o, pr_auc_1,roc_auc,grad_norm_mean,grad_norm_var))
    print("-"*80)
    auc_results_values = list(auc_results.values())
    auc_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*auc_results_values)]
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format('avg',float(str(auc_average[0])[:5]), float (str(auc_average[1])[:5]), float(str(auc_average[2])[:5]), float (str(auc_average[3])[:5]),float(str(auc_average[4])[:5])))
    print("-"*80)
    total_time = time.time()-start_time
    print("total execution time is %.3f seconds" % (total_time))
    print("avg execution time %.3f seconds"%(total_time/len(seed_list)))
    print("best hyperparameters are",best_hyperparameters)