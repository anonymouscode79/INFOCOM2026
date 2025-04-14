import numpy as np
import torch
from math import ceil
from torchmetrics import Accuracy
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score,precision_recall_curve,auc
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from sklearn.manifold import TSNE
from matplotlib import cm
import os
import math
import matplotlib.pyplot as plt
# from utils import truncate
import pandas as pd
from datetime import datetime
def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n



def plot_tsne(y_test,yhat,test_embeddings,filename,perplexity):
    #code is adapted from https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42
    lr_fpr, lr_tpr, thresholds = roc_curve(y_test, yhat)
    optimal_idx = np.argmax(lr_tpr - lr_fpr)
    optimal_threshold = thresholds[optimal_idx]
    yhat= np.where(yhat >= optimal_threshold , 1, 0)

    tsne = TSNE(2, random_state=0,verbose=1,perplexity=perplexity,init='pca')
    tsne_proj = tsne.fit_transform(test_embeddings)
    tsne_proj = tsne.fit_transform(test_embeddings)
    
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = pyplot.subplots(figsize=(8,8))
    num_categories = 2
    colors = ['g','r']
    target_names=['Benign','Attack']
    for lab in range(num_categories):
        indices = yhat==lab
        indices = indices.flatten().tolist()
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=colors[lab], label = target_names[lab] ,alpha=0.5)
    ax.legend(fontsize='xx-large', markerscale=2)
    # pyplot.show()
    dir_path = filename[0]+"/"+filename[1]+"/"+filename[2]
    os.makedirs(dir_path,exist_ok=True)
    file_path = dir_path+"/"+filename[3]+"_ppt_"+str(perplexity)
    pyplot.savefig(file_path+".pdf",format='pdf')



def compute_results_new(y_test, lr_probs:np.ndarray,name,seed,task_id,type,ablation_dict={},run_order="",curr_train_task = 0,analyst_labels=50,path="output/new_method/"):
    auc_values = []


    # Define threshold
    threshold = 0.5

    # Convert probabilities to binary predictions
    y_pred = (lr_probs >= threshold).astype(int)

    # Calculate F1 scores
    f1_benign = f1_score(y_test, y_pred, zero_division=0, pos_label=0)
    f1_attack = f1_score(y_test, y_pred, zero_division=0, pos_label=1)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    # Calculate FPR and FNR for benign and attack samples
    fpr_benign = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_benign = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr_attack = fn / (fn + tp) if (fn + tp) > 0 else 0
    fnr_attack = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Calculate PR-AUC for attack samples (positive class)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs, pos_label=1)
    lr_auc_attack = auc(lr_recall, lr_precision)

    # Calculate PR-AUC for benign samples (negative class)
    lr_precision_benign, lr_recall_benign, _ = precision_recall_curve(y_test, 1 - lr_probs, pos_label=0)
    lr_auc_benign = auc(lr_recall_benign, lr_precision_benign)

    # Create a dictionary of results
    results = {
    "F1 Score (Benign)": f"{f1_benign:.3f}",
    "F1 Score (Attack)": f"{f1_attack:.3f}",
    "FPR (Benign)": f"{fpr_benign:.3f}",
    "FNR (Benign)": f"{fnr_benign:.3f}",
    "FPR (Attack)": f"{fpr_attack:.3f}",
    "FNR (Attack)": f"{fnr_attack:.3f}",
    "PR-AUC (Attack)": f"{lr_auc_attack:.3f}",
    "PR-AUC (Benign)": f"{lr_auc_benign:.3f}"
}

    # Convert the dictionary to a DataFrame
    results_df = pd.DataFrame([results])
    # Generate a unique filename with a timestamp
    # if run_order != "":     
    #     filename = f"{name}/{analyst_labels}/{run_order}/classification_metrics_seed_{type}{seed}_task_id_{task_id}_curr_task_{curr_train_task}.csv"
    # else:
    #     filename = f"{name}/{analyst_labels}/classification_metrics_seed_{type}{seed}_task_id_{task_id}.csv"
    os.makedirs(f"{path}{name}",exist_ok=True)
    filename = f"{path}{name}/classification_metrics_seed_{type}{seed}_task_id_{task_id}.csv"
    results_df.to_csv(filename)
    yhat = lr_probs
    # lr_probs = np.nan_to_num(lr_probs, copy=True, nan=0.0)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs,pos_label=1)
    # calculate scores
    lr_auc =  auc(lr_recall, lr_precision)
    # summarize scores
    auc_values.append(float("%.3f" %lr_auc))

    lr_precision, lr_recall, _ = precision_recall_curve(y_test, [1-x for x in lr_probs],pos_label=0)
    # calculate scores
    lr_auc =  auc(lr_recall, lr_precision)
    # summarize scores
    # print('PR-AUC(I)=%.3f' % ( lr_auc))
    auc_values.append(float ("%.3f" %lr_auc))
    lr_fpr, lr_tpr, thresholds = roc_curve(y_test, lr_probs)
    optimal_idx = np.argmax(lr_tpr - lr_fpr)
    optimal_threshold = thresholds[optimal_idx]
    

    
    ns_probs = [0 for _ in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    lr_auc = float (str (lr_auc)[:5])

    auc_values.append(lr_auc)

    # print("{:<20}  {:<20}  {:20}".format('PR-AUC(O)', 'PR-AUC(I)', 'ROC-AUC'))
    # print("-"*80)
    # print("{:<20}  {:<20}  {:<20}".format(auc_values[0],auc_values[1],auc_values[2]))
    # print("-"*80)

#     ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
#     lr_fpr, lr_tpr, thresholds = roc_curve(y_test, lr_probs)
#      # plot the roc curve for the model
#     pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
#     pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# # axis labels
#     pyplot.xlabel('False Positive Rate')    
#     pyplot.ylabel('True Positive Rate')
#     # show the legend
#     pyplot.legend()
#     # show the plot
#     pyplot.savefig('foo2.png')
#     pyplot.show()
    #yhat = yhat.ravel()
    #yhat = yhat.round()
    yhat= np.where(yhat >= optimal_threshold , 1, 0)

    # print("="*40)
    # print("accuracy    ", accuracy_score(y_test, yhat))
    # print("f1 score    ", f1_score(y_test, yhat))
    # print("precision    ", precision_score(y_test, yhat))
    # print("recall    ", recall_score(y_test, yhat))
    # print("="*40)
# Getting categorical encoding format ---- only use this if you're doing multiclass classification

    target_names = ['BENIGN', 'ATTACK']

# Printing out the confusion matrix
    # print(confusion_matrix(y_test, yhat))

    from sklearn.metrics import classification_report
    

    # print(classification_report(y_test, yhat, target_names = target_names, digits = 6))

    return auc_values,results

def compute_results(y_test, lr_probs):
    auc_values = []
    
    # print(lr_probs)
    yhat = lr_probs
    # print(lr_probs)
    # lr_probs = np.nan_to_num(lr_probs, copy=True, nan=0.0)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs,pos_label=1)
    # calculate scores
    lr_auc =  auc(lr_recall, lr_precision)
    # summarize scores
    # print('PR-AUC (O)=%.3f' % ( lr_auc))
    auc_values.append(float("%.3f" %lr_auc))

    lr_precision, lr_recall, _ = precision_recall_curve(y_test, [1-x for x in lr_probs],pos_label=0)
    # calculate scores
    lr_auc =  auc(lr_recall, lr_precision)
    # summarize scores
    # print('PR-AUC(I)=%.3f' % ( lr_auc))
    auc_values.append(float ("%.3f" %lr_auc))
    lr_fpr, lr_tpr, thresholds = roc_curve(y_test, lr_probs)
    optimal_idx = np.argmax(lr_tpr - lr_fpr)
    optimal_threshold = thresholds[optimal_idx]
    

    
    ns_probs = [0 for _ in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    lr_auc = float (str (lr_auc)[:5])
    # print("ROC-AUC=",lr_auc)
    auc_values.append(lr_auc)
    #
    print("{:<20}  {:<20}  {:20}".format('PR-AUC(O)', 'PR-AUC(I)', 'ROC-AUC'))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}".format(auc_values[0],auc_values[1],auc_values[2]))
    print("-"*80)

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, thresholds = roc_curve(y_test, lr_probs)
     # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
    pyplot.xlabel('False Positive Rate')    
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.savefig('foo2.png')
    pyplot.show()
    #yhat = yhat.ravel()
    #yhat = yhat.round()
    yhat= np.where(yhat >= optimal_threshold , 1, 0)

    # print("="*40)
    # print("accuracy    ", accuracy_score(y_test, yhat))
    # print("f1 score    ", f1_score(y_test, yhat))
    # print("precision    ", precision_score(y_test, yhat))
    # print("recall    ", recall_score(y_test, yhat))
    # print("="*40)
# Getting categorical encoding format ---- only use this if you're doing multiclass classification

    target_names = ['BENIGN', 'ATTACK']

# Printing out the confusion matrix
    # print(confusion_matrix(y_test, yhat))

    from sklearn.metrics import classification_report
    

    # print(classification_report(y_test, yhat, target_names = target_names, digits = 6))

    return auc_values



