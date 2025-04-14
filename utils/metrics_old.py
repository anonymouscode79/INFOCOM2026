import numpy as np
import torch
from math import ceil
from torchmetrics import Accuracy
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score,precision_recall_curve,auc
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot


def compute_results(y_test, lr_probs):
    yhat = lr_probs
    # print(lr_probs)
    # lr_probs = np.nan_to_num(lr_probs, copy=True, nan=0.0)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs,pos_label=1)
    # calculate scores
    lr_auc =  auc(lr_recall, lr_precision)
    # summarize scores
    print('precisionrecall aucfor outliers:  auc=%.3f' % ( lr_auc))

    lr_precision, lr_recall, _ = precision_recall_curve(y_test, [1-x for x in lr_probs],pos_label=0)
    # calculate scores
    lr_auc =  auc(lr_recall, lr_precision)
    # summarize scores
    print('precisionrecall auc for inliers:  auc=%.3f' % ( lr_auc))
    lr_fpr, lr_tpr, thresholds = roc_curve(y_test, lr_probs)
    optimal_idx = np.argmax(lr_tpr - lr_fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("tpr:",lr_tpr[optimal_idx])
    print("fpr:",lr_fpr[optimal_idx])
    print("optimum threshold",optimal_threshold)

    #lr_probs = np.where(yhat >= optimal_threshold , 1, 0)#.round()#[:,1]
    ns_probs = [0 for _ in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    print("auc suuve=",lr_auc)
    print('No Skill: ROC AUC=%.3f' % (ns_auc))

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

    print("="*40)
    print("accuracy    ", accuracy_score(y_test, yhat))
    print("f1 score    ", f1_score(y_test, yhat))
    print("precision    ", precision_score(y_test, yhat))
    print("recall    ", recall_score(y_test, yhat))
    print("="*40)
# Getting categorical encoding format ---- only use this if you're doing multiclass classification

    target_names = ['BENIGN', 'ATTACK']

# Printing out the confusion matrix
    print(confusion_matrix(y_test, yhat))

    from sklearn.metrics import classification_report
    

    print(classification_report(y_test, yhat, target_names = target_names, digits = 6))


