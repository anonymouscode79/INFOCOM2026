

from turtle import st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torch.optim.lr_scheduler import StepLR
from scipy.spatial.distance import cdist
from scipy import stats
import matplotlib.pyplot as plt
# from pytorchtools import EarlyStopping
import subprocess
import os
import tempfile
import numpy as np
import pandas as pd
import math
from utils.customdataloader import load_dataset,Tempdataset,compute_total_minority_testsamples,get_inputshape,load_teset
from utils.buffermemory import memory_update_equal_allocation,memory_update_equal_allocation2,memory_update_equal_allocation3
from utils.metrics import compute_results,plot_tsne,compute_results_new
from utils.utils import log,create_directories,trigger_logging,set_seed,get_gpu,load_model,EarlyStopping,GradientRejection
from utils.config.configurations import cfg
from utils.metadata import initialize_metadata


import time
import random
from math import floor
from collections import Counter
from sklearn.metrics import precision_recall_curve,auc
from sklearn.metrics import f1_score
from tqdm import tqdm
import itertools
import argparse
import json
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


from androzoo_data_set_info import androzoo
from api_graph_data_set_info import api_graph
from bodmas_data_set_info import bodmas
from ember_data_set_info import ember
### New Changes

encoder =None
list_self_labelled = {}
list_analys_labelled ={}


def get_dataset_info_local(dataset):
    if dataset == "api_graph":
        return api_graph
    elif dataset=="androzoo":
        return androzoo
    elif dataset=="bodmas":
        return bodmas
    elif dataset=="ember":
        return ember
##
scaler = MinMaxScaler()
memory_population_time=0
global_priority_list=dict()
local_priority_list=dict()
local_count = Counter()
classes_so_far = set()
full = set()
local_store = {}
global_count, local_count, replay_count,replay_individual_count = Counter(), Counter(),Counter(),Counter()
input_shape,task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,learning_rate,ecbrs_taskaware = None,None,None,None,None,None,None,None,None,None
replay_size,memory_size,minority_allocation,batch_size,device,pattern_per_exp,is_lazy_training,task_num = None,None,None,None,None,None,None,None
memory_X, memory_y, memory_y_name,ecbrs_taskaware_memory_X, ecbrs_taskaware_memory_y,ecbrs_taskaware_memory_y_name,memory_per_task = None,None,None,None,None,None,None
loss_fn,train_acc_metric = None,None
student_model1,student_model2,student_supervised,student_optimizer1,student_optimizer2,student_supervised_optimizer = None,None,None,None,None,None
teacher_model1,teacher_model2,teacher_supervised =None,None,None
pth_testset,testset_class_ids =None,None
test_x,test_y = None,None
val_x_all_tasks,val_y_all_tasks = None, None
image_resolution = None
bool_encode_anomaly,bool_encode_benign,load_whole_train_data=None,None,None
nc = 0
no_tasks = 0
grad_norm_dict = []
temp_norm =1
avg_rej_samples_per_unseen_task = []
avg_rej_samples_per_seen_task = []
global_train_loss,global_ce_loss,global_bce_loss = [],[],[]
task_change_indices = []
# labels_ratio, no_of_rand_samples, minority_alloc = None,None,None

# consecutive_otdd = []
owl_self_labelled_count_class_0 = 0
owl_self_labelled_count_class_1 = 0
owl_analyst_labelled_count_class_0 = 0
owl_analyst_labelled_count_class_1 = 0
truth_agreement_fraction_0, truth_agreement_fraction_1 = 0, 0

CI_list = []
avg_CI = None

### From December 1st
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from utils.losses.loss import MarginLoss, entropy

import torch

def maybe_unsqueeze(tensor):
  """
  Unsqueeze the tensor if it has a single dimension.

  Args:
    tensor: The input tensor.

  Returns:
    The input tensor, unsqueezed if it has a single dimension.
  """
  if len(tensor.shape) == 1:
    return tensor.unsqueeze(1)
  else:
    return tensor
  


def get_uncertainity( model, device, test_loader):
    model.eval()
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x) in enumerate(test_loader):
            x= x.to(device)
            output= model(x)
            prob = F.softmax(output, dim=1)
            #conf_mul = prob[0] * prob[1]
            if prob.size(0) == 1:
                conf_mul = prob.squeeze(0)[0] * prob.squeeze(0)[1]  # Squeeze and access elements
            else:
                conf_mul = prob[0] * prob[1] 
            confs = np.append(confs, conf_mul.cpu().numpy())
    mean_uncert = np.mean(confs)
    return mean_uncert
def apply_soft_threshold(cosine_similarities_matrix, initial_threshold=0.1, threshold_step=0.003, min_threshold=0.15):
    indices_with_high_similarity = []
    valid_indices =[]
    for i in range(len(cosine_similarities_matrix)):
        threshold = initial_threshold
        while threshold <= min_threshold:
            indices = np.where(cosine_similarities_matrix[i] < threshold)[0]  
            # indices = np.where(np.abs(cosine_similarities_matrix[i] - threshold) <= 0.01)[0] 

            if len(indices) > 0:
                indices_with_high_similarity.append([i,indices])
                valid_indices.append(i)
                break  # Move to the next unlabeled sample once we find at least one index
                
            threshold += threshold_step  # Reduce the threshold if no indices are found
        # If no indices found even with the minimum threshold, add an empty list (or handle as needed)
        # if len(indices) ! 0:
        #     # print(f"similarites max : {np.max(similarities)} and min : {np.min(similarities)} ")
        #     indices_with_high_similarity.append([])
    return indices_with_high_similarity,valid_indices
def major_representations(memory_y, indices_with_high_similarity, cosine_similarity_matrix):
    filtered_indices = []
    majority_labels = []
    max_similarity_indices = []  # To store the index with the highest cosine similarity

    for i, group_indices in indices_with_high_similarity:
        # Get the labels for the current group of indices
        group_labels = memory_y[group_indices]

        # Find the majority label
        unique_labels, counts = np.unique(group_labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        majority_labels.append(majority_label)

        # Filter indices with the majority label
        filtered_group = group_indices[group_labels == majority_label]
        filtered_indices.append(filtered_group)

        highest_similarity_index = group_indices[np.argmax(cosine_similarity_matrix[i][filtered_group])]
        max_similarity_indices.append(highest_similarity_index)

    return filtered_indices, majority_labels, max_similarity_indices
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def load_metadata(dataset_name,l_rate,w_decay):
    global task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,learning_rate,input_shape,ecbrs_taskaware,image_resolution,pth_testset,testset_class_ids,bool_encode_anomaly,bool_encode_benign,load_whole_train_data
    global replay_size,memory_size,minority_allocation,epochs,batch_size,device,pattern_per_exp,is_lazy_training,ecbrs_taskaware_memory_X, ecbrs_taskaware_memory_y,ecbrs_taskaware_memory_y_name,memory_per_task
    ds = get_dataset_info_local(dataset_name)
    label = ds.label
    cfg.avalanche_dir = False
    set_cl_strategy_name(1)
    no_tasks = ds.no_tasks
    metadata_dict = initialize_metadata(label)
    temp_dict = metadata_dict[no_tasks]
    task_order = temp_dict['task_order']
    class_ids = temp_dict['class_ids']
    minorityclass_ids = temp_dict['minorityclass_ids']
    pth = temp_dict['path']
    if 'path_testset' in temp_dict:
        pth_testset = temp_dict['path_testset']
        testset_class_ids = temp_dict['testset_class_ids']
    tasks_list = temp_dict['tasks_list']
    task2_list = temp_dict['task2_list']
    replay_size = ds.replay_size
    memory_size = ds.mem_size
    # memory_size = ds.mem_size_semi_supervised
    minority_allocation = ds.minority_allocation
    batch_size = ds.batch_size
    device = cfg.device
    learning_rate = l_rate# 0.001#ds.learning_rate
    no_tasks = ds.no_tasks
    pattern_per_exp = ds.pattern_per_exp
    is_lazy_training = ds.is_lazy_training
    ecbrs_taskaware = ds.taskaware_ecbrs
    bool_encode_anomaly = False
    bool_encode_benign = ds.bool_encode_benign
    load_whole_train_data = ds.load_whole_train_data
    if ecbrs_taskaware:
        minority_allocation = ds.taskaware_minority_allocation
    input_shape = get_inputshape(pth,class_ids)
    compute_total_minority_testsamples(pth=pth,dataset_label=label,minorityclass_ids=minorityclass_ids,no_tasks=no_tasks)
    load_model_metadata(w_decay)
    create_directories(label)
    trigger_logging(label=label)
    image_resolution = ds.image_resolution
    if ecbrs_taskaware:
        set_cl_strategy_name(2)
        ecbrs_taskaware_memory_X = np.zeros((1,input_shape))
        ecbrs_taskaware_memory_y = np.zeros(1)
        ecbrs_taskaware_memory_y_name = np.zeros(1)
        memory_per_task = ds.pattern_per_exp
        memory_size = memory_per_task

def load_model_metadata(w_decay):
    log("loading model parameter")
    global student_model1,student_model2,student_supervised,student_optimizer1,student_optimizer2,student_supervised_optimizer,loss_fn,train_acc_metric,input_shape
    global teacher_model1,teacher_model2,teacher_supervised

    w_d = w_decay
    student_model1 = load_model(label=label,inputsize=get_inputshape(pth,class_ids))
    print(student_model1)
    teacher_model1 = load_model(label=label,inputsize=get_inputshape(pth,class_ids))
    student_optimizer1 = torch.optim.SGD(student_model1.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=w_d)
    student_model1 = student_model1.to(device)
    teacher_model1 = teacher_model1.to(device)

    if mlps >=2:
        student_model2 = load_model(label=label+"_student2",inputsize=get_inputshape(pth,class_ids))
        teacher_model2 = load_model(label=label+"_student2",inputsize=get_inputshape(pth,class_ids))
        student_optimizer2 = torch.optim.SGD(student_model2.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=w_d)
        student_model2 = student_model2.to(device)
        teacher_model2 = teacher_model2.to(device)
    if mlps == 3:
        student_supervised = load_model(label=label+"_supervised",inputsize=get_inputshape(pth,class_ids))      
        teacher_supervised = load_model(label=label+"_supervised",inputsize=get_inputshape(pth,class_ids))
        student_supervised_optimizer = torch.optim.SGD(student_supervised.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=w_d)
        student_supervised = student_supervised.to(device)
        teacher_supervised = teacher_supervised.to(device)

    
    
    
    
    loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    # train_acc_metric = Accuracy(task='multiclass',                                           
    #                                  num_classes=2).to(device)

def set_cl_strategy_name(strategy_id):
    if strategy_id == 0:
        cfg.clstrategy = "CBRS"            
    elif strategy_id == 1:
        cfg.clstrategy = "ECBRS"
    elif strategy_id == 2:
        cfg.clstrategy = "ECBRS_taskaware"    
     
          

def initialize_buffermemory(tasks,mem_size):
    global memory_X, memory_y, memory_y_name
    attack_class_indicies_list = []
    attack_class_indicies = []
    initial_X, initial_y, initial_yname,_ = tasks[0]
    unique_class = [int(x) for x in np.unique(initial_yname)]
    attack_class = [int(x) for x in minorityclass_ids]
    # common_attack_class = [x for x in attack_class if x in unique_class]
    common_attack_class = unique_class
    # print("common attack classes",common_attack_class)
    # exit()
    for idx,class_idx in enumerate(common_attack_class):
         indices = list(np.where(initial_yname == int(class_idx))[0])
         attack_class_indicies_list.insert(idx,indices)


    for concat_list in itertools.zip_longest(*attack_class_indicies_list):
        attack_class_indicies.extend(list(concat_list))

    attack_class_indicies = [x for x in attack_class_indicies if x is not None]  
    if len(attack_class_indicies) > mem_size:
        attack_class_indicies = attack_class_indicies[0:mem_size]   

    memory_X, memory_y, memory_y_name = initial_X[attack_class_indicies,:], initial_y[attack_class_indicies], initial_yname[attack_class_indicies]
    







def update_mem_samples_indexdict(memorysamples):
    global local_store
    for idx,class_ in enumerate(memorysamples):
        if class_ in local_store :
            local_store[class_].append(idx)
        else:
            local_store[class_] = [idx]





def get_representation_matrix (net, device, x, y=None,rand_samples=1000): 
    # Collect activations by forward pass
    # benign_indices = random.sample(list(range(x.shape[0])),min(100000,x.shape[0]))
    # print(x.shape[0])
    
    
    print("number of rand samples",rand_samples)
    print(x.shape[0])
    
    benign_indices = (np.where(y == int(0))[0]).tolist()
    attack_indices = list(set(range(0,x.shape[0]))-set(benign_indices))
    a_min_length = min(len(attack_indices),rand_samples)
    b_min_length = min(len(benign_indices),rand_samples)
    print(a_min_length,b_min_length)
    random.shuffle(benign_indices)
    random.shuffle(attack_indices)
    # benign_indices = benign_indices[0:b_min_length]
    # benign_indices.extend(attack_indices[0:a_min_length])
    benign_indices = attack_indices[0:a_min_length]
    
                                    

    
    b=benign_indices
    rand_samples = len(benign_indices)
    x=torch.tensor(x,dtype=torch.float32).to(device)
    example_data = x[b]#.view(-1,70)
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    batch_list=[int(rand_samples),int(rand_samples),int(rand_samples),int(rand_samples),int(rand_samples),rand_samples,rand_samples] 
    mat_list=[] # list contains representation matrix of each layer
    act_key=list(net.act.keys())
    print("keys are",act_key)

    for i in range(len(act_key)):
        bsz=batch_list[i]
        act = net.act[act_key[i]].detach().cpu().numpy()
        activation = act[0:bsz].transpose()
        mat_list.append(activation)

    
    return mat_list


def update_GPM (model, mat_list, threshold, feature_list=[],):
    # print ('Threshold: ', threshold) 
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            feature_list.append(U[:,0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                # print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                continue
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))  
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
    
    return feature_list  




def split_a_task(task,lab_ratio,task_class_ids=None):
    global batch_size
    labeled_indices,unlabeled_indices = [],[]
    # print("task classes",task_class_ids)
    
    X,y,y_classname = task[0][0],task[0][1],task[0][2]
    # print("y class name",Counter(y_classname))
    for class_idx in np.unique(y_classname):
        indices = (np.where(y_classname == int(class_idx))[0]).tolist()
        # random.shuffle(indices)
        labeled_index = max(1,floor(len(indices)*lab_ratio))
        labeled_indices.extend(indices[0:labeled_index])
        unlabeled_indices.extend(indices[labeled_index:])
    random.shuffle(labeled_indices)    
    random.shuffle(unlabeled_indices)

    # print("****************************")
    # # labeled_indices.sort()
    # # print(labeled_indices)
    # print("labelled size",len(labeled_indices))
    # print("ulabelled size",len(unlabeled_indices))

    return labeled_indices,unlabeled_indices


class dataset(Dataset):

    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
 
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.length


def compute_distill_loss(unlabeled_pre,unlabeled_x):
    
    global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised

    if image_resolution is not None:
       unlabeled_x = unlabeled_x.reshape(image_resolution)

    if mlps == 1:
        models = [student_model1]
    elif mlps == 2:
        models = [student_model1,student_model2]
    else:
        models = [student_model1,student_model2,student_supervised]    

    #models = [teacher_model1,teacher_model2]#,teacher_supervised] 
    
    class_probs = []  

    with torch.no_grad():
        for model in models:
            outputs = torch.softmax(model(unlabeled_x), dim=1)
            class_probs.append(outputs)

    avg_probs = torch.stack(class_probs).mean(dim=0)
    predicted_labels = torch.argmax(avg_probs, dim=1) 
    unlabeled_gt = F.one_hot(predicted_labels,2)
    # unlabeled_gt = model(unlabeled_x)
    distillation_loss = loss_fn(unlabeled_pre.float(),unlabeled_gt.float())

    return [distillation_loss,predicted_labels]


    
# Contrastive loss function

def contrastive_loss(anchor_representations, positive_representations, negative_representations, temperature=0.5):
    # Combine representations for efficient matrix operations
    # print(anchor_representations.shape,positive_representations.shape,negative_representations.shape)
    all_representations = torch.cat([anchor_representations, positive_representations, negative_representations], dim=0)

    # Calculate cosine similarities in a single matrix operation
    similarities = torch.mm(all_representations, all_representations.T) / temperature

    # Extract relevant similarities efficiently
    positive_similarities = similarities[:anchor_representations.shape[0], anchor_representations.shape[0]:anchor_representations.shape[0] * 2]
    negative_similarities = similarities[:anchor_representations.shape[0], anchor_representations.shape[0] * 2:]

    # Calculate loss with optimized operations
    log_positive_similarities = torch.log(positive_similarities + 1e-8)
    negative_loss = torch.logsumexp(negative_similarities, dim=-1).mean()
    loss = -log_positive_similarities.mean() + negative_loss
    return loss

def contrastive_loss_cluster(anchor_representations, positive_representations, negative_representations):
    # Combine representations for efficient matrix operations
    # print(anchor_representations.shape,positive_representations.shape,negative_representations.shap
    
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if labels == None:
            raise ValueError('Need to define labels in DistanceLoss')

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        # similar masks
        # mask_{i,j}=1 if sample j has the same class as sample i.
        binary_labels = binary_cat_labels[:, 1].view(-1, 1)
        # mask: both malware, or both benign
        binary_mask = torch.eq(binary_labels, binary_labels.T).float().to(device)
        # multi_mask: same malware family, or benign
        multi_mask = torch.eq(labels, labels.T).float().to(device)
        # malware but not the same family. does not have benign.
        other_mal_mask = binary_mask - multi_mask
        # both benign samples
        ben_labels = torch.logical_not(binary_labels).float().to(device)
        same_ben_mask = torch.matmul(ben_labels, ben_labels.T)
        # same malware family mask
        same_mal_fam_mask = multi_mask - same_ben_mask
        
        # logging.debug("=== new batch ===")
        # pseudo loss
        if self.reduce == 'none':
            tmp = other_mal_mask
            other_mal_mask = same_mal_fam_mask
            same_mal_fam_mask = tmp
            # debug
            # split_index = torch.nonzero(split, as_tuple=True)[0]
            # logging.debug(f'split_index, {split_index}')
        # logging.debug(f'binary_labels {binary_labels}')
        # logging.debug(f'binary_mask {binary_mask}')
        # logging.debug(f'labels {labels}')
        # logging.debug(f'multi_mask {multi_mask}')
        # logging.debug(f'other_mal_mask = binary_mask - multi_mask {other_mal_mask}')
        # logging.debug(f'ben_labels {ben_labels}')
        # logging.debug(f'same_ben_mask {same_ben_mask}')
        # logging.debug(f'same_mal_fam_mask = multi_mask - same_ben_mask {same_mal_fam_mask}')
        
        # dissimilar mask. malware vs benign binary labels
        binary_negate_mask = torch.logical_not(binary_mask).float().to(device)
        # multi_negate_mask = torch.logical_not(multi_mask).float().to(device)

        # mask-out self-contrast cases
        diag_mask = torch.logical_not(torch.eye(batch_size)).float().to(device)
        # similar mask
        binary_mask = binary_mask * diag_mask
        multi_mask = multi_mask * diag_mask
        other_mal_mask = other_mal_mask * diag_mask
        same_ben_mask = same_ben_mask * diag_mask
        same_mal_fam_mask = same_mal_fam_mask * diag_mask

        if split is not None:
            split_index = torch.nonzero(split, as_tuple=True)[0]
            binary_negate_mask[:, split_index] = 0
            binary_mask[:, split_index] = 0
            multi_mask[:, split_index] = 0
            other_mal_mask[:, split_index] = 0
            same_ben_mask[:, split_index] = 0
            same_mal_fam_mask[:, split_index] = 0
        x = features
        y = features
        x_norm = x.norm(dim=1, keepdim=True)
        y_norm = y.norm(dim=1).T
        distance_matrix = x_norm * x_norm + y_norm * y_norm - 2 * x.mm(y.T)
        distance_matrix = torch.maximum(torch.tensor(1e-10), distance_matrix)

        if self.sample_reduce == 'mean' or self.sample_reduce == None:
            if weight == None:
                sum_same_ben = torch.maximum(
                                    torch.sum(same_ben_mask * distance_matrix, dim=1) - \
                                            same_ben_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                sum_other_mal = torch.maximum(
                                    torch.sum(other_mal_mask * distance_matrix, dim=1) - \
                                            other_mal_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                sum_same_mal_fam = torch.sum(same_mal_fam_mask * distance_matrix, dim=1)
                sum_bin_neg = torch.maximum(
                                    binary_negate_mask.sum(1) * torch.tensor(2 * margin) - \
                                            torch.sum(binary_negate_mask * distance_matrix,
                                                    dim=1),
                                    torch.tensor(0))
                # logging.debug(f'sum_same_ben {sum_same_ben}, same_ben_mask.sum(1) {same_ben_mask.sum(1)}')
                # logging.debug(f'sum_other_mal {sum_other_mal}, other_mal_mask.sum(1) {other_mal_mask.sum(1)}')
                # logging.debug(f'sum_same_mal_fam {sum_same_mal_fam}, same_mal_fam_mask.sum(1) {same_mal_fam_mask.sum(1)}')
                # logging.debug(f'sum_bin_neg {sum_bin_neg}, binary_negate_mask.sum(1) {binary_negate_mask.sum(1)}')
            # weighted loss
            else:
                weight_matrix = torch.matmul(weight.view(-1, 1), weight.view(1, -1)).to(device)
                sum_same_ben = torch.maximum(
                                    torch.sum(same_ben_mask * distance_matrix * weight_matrix, dim=1) - \
                                            same_ben_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                sum_other_mal = torch.maximum(
                                    torch.sum(other_mal_mask * distance_matrix * weight_matrix, dim=1) - \
                                            other_mal_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                sum_same_mal_fam = torch.sum(same_mal_fam_mask * distance_matrix * weight_matrix, dim=1)
                weight_prime = torch.div(1.0, weight)
                weight_matrix_prime = torch.matmul(weight_prime.view(-1, 1), weight_prime.view(1, -1)).to(device)
                sum_bin_neg = torch.maximum(
                                    binary_negate_mask.sum(1) * torch.tensor(2 * margin) - \
                                            torch.sum(binary_negate_mask * distance_matrix * weight_matrix_prime,
                                                    dim=1),
                                    torch.tensor(0))
            loss = sum_same_ben / torch.maximum(same_ben_mask.sum(1), torch.tensor(1)) + \
                    sum_other_mal / torch.maximum(other_mal_mask.sum(1), torch.tensor(1)) + \
                    sum_same_mal_fam / torch.maximum(same_mal_fam_mask.sum(1), torch.tensor(1)) + \
                    sum_bin_neg / torch.maximum(binary_negate_mask.sum(1), torch.tensor(1))
        elif self.sample_reduce == 'max':
            max_same_ben = torch.maximum(
                                torch.amax(same_ben_mask * distance_matrix, 1) - \
                                        torch.tensor(margin),
                                torch.tensor(0))
            max_other_mal = torch.maximum(
                                torch.amax(other_mal_mask * distance_matrix, 1) - \
                                        torch.tensor(margin),
                                torch.tensor(0))
            max_same_mal_fam = torch.amax(same_mal_fam_mask * distance_matrix, 1)
            max_bin_neg = torch.maximum(
                                torch.tensor(2 * margin) - \
                                        torch.amin(binary_negate_mask * distance_matrix, 1),
                                torch.tensor(0))
            loss = max_same_ben + max_other_mal + max_same_mal_fam + max_bin_neg
        else:
            raise Exception(f'sample_reduce = {self.sample_reduce} not implemented yet.')

        if self.reduce == 'mean':
            loss = loss.mean()
        
        return loss

    


def construct_positive_negative_samples(batch_data, batch_labels):
    batch_size = batch_data.shape[0]

    # Pre-compute indices for positive and negative samples for each data point
    pos_neg_indices = torch.zeros((batch_size, 2), dtype=torch.long)
    for i in range(batch_size):
        
        same_class_indices = (batch_labels == batch_labels[i]).nonzero(as_tuple=True)[0]
        different_class_indices = (batch_labels != batch_labels[i]).nonzero(as_tuple=True)[0]
        pos_neg_indices[i, 0] = torch.randint(len(same_class_indices), (1,))[0]
        pos_neg_indices[i, 1] = torch.randint(len(different_class_indices), (1,))[0]

    # Get positive and negative samples directly using pre-computed indices
    positive_samples = batch_data[pos_neg_indices[:, 0]]
    negative_samples = batch_data[pos_neg_indices[:, 1]]

    # print("sample shape",positive_samples.shape,negative_samples.shape)
    return positive_samples, negative_samples

def construct_positive_negative_samples_from_memory(batch_labels):
    batch_size = batch_labels.shape[0]
    memory_X_tensor,memory_y_tensor = torch.from_numpy(memory_X).to(device),torch.from_numpy(memory_y).to(device)

    # Pre-compute indices for positive and negative samples for each data point
    pos_neg_indices = torch.zeros((batch_size, 2), dtype=torch.long)
    # print("mem_",memory_y)
    # print(batch_labels)
    for i in range(batch_size):
        
        same_class_indices = (memory_y_tensor == batch_labels[i]).nonzero(as_tuple=True)[0]
        different_class_indices = (memory_y_tensor != batch_labels[i]).nonzero(as_tuple=True)[0]
        pos_neg_indices[i, 0] = torch.randint(len(same_class_indices), (1,))[0]
        pos_neg_indices[i, 1] = torch.randint(len(different_class_indices), (1,))[0]

    # Get positive and negative samples directly using pre-computed indices
    positive_samples = memory_X_tensor[pos_neg_indices[:, 0]]
    negative_samples = memory_X_tensor[pos_neg_indices[:, 1]]

    # print("sample shape",positive_samples.shape,negative_samples.shape)
    return positive_samples, negative_samples    



def sample_batch_from_memory(mem_batchsize,minority_alloc):
    if mem_batchsize > 0:
        majority_class_idices,minority_class_indices = [],[]
        global memory_X,memory_y,memory_y_name,minorityclass_ids
        minority_classes = [int(class_idx) for class_idx in minorityclass_ids]
        unique_class = np.unique(memory_y_name).tolist()
        majority_class = list(set(unique_class)-set(minority_classes))
    
        for class_idx in majority_class:
            indices = (np.where(memory_y_name == int(class_idx))[0]).tolist()
            majority_class_idices.extend(indices)

    
        minority_class_indices = list(set(range(0,memory_X.shape[0]))-set(majority_class_idices))
        minority_offset = floor(mem_batchsize*minority_alloc)
        majority_offset = mem_batchsize-minority_offset
        select_indices = min(minority_offset,len(minority_class_indices))
        select_indices = max(1, select_indices)
        # weights = np.array([1-(1 / (i + 1)) for i in range(len(minority_class_indices))])
        # weights /= np.sum(weights)
        # weights = 1-weights
        # weights /= np.sum(weights)
        # minority_class_indices = np.random.choice(minority_class_indices, size=select_indices, replace=False, p=weights).tolist()
        minority_class_indices = random.sample(minority_class_indices,select_indices)
        select_indices = min(majority_offset,len(majority_class_idices))
        select_indices = max(1, select_indices)
        # print(majority_class_idices,select_indices)
        # weights = np.array([1-(1 / (i + 1)) for i in range(len(majority_class_idices))])
        # weights /= np.sum(weights)
        # weights = 1-weights
        # weights /= np.sum(weights)
        # majority_class_idices = np.random.choice(majority_class_idices, size=select_indices, replace=False, p=weights).tolist()
        majority_class_idices = random.sample(majority_class_idices,select_indices)
        minority_class_indices.extend(majority_class_idices)
        # select_indices = min(mem_batchsize,memory_X.shape[0])
        # minority_class_indices = random.sample(list(range(0,memory_X.shape[0])),select_indices)
    
    # print(minority_class_indices)
    

        return memory_X[minority_class_indices],memory_y[minority_class_indices],memory_y_name[minority_class_indices]
    
def owl_data_labeling_strategy_old(X, y, y_classname,task_id,  unseen_task=True):

    global owl_self_labelled_count_class_0, owl_self_labelled_count_class_1
    global owl_analyst_labelled_count_class_0, owl_analyst_labelled_count_class_1
    global truth_agreement_fraction_0,truth_agreement_fraction_1
    global avg_CI

    print(f'X shape: {X.shape}')
    dummy_target_label = torch.zeros(X.shape[0])
    data_loader = torch.utils.data.DataLoader(dataset(X,dummy_target_label),
                                            batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                            num_workers=0)
        
    predictions,predicted_labels = None,None

    if mlps == 1:
        models = [student_model1]
    elif mlps == 2:
        models = [student_model1,student_model2]
    else:
        models = [student_model1,student_model2,student_supervised]
    
    for data, _ in data_loader:
        class_probs = [] 
        with torch.no_grad():
            for model in models:
                outputs = torch.softmax(model(data.to(device)), dim=1)
                class_probs.append(outputs)
        
            pred = torch.stack(class_probs).mean(dim=0)#[:,1].reshape(target.shape)
            if predictions is None:
                predictions = pred
                predicted_labels = torch.argmax(pred,dim=1)
            else:
                predictions = torch.cat((predictions,pred),dim=0)   
                predicted_labels = torch.cat((predicted_labels,torch.argmax(pred,dim=1)),dim=0) 

    # Step 2: Extracting the high confidence samples for class 0 and class 1 respectively 
    class_0_indices = ((predicted_labels == 0).nonzero(as_tuple=False)[:, 0]).detach().cpu().numpy()
    class_1_indices = ((predicted_labels == 1).nonzero(as_tuple=False)[:, 0]).detach().cpu().numpy()
    
    print(f'Number of predicted 0s: {len(class_0_indices)}')
    print(f'Number of predicted 1s: {len(class_1_indices)}')

    total_samples = X.shape[0]    
    est_class_1_samples = int(total_samples*avg_CI)    
    est_class_0_samples = total_samples - est_class_1_samples
    print(f'Estimated no. of class 0 samples = {est_class_0_samples}')
    print(f'Estimated no. of class 1 samples = {est_class_1_samples}')
    
    sorted_pred_class_0 = torch.sort(predictions[class_0_indices, 0], dim=0, descending=True)    
    top_class_0_indices = sorted_pred_class_0[1][sorted_pred_class_0[0] > 0.6].detach().cpu()
    if len(top_class_0_indices) > int(labels_ratio*est_class_0_samples):
        top_class_0_indices = top_class_0_indices[:int(labels_ratio*est_class_0_samples)]
    print(f'(few) Highest confidence prediction values (class 0): {sorted_pred_class_0[0][:4]}')

    # print(predictions[class_1_indices, 1])
    sorted_pred_class_1 = torch.sort(predictions[class_1_indices, 1], dim=0, descending=True)
    top_class_1_indices = (sorted_pred_class_1[1][sorted_pred_class_1[0] > 0.5]).detach().cpu()
    if len(top_class_1_indices) > int(labels_ratio*est_class_1_samples):
        top_class_1_indices = top_class_1_indices[:int(labels_ratio*est_class_1_samples)]
    print(f'(few) Highest confidence prediction values (class 1): {sorted_pred_class_1[0][:4]}')
    
    labeled_X,labeled_y,labeled_y_classname = None, None, None
    X_unlab = None
    unlabeled_indicies, labeled_indicies = np.array([]), np.array([])
    member_inference_class_0 = 0 # dummy assignment
    member_inference_class_1 = 1 # dummy assignment
    selection_count_0, selection_count_1 = 0, 0
    n_agreements_0, n_agreements_1 = 0, 0
    curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1 = 0, 0

    # Computing the sample means for each class in memory
    sample_means = None
    associated_label = []

    class_in_memory = np.unique(memory_y_name)
    # print(f'\nclasses in memory: {class_in_memory}')

    for class_idx in class_in_memory:
        indices = np.where(memory_y_name == int(class_idx))[0]
        if str(int(class_idx)) in minorityclass_ids:
            associated_label.append(1)
        else:
            associated_label.append(0)

        if sample_means is None:
            sample_means = torch.mean(torch.tensor(memory_X[indices]), dim=0).unsqueeze(0)
        else:
            sample_means = torch.cat((sample_means, torch.mean(torch.tensor(memory_X[indices]), dim=0).unsqueeze(0)), dim=0)
    associated_label = np.array(associated_label)

    # Setting unique y_classname labels for the new unseen labelled data (for buffer memory storage purpose)
    if unseen_task:
        attack_y_name = np.max(class_in_memory) + 1
        benign_y_name = np.max(class_in_memory) + 2
        print(f'New classes added to memory: {attack_y_name}, {benign_y_name}')

    if len(top_class_0_indices) != 0:

        top_class_0_data = X[class_0_indices[top_class_0_indices]]
        top_class_0_truth = y[class_0_indices[top_class_0_indices]]
        top_class_0_y_classname = y_classname[class_0_indices[top_class_0_indices]]

        # Member inference of the top samples based on distance from buffer memory samples
        start_inference = time.time()
        cos_dist = cdist(top_class_0_data, memory_X,'cosine')   
        # sorted_indices_temp = np.argsort(cos_dist, axis=1)[::-1]
        # top_k_indices = sorted_indices_temp[:, :1000]
        # # Create boolean mask with True for top k indices, False otherwise
        # mask = np.zeros_like(cos_dist, dtype=bool)
        # mask[np.arange(len(cos_dist))[:, None], top_k_indices] = True
        # filtered_indices = mask
        # # filtered_indices = cos_dist <max_value
        # row_indices_to_keep = np.any(filtered_indices, axis=1)
        # filtered_arr = filtered_indices[row_indices_to_keep]  
        # top_class_0_indices = top_class_0_indices[row_indices_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_0_truth = top_class_0_truth[row_indices_to_keep]#removes the indices whose cosine distance > 0.2        
        # maj_labels = []
        # for row in filtered_arr:
        #     maj_labels.append(stats.mode(memory_y[row])[0])
        # maj_labels = np.array(maj_labels)          
        # member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
        # member_inference_class_0 = np.asarray(maj_labels)
        maj_labels = []
        row_indices_to_keep = []
        percentage_mode_value_contributors = []
        Avg_sample_support = []
        Avg_sample_support_counter = 0
        filtered_indices = cos_dist < cos_dist_ip
        print(top_class_0_data.shape,filtered_indices.shape)
        row_indices_to_keep = np.where(np.any(filtered_indices, axis=1))[0]
        rows_to_keep = np.any(filtered_indices, axis=1)
        for i in range(filtered_indices.shape[0]):
            valid_indices = np.where(filtered_indices[i])[0]
            if valid_indices.size > 0:
                Avg_sample_support_counter += 1
                Avg_sample_support.append(valid_indices.size)
                Mode_value_and_count = stats.mode(memory_y[valid_indices])
                Mode_value_percentage = (Mode_value_and_count[1]/valid_indices.size)*100
                if Mode_value_percentage > mode_value:
                    maj_labels.append(Mode_value_and_count[0])
                    percentage_mode_value_contributors.append(Mode_value_percentage)
                else:
                    maj_labels.append(1) 
                
                # rows_to_keep.append(True)
            else:
                maj_labels.append(1)#Adding a flipped label for class 0 samples as no confident labels (cost dis <0.2) found in the memory    
                # rowrows_to_keeps_to_keep.append(False)
        print(len(maj_labels))      
        # exit()  
        maj_labels = np.array(maj_labels)
        member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
        # member_inference_class_0 = np.asarray(maj_labels)
        print("Average number of sample support for Attack is", stats.tmean(Avg_sample_support), stats.tstd(Avg_sample_support))
        print("Percentage of samples contributed to each Attack sample is",stats.tmean(percentage_mode_value_contributors),stats.tstd(percentage_mode_value_contributors))
        # top_class_0_indices = top_class_0_indices[rows_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_0_truth = top_class_0_truth[rows_to_keep]#removes the indices whose cosine distance > 0.2           

       

        
        end_inference = time.time()
        print(f'\nNumber of class 0 agreements (between model and member inference): {np.sum(member_inference_class_0 == 0)}/{len(member_inference_class_0)} - ({np.sum(member_inference_class_0 == 0)*100./len(member_inference_class_0):.3f}%)')
        print(f'Number of class 0 common agreements with ground truth: {np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)}/{np.sum(member_inference_class_0 == 0)} - ({np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)*100./np.sum(member_inference_class_0 == 0):.3f}%)')
        print(f'Time taken for member inference = {end_inference - start_inference}seconds')

        n_agreements_0 = np.sum(member_inference_class_0 == 0)
        curr_truth_agreement_fraction_0 = np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)/np.sum(member_inference_class_0 == 0)
        if math.isnan(curr_truth_agreement_fraction_0):
            curr_truth_agreement_fraction_0 = 0

        # 0-(self)labelled data
        if unseen_task:
            if n_agreements_0 > 0:
                if truth_agreement_fraction_0 is None or math.isnan(truth_agreement_fraction_0):
                    truth_agreement_fraction_0 = 1
                selection_count_0 = int(n_agreements_0*truth_agreement_fraction_0)
                selected_0_indices = np.random.choice(top_class_0_indices[member_inference_class_0 == 0], size=selection_count_0, replace=False)
                labeled_indicies = np.hstack((labeled_indicies, selected_0_indices))

                labeled_X = np.vstack((labeled_X, X[selected_0_indices])) if labeled_X is not None else X[selected_0_indices]
                labeled_y = np.hstack((labeled_y, [0]*selection_count_0)) if labeled_y is not None else [0]*selection_count_0
                labeled_y_classname = np.hstack((labeled_y_classname, [benign_y_name]*selection_count_0)) if labeled_y_classname is not None else [benign_y_name]*selection_count_0
                print(f'No. of self-labelled samples (class 0): {selection_count_0}')
                list_self_labelled.setdefault(f"{task_id}", {})["Benign"] = selection_count_0
                owl_self_labelled_count_class_0 += selection_count_0
            else:
                print('No. of self-labelled samples (class 0): 0')
                list_self_labelled.setdefault(f"{task_id}", {})["Benign"] = selection_count_0

    if len(top_class_1_indices) > 1:#!= 0:
    
        top_class_1_data = X[class_1_indices[top_class_1_indices]]
        top_class_1_truth = y[class_1_indices[top_class_1_indices]]
        top_class_1_y_classname = y_classname[class_1_indices[top_class_1_indices]]
 
        # Member inference of the top samples based on distance from sample means
        start_inference = time.time()
        cos_dist = cdist(top_class_1_data, memory_X,'cosine')   
        # sorted_indices_temp = np.argsort(cos_dist, axis=1)[::-1]
        # top_k_indices = sorted_indices_temp[:, :1000]
        # # Create boolean mask with True for top k indices, False otherwise
        # mask = np.zeros_like(cos_dist, dtype=bool)
        # mask[np.arange(len(cos_dist))[:, None], top_k_indices] = True        
        # filtered_indices = mask
        # # filtered_indices = cos_dist <max_value
        # row_indices_to_keep = np.any(filtered_indices, axis=1)
        # filtered_arr = filtered_indices[row_indices_to_keep]  
        # top_class_1_indices = top_class_1_indices[row_indices_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_1_truth = top_class_1_truth[row_indices_to_keep]#removes the indices whose cosine distance > 0.2        
        # maj_labels = []
        # for row in filtered_arr:
        #     maj_labels.append(stats.mode(memory_y[row])[0])
        # maj_labels = np.array(maj_labels)    
        # maj_labels = []
        # for row in filtered_arr:
        #     maj_labels.append(stats.mode(memory_y[row])[0])
        # maj_labels = np.array(maj_labels)          
        # member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
        # member_inference_class_0 = np.asarray(maj_labels)
        maj_labels = []
        row_indices_to_keep = []
        percentage_mode_value_contributors = []
        Avg_sample_support = []
        Avg_sample_support_counter = 0
        filtered_indices = cos_dist < cos_dist_ip
        print(top_class_1_data.shape,filtered_indices.shape)
        row_indices_to_keep = np.where(np.any(filtered_indices, axis=1))[0]
        rows_to_keep = np.any(filtered_indices, axis=1)
        for i in range(filtered_indices.shape[0]):
            valid_indices = np.where(filtered_indices[i])[0]
            if valid_indices.size > 0:
                Avg_sample_support_counter += 1
                Avg_sample_support.append(valid_indices.size)
                Mode_value_and_count = stats.mode(memory_y[valid_indices])
                Mode_value_percentage = (Mode_value_and_count[1]/valid_indices.size)*100
                if Mode_value_percentage > mode_value:
                    maj_labels.append(Mode_value_and_count[0])
                    percentage_mode_value_contributors.append(Mode_value_percentage)
                else:
                    maj_labels.append(0)    

                # rows_to_keep.append(True)
            else:
                maj_labels.append(0)#Adding a flipped label for class 0 samples as no confident labels found in the memory    
            #     rows_to_keep.append(False)
        print(len(maj_labels))    
        maj_labels = np.array(maj_labels)
        member_inference_class_1 = np.asarray(maj_labels.ravel().tolist())
        print("Average number of sample support for Attack is", stats.tmean(Avg_sample_support), stats.tstd(Avg_sample_support))
        print("Percentage of samples contributed to each Attack sample is",stats.tmean(percentage_mode_value_contributors),stats.tstd(percentage_mode_value_contributors))
        # member_inference_class_1 = np.asarray(maj_labels)
        # top_class_1_indices = top_class_1_indices[rows_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_1_truth = top_class_1_truth[rows_to_keep]#removes the indices whose cosine distance > 0.2 
        # member_inference_class_1 = np.asarray(maj_labels.ravel().tolist())
        end_inference = time.time()
        print(f'\nNumber of class 1 agreements (between model and member inference): {np.sum(member_inference_class_1 == 1)}/{len(member_inference_class_1)} - ({np.sum(member_inference_class_1 == 1)*100./len(member_inference_class_1):.3f})%')
        print(f'Number of class 1 common agreements with ground truth: {np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)}/{np.sum(member_inference_class_1 == 1)} - ({np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)*100./np.sum(member_inference_class_1 == 1):.3f}%)')
        print(f'Time taken for member inference = {end_inference - start_inference}seconds')

        n_agreements_1 = np.sum(member_inference_class_1 == 1)
        curr_truth_agreement_fraction_1 = np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)/np.sum(member_inference_class_1 == 1)
        if math.isnan(curr_truth_agreement_fraction_1):
            curr_truth_agreement_fraction_1 = 0

        # 1-(self)labelled data
        if unseen_task:
            if n_agreements_1 > 0:
                if truth_agreement_fraction_1 is None or math.isnan(truth_agreement_fraction_1):
                    truth_agreement_fraction_1 = 1
                
                selection_count_1 = int(n_agreements_1*truth_agreement_fraction_1)
                selected_1_indices = np.random.choice(top_class_1_indices[member_inference_class_1 == 1], size=selection_count_1, replace=False)
                labeled_indicies = np.hstack((labeled_indicies, selected_1_indices))

                labeled_X = np.vstack((labeled_X, X[selected_1_indices])) if labeled_X is not None else X[selected_1_indices]
                labeled_y = np.hstack((labeled_y, [1]*selection_count_1)) if labeled_y is not None else [1]*selection_count_1
                labeled_y_classname = np.hstack((labeled_y_classname, [attack_y_name]*selection_count_1)) if labeled_y_classname is not None else [attack_y_name]*selection_count_1
                print(f'No. of self-labelled samples (class 1): {selection_count_1}')
                list_self_labelled.setdefault(f"{task_id}", {})["Attack"] = selection_count_1
                owl_self_labelled_count_class_1 += selection_count_1
            else:
                print('No. of self-labelled samples (class 1): 0')
                list_self_labelled.setdefault(f"{task_id}", {})["Attack"] = selection_count_1


    if not unseen_task:
        return [curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1]
    
    print(f'\nTotal no. of self-labeled samples = {selection_count_0 + selection_count_1} (0: {selection_count_0}, 1: {selection_count_1})')

    # Get security analyst to label the remaining high confidence samples
    count_class_0 = int(labels_ratio*est_class_0_samples) - selection_count_0 #- n_agreements_0
    count_class_1 = int(labels_ratio*est_class_1_samples) - selection_count_1 #- n_agreements_1
    
    remaining_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    y_rem = y[remaining_indices]
   
    remaining_0_indices = remaining_indices[np.where(y_rem == 0)[0]] # remaining indices where y == 0 
    remaining_1_indices = remaining_indices[np.where(y_rem == 1)[0]] # remaining indices where y == 1
    print(len(remaining_0_indices), count_class_0)
    
    selected_0_indices = np.random.choice(remaining_0_indices, size=min(len(remaining_0_indices), count_class_0), replace=False)
    selected_1_indices = np.random.choice(remaining_1_indices, size=min(len(remaining_1_indices), count_class_1), replace=False)
    list_analys_labelled.setdefault(f"{task_id}", {})["Benign"] = len(selected_0_indices)
    list_analys_labelled.setdefault(f"{task_id}", {})["Attack"] = len(selected_1_indices)
    temp_X = np.vstack((X[selected_0_indices], X[selected_1_indices]))
    temp_y = np.hstack(([0]*count_class_0, [1]*count_class_1))
    temp_y_classname = np.hstack(([benign_y_name]*count_class_0, [attack_y_name]*count_class_1))
    print(f'No. of security analyst-labelled samples: {temp_X.shape[0]} (0:{len(selected_0_indices)}, 1:{len(selected_1_indices)})')

    owl_analyst_labelled_count_class_0 += len(selected_0_indices)
    owl_analyst_labelled_count_class_1 += len(selected_1_indices)

    labeled_X = np.vstack((labeled_X, temp_X)) if labeled_X is not None else temp_X
    labeled_y = np.hstack((labeled_y, temp_y)) if labeled_y is not None else temp_y
    labeled_y_classname = np.hstack((labeled_y_classname, temp_y_classname)) if labeled_y_classname is not None else temp_y_classname
    labeled_indicies = np.hstack((labeled_indicies, np.hstack((selected_0_indices, selected_1_indices))))
    print(f'Total no. of labelled samples: {labeled_X.shape[0]}')

    unlabeled_indicies = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    X_unlab = X[unlabeled_indicies]
    y_unlab = y[unlabeled_indicies]
    y_classname_unlab = y_classname[unlabeled_indicies]
    print(f'No. of unlabelled samples: {X_unlab.shape}\n')

    labeled_indicies = labeled_indicies.astype(int)
    unlabeled_indicies = unlabeled_indicies.astype(int)

    return labeled_X,labeled_y,labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies

    
def owl_data_labeling_strategy(X, y, y_classname,task_id, unseen_task=True):

    global owl_self_labelled_count_class_0, owl_self_labelled_count_class_1
    global owl_analyst_labelled_count_class_0, owl_analyst_labelled_count_class_1
    global truth_agreement_fraction_0,truth_agreement_fraction_1
    global avg_CI

    print(f'X shape: {X.shape}') # The no of data points for this task
    dummy_target_label = torch.zeros(X.shape[0])
    data_loader = torch.utils.data.DataLoader(dataset(X,dummy_target_label),
                                            batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                            num_workers=0)
        
    predictions,predicted_labels = None,None
    if mlps == 1:
        models = [student_model1]
    elif mlps == 2:
        models = [student_model1,student_model2]
    else:
        models = [student_model1,student_model2,student_supervised]
    
    for data, _ in data_loader:
        class_probs = [] 
        with torch.no_grad():
            for model in models:
                outputs = torch.softmax(model(data.to(device)), dim=1)
                class_probs.append(outputs)
        
            pred = torch.stack(class_probs).mean(dim=0)#[:,1].reshape(target.shape)
            if predictions is None:
                predictions = pred
                predicted_labels = torch.argmax(pred,dim=1)
            else:
                predictions = torch.cat((predictions,pred),dim=0)   
                predicted_labels = torch.cat((predicted_labels,torch.argmax(pred,dim=1)),dim=0) 

    # Step 2: Extracting the high confidence samples for class 0 and class 1 respectively 
    class_0_indices = ((predicted_labels == 0).nonzero(as_tuple=False)[:, 0]).detach().cpu().numpy()
    class_1_indices = ((predicted_labels == 1).nonzero(as_tuple=False)[:, 0]).detach().cpu().numpy()
    
    print(f'Number of predicted 0s: {len(class_0_indices)}')
    print(f'Number of predicted 1s: {len(class_1_indices)}')

    total_samples = X.shape[0]
    est_class_1_samples = int(total_samples*avg_CI)
    est_class_0_samples = total_samples - est_class_1_samples
    print(f'Estimated no. of class 0 samples = {est_class_0_samples}')
    print(f'Estimated no. of class 1 samples = {est_class_1_samples}')
    
    sorted_pred_class_0 = torch.sort(predictions[class_0_indices, 0], dim=0, descending=True)    
    top_class_0_indices = sorted_pred_class_0[1][sorted_pred_class_0[0] > 0.8].detach().cpu()
    if len(top_class_0_indices) > int(labels_ratio*est_class_0_samples):
        top_class_0_indices = top_class_0_indices[:int(labels_ratio*est_class_0_samples)]
    print(f'(few) Highest confidence prediction values (class 0): {sorted_pred_class_0[0][:4]}')

    # print(predictions[class_1_indices, 1])
    sorted_pred_class_1 = torch.sort(predictions[class_1_indices, 1], dim=0, descending=True)
    top_class_1_indices = (sorted_pred_class_1[1][sorted_pred_class_1[0] > 0.8]).detach().cpu()
    if len(top_class_1_indices) > int(labels_ratio*est_class_1_samples):
        top_class_1_indices = top_class_1_indices[:int(labels_ratio*est_class_1_samples)]
    print(f'(few) Highest confidence prediction values (class 1): {sorted_pred_class_1[0][:4]}')
    
    labeled_X,labeled_y,labeled_y_classname = None, None, None
    X_unlab = None
    unlabeled_indicies, labeled_indicies = np.array([]), np.array([])
    member_inference_class_0 = 0 # dummy assignment
    member_inference_class_1 = 1 # dummy assignment
    selection_count_0, selection_count_1 = 0, 0
    n_agreements_0, n_agreements_1 = 0, 0
    curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1 = 0, 0

    # Computing the sample means for each class in memory
    sample_means = None
    associated_label = []

    class_in_memory = np.unique(memory_y_name)
    # print(f'\nclasses in memory: {class_in_memory}')

    for class_idx in class_in_memory:
        indices = np.where(memory_y_name == int(class_idx))[0] 
        if str(int(class_idx)) in minorityclass_ids:
            associated_label.append(1)
        else:
            associated_label.append(0)
        if sample_means is None: 
            sample_means = torch.mean(torch.tensor(memory_X[indices]), dim=0).unsqueeze(0)
        else:
            sample_means = torch.cat((sample_means, torch.mean(torch.tensor(memory_X[indices]), dim=0).unsqueeze(0)), dim=0)
    associated_label = np.array(associated_label)
 
    # Setting unique y_classname labels for the new unseen labelled data (for buffer memory storage purpose)
    if unseen_task:
        attack_y_name = np.max(class_in_memory) + 1
        benign_y_name = np.max(class_in_memory) + 2
        print(f'New classes added to memory: {attack_y_name}, {benign_y_name}')

    if len(top_class_0_indices) >1:
        top_class_0_data = X[class_0_indices[top_class_0_indices]]
        top_class_0_truth = y[class_0_indices[top_class_0_indices]]
        # top_class_0_y_classname = y_classname[class_0_indices[top_class_0_indices]]

        # Member inference of the top samples based on distance from buffer memory samples
        start_inference = time.time()
        cos_dist = cdist(top_class_0_data, memory_X,'cosine')   
        # sorted_indices_temp = np.argsort(cos_dist, axis=1)[::-1]
        # top_k_indices = sorted_indices_temp[:, :1000]
        # # Create boolean mask with True for top k indices, False otherwise
        # mask = np.zeros_like(cos_dist, dtype=bool)
        # mask[np.arange(len(cos_dist))[:, None], top_k_indices] = True
        # filtered_indices = mask
        # # filtered_indices = cos_dist <max_value
        # row_indices_to_keep = np.any(filtered_indices, axis=1)
        # filtered_arr = filtered_indices[row_indices_to_keep]  
        # top_class_0_indices = top_class_0_indices[row_indices_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_0_truth = top_class_0_truth[row_indices_to_keep]#removes the indices whose cosine distance > 0.2        
        # maj_labels = []
        # for row in filtered_arr:
        #     maj_labels.append(stats.mode(memory_y[row])[0])
        # maj_labels = np.array(maj_labels)          
        # member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
        # member_inference_class_0 = np.asarray(maj_labels)
        maj_labels = []
        # row_indices_to_keep = []
        percentage_mode_value_contributors = []
        Avg_sample_support = []
        Avg_sample_support_counter = 0
        filtered_indices = cos_dist < cos_dist_ip
        print(top_class_0_data.shape,filtered_indices.shape)
        # row_indices_to_keep = np.where(np.any(filtered_indices, axis=1))[0]
        # rows_to_keep = np.any(filtered_indices, axis=1)
        for i in range(filtered_indices.shape[0]):
            valid_indices = np.where(filtered_indices[i])[0]
            if valid_indices.size > 0:
                Avg_sample_support_counter += 1
                Avg_sample_support.append(valid_indices.size)
                Mode_value_and_count = stats.mode(memory_y[valid_indices])
                Mode_value_percentage = (Mode_value_and_count[1]/valid_indices.size)*100
                if Mode_value_percentage > mode_value:
                    maj_labels.append(Mode_value_and_count[0])
                    percentage_mode_value_contributors.append(Mode_value_percentage)
                else:
                    maj_labels.append(1) 
                # rows_to_keep.append(True)
            else:
                maj_labels.append(1)#Adding a flipped label for class 0 samples as no confident labels (cost dis <0.2) found in the memory    
                # rowrows_to_keeps_to_keep.append(False)
        print(len(maj_labels))      
        # exit()  
        maj_labels = np.array(maj_labels)
        member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
        # member_inference_class_0 = np.asarray(maj_labels)
        print("Average number of sample support for Attack is", stats.tmean(Avg_sample_support), stats.tstd(Avg_sample_support))
        print("Percentage of samples contributed to each Attack sample is",stats.tmean(percentage_mode_value_contributors),stats.tstd(percentage_mode_value_contributors))
        # top_class_0_indices = top_class_0_indices[rows_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_0_truth = top_class_0_truth[rows_to_keep]#removes the indices whose cosine distance > 0.2           

       

        
        end_inference = time.time()
        print(f'\nNumber of class 0 agreements (between model and member inference): {np.sum(member_inference_class_0 == 0)}/{len(member_inference_class_0)} - ({np.sum(member_inference_class_0 == 0)*100./len(member_inference_class_0):.3f}%)')
        print(f'Number of class 0 common agreements with ground truth: {np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)}/{np.sum(member_inference_class_0 == 0)} - ({np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)*100./np.sum(member_inference_class_0 == 0):.3f}%)')
        print(f'Time taken for member inference = {end_inference - start_inference}seconds')

        n_agreements_0 = np.sum(member_inference_class_0 == 0)
        curr_truth_agreement_fraction_0 = np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)/np.sum(member_inference_class_0 == 0)
        if math.isnan(curr_truth_agreement_fraction_0):
            curr_truth_agreement_fraction_0 = 0

        # 0-(self)labelled data
        if unseen_task:
            if n_agreements_0 > 0:
                if truth_agreement_fraction_0 is None or math.isnan(truth_agreement_fraction_0):
                    truth_agreement_fraction_0 = 1
                selection_count_0 = int(n_agreements_0*truth_agreement_fraction_0)
                selected_0_indices = np.random.choice(top_class_0_indices[member_inference_class_0 == 0], size=selection_count_0, replace=False)
                labeled_indicies = np.hstack((labeled_indicies, selected_0_indices))

                labeled_X = np.vstack((labeled_X, X[selected_0_indices])) if labeled_X is not None else X[selected_0_indices]
                labeled_y = np.hstack((labeled_y, [0]*selection_count_0)) if labeled_y is not None else [0]*selection_count_0
                labeled_y_classname = np.hstack((labeled_y_classname, [benign_y_name]*selection_count_0)) if labeled_y_classname is not None else [benign_y_name]*selection_count_0
                print(f'No. of self-labelled samples (class 0): {selection_count_0}')
                list_self_labelled.setdefault(f"{task_id}", {})["Benign"] = selection_count_0
                owl_self_labelled_count_class_0 += selection_count_0
            else:
                print('No. of self-labelled samples (class 0): 0')
                list_self_labelled.setdefault(f"{task_id}", {})["Benign"] = selection_count_0

    if len(top_class_1_indices) > 1:#!= 0:
    
        top_class_1_data = X[class_1_indices[top_class_1_indices]]
        top_class_1_truth = y[class_1_indices[top_class_1_indices]]
        top_class_1_y_classname = y_classname[class_1_indices[top_class_1_indices]]
 
        # Member inference of the top samples based on distance from sample means
        start_inference = time.time()
        cos_dist = cdist(top_class_1_data, memory_X,'cosine')   
        # sorted_indices_temp = np.argsort(cos_dist, axis=1)[::-1]
        # top_k_indices = sorted_indices_temp[:, :1000]
        # # Create boolean mask with True for top k indices, False otherwise
        # mask = np.zeros_like(cos_dist, dtype=bool)
        # mask[np.arange(len(cos_dist))[:, None], top_k_indices] = True        
        # filtered_indices = mask
        # # filtered_indices = cos_dist <max_value
        # row_indices_to_keep = np.any(filtered_indices, axis=1)
        # filtered_arr = filtered_indices[row_indices_to_keep]  
        # top_class_1_indices = top_class_1_indices[row_indices_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_1_truth = top_class_1_truth[row_indices_to_keep]#removes the indices whose cosine distance > 0.2        
        # maj_labels = []
        # for row in filtered_arr:
        #     maj_labels.append(stats.mode(memory_y[row])[0])
        # maj_labels = np.array(maj_labels)    
        # maj_labels = []
        # for row in filtered_arr:
        #     maj_labels.append(stats.mode(memory_y[row])[0])
        # maj_labels = np.array(maj_labels)          
        # member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
        # member_inference_class_0 = np.asarray(maj_labels)
        maj_labels = []
        row_indices_to_keep = []
        percentage_mode_value_contributors = []
        Avg_sample_support = []
        Avg_sample_support_counter = 0
        filtered_indices = cos_dist < cos_dist_ip
        print(top_class_1_data.shape,filtered_indices.shape)
        row_indices_to_keep = np.where(np.any(filtered_indices, axis=1))[0]
        rows_to_keep = np.any(filtered_indices, axis=1)
        for i in range(filtered_indices.shape[0]):
            valid_indices = np.where(filtered_indices[i])[0]
            if valid_indices.size > 0:
                Avg_sample_support_counter += 1
                Avg_sample_support.append(valid_indices.size)
                Mode_value_and_count = stats.mode(memory_y[valid_indices])
                Mode_value_percentage = (Mode_value_and_count[1]/valid_indices.size)*100
                if Mode_value_percentage > mode_value:
                    maj_labels.append(Mode_value_and_count[0])
                    percentage_mode_value_contributors.append(Mode_value_percentage)
                else:
                    maj_labels.append(0)    

                # rows_to_keep.append(True)
            else:
                maj_labels.append(0)#Adding a flipped label for class 0 samples as no confident labels found in the memory    
            #     rows_to_keep.append(False)
        print(len(maj_labels))    
        maj_labels = np.array(maj_labels)
        member_inference_class_1 = np.asarray(maj_labels.ravel().tolist())
        print("Average number of sample support for Attack is", stats.tmean(Avg_sample_support), stats.tstd(Avg_sample_support))
        print("Percentage of samples contributed to each Attack sample is",stats.tmean(percentage_mode_value_contributors),stats.tstd(percentage_mode_value_contributors))
        # member_inference_class_1 = np.asarray(maj_labels)
        # top_class_1_indices = top_class_1_indices[rows_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_1_truth = top_class_1_truth[rows_to_keep]#removes the indices whose cosine distance > 0.2 
        # member_inference_class_1 = np.asarray(maj_labels.ravel().tolist())
        end_inference = time.time()
        print(f'\nNumber of class 1 agreements (between model and member inference): {np.sum(member_inference_class_1 == 1)}/{len(member_inference_class_1)} - ({np.sum(member_inference_class_1 == 1)*100./len(member_inference_class_1):.3f})%')
        print(f'Number of class 1 common agreements with ground truth: {np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)}/{np.sum(member_inference_class_1 == 1)} - ({np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)*100./np.sum(member_inference_class_1 == 1):.3f}%)')
        print(f'Time taken for member inference = {end_inference - start_inference}seconds')

        n_agreements_1 = np.sum(member_inference_class_1 == 1)
        curr_truth_agreement_fraction_1 = np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)/np.sum(member_inference_class_1 == 1)
        if math.isnan(curr_truth_agreement_fraction_1):
            curr_truth_agreement_fraction_1 = 0

        # 1-(self)labelled data
        if unseen_task:
            if n_agreements_1 > 0:
                if truth_agreement_fraction_1 is None or math.isnan(truth_agreement_fraction_1):
                    truth_agreement_fraction_1 = 1
                
                selection_count_1 = int(n_agreements_1*truth_agreement_fraction_1)
                selected_1_indices = np.random.choice(top_class_1_indices[member_inference_class_1 == 1], size=selection_count_1, replace=False)
                labeled_indicies = np.hstack((labeled_indicies, selected_1_indices))

                labeled_X = np.vstack((labeled_X, X[selected_1_indices])) if labeled_X is not None else X[selected_1_indices]
                labeled_y = np.hstack((labeled_y, [1]*selection_count_1)) if labeled_y is not None else [1]*selection_count_1
                labeled_y_classname = np.hstack((labeled_y_classname, [attack_y_name]*selection_count_1)) if labeled_y_classname is not None else [attack_y_name]*selection_count_1
                
                print(f'No. of self-labelled samples (class 1): {selection_count_1}')
                list_self_labelled.setdefault(f"{task_id}", {})["Attack"] = selection_count_1
                owl_self_labelled_count_class_1 += selection_count_1
            else:
                print('No. of self-labelled samples (class 1): 0')
                list_self_labelled.setdefault(f"{task_id}", {})["Attack"] = selection_count_1

    if not unseen_task:
        return [curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1]
    
    print(f'\nTotal no. of self-labeled samples = {selection_count_0 + selection_count_1} (0: {selection_count_0}, 1: {selection_count_1})')

    # Get security analyst to label the remaining high confidence samples
    count_class_0 = int(labels_ratio*est_class_0_samples) - selection_count_0 #- n_agreements_0
    count_class_1 = int(labels_ratio*est_class_1_samples) - selection_count_1 #- n_agreements_1
    
    remaining_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    y_rem = y[remaining_indices]
   
    remaining_0_indices = remaining_indices[np.where(y_rem == 0)[0]] # remaining indices where y == 0 
    remaining_1_indices = remaining_indices[np.where(y_rem == 1)[0]] # remaining indices where y == 1
    print(len(remaining_0_indices), count_class_0)
    selected_0_indices = np.random.choice(remaining_0_indices, size=min(len(remaining_0_indices), count_class_0), replace=False)
    selected_1_indices = np.random.choice(remaining_1_indices, size=min(len(remaining_1_indices), count_class_1), replace=False)

    temp_X = np.vstack((X[selected_0_indices], X[selected_1_indices]))
    temp_y = np.hstack(([0]*count_class_0, [1]*count_class_1))
    temp_y_classname = np.hstack(([benign_y_name]*count_class_0, [attack_y_name]*count_class_1))
    print(f'No. of security analyst-labelled samples: {temp_X.shape[0]} (0:{len(selected_0_indices)}, 1:{len(selected_1_indices)})')
    list_analys_labelled.setdefault(f"{task_id}", {})["Benign"] = len(selected_0_indices)
    list_analys_labelled.setdefault(f"{task_id}", {})["Attack"] = len(selected_1_indices)
    owl_analyst_labelled_count_class_0 += len(selected_0_indices)
    owl_analyst_labelled_count_class_1 += len(selected_1_indices)

    labeled_X = np.vstack((labeled_X, temp_X)) if labeled_X is not None else temp_X
    labeled_y = np.hstack((labeled_y, temp_y)) if labeled_y is not None else temp_y
    labeled_y_classname = np.hstack((labeled_y_classname, temp_y_classname)) if labeled_y_classname is not None else temp_y_classname
    labeled_indicies = np.hstack((labeled_indicies, np.hstack((selected_0_indices, selected_1_indices))))
    print(f'Total no. of labelled samples: {labeled_X.shape[0]}')

    unlabeled_indicies = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    X_unlab = X[unlabeled_indicies]
    y_unlab = y[unlabeled_indicies]
    y_classname_unlab = y_classname[unlabeled_indicies]
    print(f'No. of unlabelled samples: {X_unlab.shape}\n')

    labeled_indicies = labeled_indicies.astype(int)
    unlabeled_indicies = unlabeled_indicies.astype(int)

    return labeled_X,labeled_y,labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies




def owl_data_labeling_strategy_const_labels2(X, y, y_classname,task_id, analyst_labels = 100,unseen_task=True):
    global owl_self_labelled_count_class_0, owl_self_labelled_count_class_1
    global owl_analyst_labelled_count_class_0, owl_analyst_labelled_count_class_1
    global truth_agreement_fraction_0,truth_agreement_fraction_1
    global avg_CI

    print(f'X shape: {X.shape}')
    dummy_target_label = torch.zeros(X.shape[0])
    data_loader = torch.utils.data.DataLoader(dataset(X,dummy_target_label),
                                            batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                            num_workers=0)
        
    predictions,predicted_labels = None,None

    if mlps == 1:
        models = [student_model1]
    elif mlps == 2:
        models = [student_model1,student_model2]
    else:
        models = [student_model1,student_model2,student_supervised]
    
    for data, _ in data_loader:
        class_probs = [] 
        with torch.no_grad():
            for model in models:
                outputs = torch.softmax(model(data.to(device)), dim=1)
                class_probs.append(outputs)
        
            pred = torch.stack(class_probs).mean(dim=0)#[:,1].reshape(target.shape)
    #         if predictions is None:
    #             predictions = pred
    #             predicted_labels = torch.argmax(pred,dim=1)
    #         else:
    #             predictions = torch.cat((predictions,pred),dim=0)   
    #             predicted_labels = torch.cat((predicted_labels,torch.argmax(pred,dim=1)),dim=0) 

    # # Step 2: Extracting the high confidence samples for class 0 and class 1 respectively 
    # class_0_indices = ((predicted_labels == 0).nonzero(as_tuple=False)[:, 0]).detach().cpu().numpy()
    # class_1_indices = ((predicted_labels == 1).nonzero(as_tuple=False)[:, 0]).detach().cpu().numpy()
    
    # print(f'Number of predicted 0s: {len(class_0_indices)}')
    # print(f'Number of predicted 1s: {len(class_1_indices)}')

    # total_samples = X.shape[0]    
    # est_class_1_samples = int(total_samples*avg_CI)    
    # est_class_0_samples = total_samples - est_class_1_samples
    # print(f'Estimated no. of class 0 samples = {est_class_0_samples}')
    # print(f'Estimated no. of class 1 samples = {est_class_1_samples}')
    
    # sorted_pred_class_0 = torch.sort(predictions[class_0_indices, 0], dim=0, descending=True)    
    # top_class_0_indices = sorted_pred_class_0[1][sorted_pred_class_0[0] > 0.8].detach().cpu()
    # if len(top_class_0_indices) > int(labels_ratio*est_class_0_samples):
    #     top_class_0_indices = top_class_0_indices[:int(labels_ratio*est_class_0_samples)]
    # print(f'(few) Highest confidence prediction values (class 0): {sorted_pred_class_0[0][:4]}')

    # # print(predictions[class_1_indices, 1])
    # sorted_pred_class_1 = torch.sort(predictions[class_1_indices, 1], dim=0, descending=True)
    # top_class_1_indices = (sorted_pred_class_1[1][sorted_pred_class_1[0] > 0.5]).detach().cpu()
    # if len(top_class_1_indices) > int(labels_ratio*est_class_1_samples):
    #     top_class_1_indices = top_class_1_indices[:int(labels_ratio*est_class_1_samples)]
    # print(f'(few) Highest confidence prediction values (class 1): {sorted_pred_class_1[0][:4]}')
    
    labeled_X,labeled_y,labeled_y_classname = None, None, None
    X_unlab = None
    unlabeled_indicies, labeled_indicies = np.array([]), np.array([])
    member_inference_class_0 = 0 # dummy assignment
    member_inference_class_1 = 1 # dummy assignment
    selection_count_0, selection_count_1 = 0, 0
    n_agreements_0, n_agreements_1 = 0, 0
    curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1 = 0, 0

    # Computing the sample means for each class in memory
    sample_means = None
    associated_label = []

    class_in_memory = np.unique(memory_y_name)
    # print(f'\nclasses in memory: {class_in_memory}')

    for class_idx in class_in_memory:
        indices = np.where(memory_y_name == int(class_idx))[0]
        if str(int(class_idx)) in minorityclass_ids:
            associated_label.append(1)
        else:
            associated_label.append(0)

        # if sample_means is None:
        #     sample_means = torch.mean(torch.tensor(memory_X[indices]), dim=0).unsqueeze(0)
        # else:
        #     sample_means = torch.cat((sample_means, torch.mean(torch.tensor(memory_X[indices]), dim=0).unsqueeze(0)), dim=0)
    associated_label = np.array(associated_label)

    # Setting unique y_classname labels for the new unseen labelled data (for buffer memory storage purpose)
    if unseen_task:
        attack_y_name = np.max(class_in_memory) + 1
        benign_y_name = np.max(class_in_memory) + 2
        print(f'New classes added to memory: {attack_y_name}, {benign_y_name}')

    # if len(top_class_0_indices) != 0:

    #     top_class_0_data = X[class_0_indices[top_class_0_indices]]
    #     top_class_0_truth = y[class_0_indices[top_class_0_indices]]
    #     top_class_0_y_classname = y_classname[class_0_indices[top_class_0_indices]]

    #     # Member inference of the top samples based on distance from buffer memory samples
    #     start_inference = time.time()
    #     cos_dist = cdist(model.forward_encoder((torch.from_numpy((top_class_0_data))).to(device)).detach().cpu().numpy(), model.forward_encoder(torch.from_numpy(memory_X[np.where(memory_y ==0)]).to(device)).detach().cpu().numpy(),'cosine')   
    #     # cos_dist = cdist(model.forward_encoder(maybe_unsqueeze(torch.from_numpy((top_class_0_data))).to(device)).detach().cpu().numpy(), model.forward_encoder(torch.from_numpy(memory_X[np.where(memory_y ==0)]).to(device)).detach().cpu().numpy(),'cosine')   
    #     # cos_dist = 1 - F.cosine_similarity(model.forward_encoder(torch.from_numpy(top_class_0_data).to(device)), model.forward_encoder(torch.from_numpy(memory_X).to(device)))
    #     # sorted_indices_temp = np.argsort(cos_dist, axis=1)[::-1]
    #     # top_k_indices = sorted_indices_temp[:, :1000]
    #     # # Create boolean mask with True for top k indices, False otherwise
    #     # mask = np.zeros_like(cos_dist, dtype=bool)
    #     # mask[np.arange(len(cos_dist))[:, None], top_k_indices] = True
    #     # filtered_indices = mask
    #     # # filtered_indices = cos_dist <max_value
    #     # row_indices_to_keep = np.any(filtered_indices, axis=1)
    #     # filtered_arr = filtered_indices[row_indices_to_keep]  
    #     # top_class_0_indices = top_class_0_indices[row_indices_to_keep]#removes the indices whose cosine distance > 0.2 
    #     # top_class_0_truth = top_class_0_truth[row_indices_to_keep]#removes the indices whose cosine distance > 0.2        
    #     # maj_labels = []
    #     # for row in filtered_arr:
    #     #     maj_labels.append(stats.mode(memory_y[row])[0])
    #     # maj_labels = np.array(maj_labels)          
    #     # member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
    #     # member_inference_class_0 = np.asarray(maj_labels)
    #     maj_labels = []
    #     row_indices_to_keep = []
    #     percentage_mode_value_contributors = []
    #     Avg_sample_support = []
    #     Avg_sample_support_counter = 0
    #     filtered_indices = cos_dist < (cos_dist_ip*math.exp(-temperature * labels_ratio)*1e-4)
    #     print(top_class_0_data.shape,filtered_indices.shape)
    #     row_indices_to_keep = np.where(np.any(filtered_indices, axis=1))[0]
    #     rows_to_keep = np.any(filtered_indices, axis=1)
    #     for i in range(filtered_indices.shape[0]):
    #         valid_indices = np.where(filtered_indices[i])[0]
    #         if valid_indices.size > 0:
    #             Avg_sample_support_counter += 1
    #             Avg_sample_support.append(valid_indices.size)
    #             Mode_value_and_count = stats.mode(memory_y[valid_indices])
    #             Mode_value_percentage = (Mode_value_and_count[1]/valid_indices.size)*100
    #             if Mode_value_percentage > mode_value:
    #                 maj_labels.append(Mode_value_and_count[0])
    #                 percentage_mode_value_contributors.append(Mode_value_percentage)
    #             else:
    #                 maj_labels.append(1)
    #             # rows_to_keep.append(True)
    #         else:
    #             maj_labels.append(1)#Adding a flipped label for class 0 samples as no confident labels (cost dis <0.2) found in the memory    
    #             # rowrows_to_keeps_to_keep.append(False)
    #     print(len(maj_labels))      
    #     # exit()  
    #     maj_labels = np.array(maj_labels)
    #     member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
    #     # member_inference_class_0 = np.asarray(maj_labels)
    #     print("Average number of sample support for Attack is", stats.tmean(Avg_sample_support), stats.tstd(Avg_sample_support))
    #     print("Percentage of samples contributed to each Attack sample is",stats.tmean(percentage_mode_value_contributors),stats.tstd(percentage_mode_value_contributors))
    #     # top_class_0_indices = top_class_0_indices[rows_to_keep]#removes the indices whose cosine distance > 0.2 
    #     # top_class_0_truth = top_class_0_truth[rows_to_keep]#removes the indices whose cosine distance > 0.2           

       

        
    #     end_inference = time.time()
    #     print(f'\nNumber of class 0 agreements (between model and member inference): {np.sum(member_inference_class_0 == 0)}/{len(member_inference_class_0)} - ({np.sum(member_inference_class_0 == 0)*100./len(member_inference_class_0):.3f}%)')
    #     print(f'Number of class 0 common agreements with ground truth: {np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)}/{np.sum(member_inference_class_0 == 0)} - ({np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)*100./np.sum(member_inference_class_0 == 0):.3f}%)')
    #     print(f'Time taken for member inference = {end_inference - start_inference}seconds')

    #     n_agreements_0 = np.sum(member_inference_class_0 == 0)
    #     curr_truth_agreement_fraction_0 = np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)/np.sum(member_inference_class_0 == 0)
    #     if math.isnan(curr_truth_agreement_fraction_0):
    #         curr_truth_agreement_fraction_0 = 0

    #     # 0-(self)labelled data
    #     if unseen_task:
    #         if n_agreements_0 > 0:
    #             if truth_agreement_fraction_0 is None or math.isnan(truth_agreement_fraction_0):
    #                 truth_agreement_fraction_0 = 1
    #             selection_count_0 = int(n_agreements_0*truth_agreement_fraction_0)
    #             selected_0_indices = np.random.choice(top_class_0_indices[member_inference_class_0 == 0], size=selection_count_0, replace=False)
    #             labeled_indicies = np.hstack((labeled_indicies, selected_0_indices))

    #             labeled_X = np.vstack((labeled_X, X[selected_0_indices])) if labeled_X is not None else X[selected_0_indices]
    #             labeled_y = np.hstack((labeled_y, [0]*selection_count_0)) if labeled_y is not None else [0]*selection_count_0
    #             labeled_y_classname = np.hstack((labeled_y_classname, [benign_y_name]*selection_count_0)) if labeled_y_classname is not None else [benign_y_name]*selection_count_0
    #             print(f'No. of self-labelled samples (class 0): {selection_count_0}')

    #             owl_self_labelled_count_class_0 += selection_count_0
    #         else:
    #             print('No. of self-labelled samples (class 0): 0')

    # if len(top_class_1_indices) > 1:#!= 0:
    
    #     top_class_1_data = X[class_1_indices[top_class_1_indices]]
    #     top_class_1_truth = y[class_1_indices[top_class_1_indices]]
    #     top_class_1_y_classname = y_classname[class_1_indices[top_class_1_indices]]
 
    #     # Member inference of the top samples based on distance from sample means
    #     start_inference = time.time()
    #     # cos_dist = cdist(top_class_1_data, memory_X,'cosine')   
    #     cos_dist = cdist(model.forward_encoder(torch.from_numpy(top_class_1_data).to(device)).detach().cpu().numpy(), model.forward_encoder(torch.from_numpy(memory_X[np.where(memory_y ==1)]).to(device)).detach().cpu().numpy(),'cosine')
    #     # sorted_indices_temp = np.argsort(cos_dist, axis=1)[::-1]
    #     # top_k_indices = sorted_indices_temp[:, :1000]
    #     # # Create boolean mask with True for top k indices, False otherwise
    #     # mask = np.zeros_like(cos_dist, dtype=bool)
    #     # mask[np.arange(len(cos_dist))[:, None], top_k_indices] = True        
    #     # filtered_indices = mask
    #     # # filtered_indices = cos_dist <max_value
    #     # row_indices_to_keep = np.any(filtered_indices, axis=1)
    #     # filtered_arr = filtered_indices[row_indices_to_keep]  
    #     # top_class_1_indices = top_class_1_indices[row_indices_to_keep]#removes the indices whose cosine distance > 0.2 
    #     # top_class_1_truth = top_class_1_truth[row_indices_to_keep]#removes the indices whose cosine distance > 0.2        
    #     # maj_labels = []
    #     # for row in filtered_arr:
    #     #     maj_labels.append(stats.mode(memory_y[row])[0])
    #     # maj_labels = np.array(maj_labels)    
    #     # maj_labels = []
    #     # for row in filtered_arr:
    #     #     maj_labels.append(stats.mode(memory_y[row])[0])
    #     # maj_labels = np.array(maj_labels)          
    #     # member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
    #     # member_inference_class_0 = np.asarray(maj_labels)
    #     maj_labels = []
    #     row_indices_to_keep = []
    #     percentage_mode_value_contributors = []
    #     Avg_sample_support = []
    #     Avg_sample_support_counter = 0
    #     filtered_indices = cos_dist < (cos_dist_ip*math.exp(-temperature * labels_ratio)*1e-4)
    #     print(top_class_1_data.shape,filtered_indices.shape)
    #     row_indices_to_keep = np.where(np.any(filtered_indices, axis=1))[0]
    #     rows_to_keep = np.any(filtered_indices, axis=1)
    #     for i in range(filtered_indices.shape[0]):
    #         valid_indices = np.where(filtered_indices[i])[0]
    #         if valid_indices.size > 0:
    #             Avg_sample_support_counter += 1
    #             Avg_sample_support.append(valid_indices.size)
    #             Mode_value_and_count = stats.mode(memory_y[valid_indices])
    #             Mode_value_percentage = (Mode_value_and_count[1]/valid_indices.size)*100
    #             if Mode_value_percentage > mode_value:
    #                 maj_labels.append(Mode_value_and_count[0])
    #                 percentage_mode_value_contributors.append(Mode_value_percentage)
    #             else:
    #                 maj_labels.append(0)    

    #             # rows_to_keep.append(True)
    #         else:
    #             maj_labels.append(0)#Adding a flipped label for class 0 samples as no confident labels found in the memory    
    #         #     rows_to_keep.append(False)
    #     print(len(maj_labels))    
    #     maj_labels = np.array(maj_labels)
    #     member_inference_class_1 = np.asarray(maj_labels.ravel().tolist())
    #     print("Average number of sample support for Attack is", stats.tmean(Avg_sample_support), stats.tstd(Avg_sample_support))
    #     print("Percentage of samples contributed to each Attack sample is",stats.tmean(percentage_mode_value_contributors),stats.tstd(percentage_mode_value_contributors))
    #     # member_inference_class_1 = np.asarray(maj_labels)
    #     # top_class_1_indices = top_class_1_indices[rows_to_keep]#removes the indices whose cosine distance > 0.2 
    #     # top_class_1_truth = top_class_1_truth[rows_to_keep]#removes the indices whose cosine distance > 0.2 
    #     # member_inference_class_1 = np.asarray(maj_labels.ravel().tolist())
    #     end_inference = time.time()
    #     print(f'\nNumber of class 1 agreements (between model and member inference): {np.sum(member_inference_class_1 == 1)}/{len(member_inference_class_1)} - ({np.sum(member_inference_class_1 == 1)*100./len(member_inference_class_1):.3f})%')
    #     print(f'Number of class 1 common agreements with ground truth: {np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)}/{np.sum(member_inference_class_1 == 1)} - ({np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)*100./np.sum(member_inference_class_1 == 1):.3f}%)')
    #     print(f'Time taken for member inference = {end_inference - start_inference}seconds')

    #     n_agreements_1 = np.sum(member_inference_class_1 == 1)
    #     curr_truth_agreement_fraction_1 = np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)/np.sum(member_inference_class_1 == 1)
    #     if math.isnan(curr_truth_agreement_fraction_1):
    #         curr_truth_agreement_fraction_1 = 0

    #     # 1-(self)labelled data
    #     if unseen_task:
    #         if n_agreements_1 > 0:
    #             if truth_agreement_fraction_1 is None or math.isnan(truth_agreement_fraction_1):
    #                 truth_agreement_fraction_1 = 1
                
    #             selection_count_1 = int(n_agreements_1*truth_agreement_fraction_1)
    #             selected_1_indices = np.random.choice(top_class_1_indices[member_inference_class_1 == 1], size=selection_count_1, replace=False)
    #             labeled_indicies = np.hstack((labeled_indicies, selected_1_indices))

    #             labeled_X = np.vstack((labeled_X, X[selected_1_indices])) if labeled_X is not None else X[selected_1_indices]
    #             labeled_y = np.hstack((labeled_y, [1]*selection_count_1)) if labeled_y is not None else [1]*selection_count_1
    #             labeled_y_classname = np.hstack((labeled_y_classname, [attack_y_name]*selection_count_1)) if labeled_y_classname is not None else [attack_y_name]*selection_count_1
                
    #             print(f'No. of self-labelled samples (class 1): {selection_count_1}')
    #             owl_self_labelled_count_class_1 += selection_count_1
    #         else:
    #             print('No. of self-labelled samples (class 1): 0')

    if not unseen_task:
        return [curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1]
    
    # print(f'\nTotal no. of self-labeled samples = {selection_count_0 + selection_count_1} (0: {selection_count_0}, 1: {selection_count_1})')

    # # Get security analyst to label the remaining high confidence samples
   
    
    # remaining_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    remaining_indices = np.arange(X.shape[0])
    y_rem = y[remaining_indices]

    

    # feat_unlab = model.forward_encoder(torch.from_numpy(X[remaining_indices]).to(device))
    # feat_unlab_normalized = F.normalize(feat_unlab, p=2, dim=1)
    # mem_samples_normalized = F.normalize(model.forward_encoder(torch.from_numpy(memory_X).to(device)),p=2,dim=1)
    # cos_dist_feat_unlab = cdist(feat_unlab_normalized.detach().cpu().numpy(), mem_samples_normalized.detach().cpu().numpy(),'cosine')
    # avg_cos_dis = cos_dist_feat_unlab.mean(axis=1)
    # sorted_indices = np.argsort(avg_cos_dis)[::-1]
    


    # top_selected_indices = sorted_indices[:min(len(remaining_indices),analyst_labels)]
    # true_indices = y[top_selected_indices]
    # zero_indices = np.where(true_indices == 0)[0]
    # zero_indices = sorted_indices[zero_indices]
    
    # one_indices = np.where(true_indices == 1)[0]
    # one_indices = sorted_indices[one_indices]
   

    # remaining_indices = np.setdiff1d(remaining_indices, (zero_indices.tolist()+one_indices.tolist()))
    

    
    
    # selected_0_indices = zero_indices
    # selected_1_indices = one_indices
    # count_class_0 = len(selected_0_indices)
    # count_class_1 = len(selected_1_indices)
    # temp_X = np.vstack((X[selected_0_indices], X[selected_1_indices]))
    # temp_y = np.hstack(([0]*count_class_0, [1]*count_class_1))
    # temp_y_classname = np.hstack(([benign_y_name]*count_class_0, [attack_y_name]*count_class_1))
    # print(f'No. of security analyst-labelled samples: {temp_X.shape[0]} (0:{len(selected_0_indices)}, 1:{len(selected_1_indices)})')

    # owl_analyst_labelled_count_class_0 += len(selected_0_indices)
    # owl_analyst_labelled_count_class_1 += len(selected_1_indices)

    # labeled_X = np.vstack((labeled_X, temp_X)) if labeled_X is not None else temp_X
    # labeled_y = np.hstack((labeled_y, temp_y)) if labeled_y is not None else temp_y
    # labeled_y_classname = np.hstack((labeled_y_classname, temp_y_classname)) if labeled_y_classname is not None else temp_y_classname
    # labeled_indicies = np.hstack((labeled_indicies, np.hstack((selected_0_indices, selected_1_indices))))
    # print(f'Total no. of labelled samples: {labeled_X.shape[0]}')

    # unlabeled_indicies = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    # X_unlab = X[unlabeled_indicies]
    # y_unlab = y[unlabeled_indicies]
    # y_classname_unlab = y_classname[unlabeled_indicies]
    # print(f'No. of unlabelled samples: {X_unlab.shape}\n')

    # labeled_indicies = labeled_indicies.astype(int)
    # unlabeled_indicies = unlabeled_indicies.astype(int)

    # return labeled_X,labeled_y,labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies

    # remaining_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    

    feat_unlab = model.forward_encoder(torch.from_numpy(X[remaining_indices]).to(device))
    feat_unlab_normalized = F.normalize(feat_unlab, p=2, dim=1)
    mem_samples_normalized = F.normalize(model.forward_encoder(torch.from_numpy(memory_X[np.where(memory_y==0)[0]]).to(device)),p=2,dim=1)
    cos_dist_feat_unlab = cdist(feat_unlab_normalized.detach().cpu().numpy(), mem_samples_normalized.detach().cpu().numpy(),'cosine')
    avg_cos_dis = cos_dist_feat_unlab.mean(axis=1)
    sorted_indices = np.argsort(avg_cos_dis)#[::-1]
    # y_sorted_indices = remaining_indices[sorted_indices]
    # true_indices = y[sorted_indices]

    
    if ds =="api_graph":
        zero_ratio = 0.1
    else:
        zero_ratio = 0.5
    top_selected_indices_0 = sorted_indices[:min(len(remaining_indices),int(analyst_labels*zero_ratio))]
    # true_indices = y[top_selected_indices]
    # zero_indices = np.where(true_indices == 0)[0]
    # zero_indices = sorted_indices[zero_indices]
    zero_indices = top_selected_indices_0
    print(y[zero_indices])
    # print(y[sorted_indices[zero_indices]])


    mem_samples_normalized = F.normalize(model.forward_encoder(torch.from_numpy(memory_X[np.where(memory_y==1)[0]]).to(device)),p=2,dim=1)
    cos_dist_feat_unlab = cdist(feat_unlab_normalized.detach().cpu().numpy(), mem_samples_normalized.detach().cpu().numpy(),'cosine')
    avg_cos_dis = cos_dist_feat_unlab.mean(axis=1)
    sorted_indices = np.argsort(avg_cos_dis)#[::-1]
    
    # top_selected_indices_1 = sorted_indices[:min(len(remaining_indices),int(analyst_labels*0.5))]

    non_matching_indices = [idx for idx in sorted_indices if idx not in zero_indices]

    top_selected_indices_1 = non_matching_indices[:min(len(non_matching_indices), int(analyst_labels * (1-zero_ratio)))]

    # one_indices = np.where(true_indices == 1)[0]
    # one_indices = sorted_indices[one_indices]
    one_indices = top_selected_indices_1
    # print(y[sorted_indices[one_indices]])
    # print(y[sorted_indices[one_indices]])
    print(y[one_indices])

    # remaining_indices = np.setdiff1d(remaining_indices, (zero_indices.tolist()+one_indices.tolist()))
    remaining_indices = np.setdiff1d(remaining_indices, (zero_indices.tolist()+one_indices))
   
    
    selected_0_indices = zero_indices
    selected_1_indices = one_indices
    total_labels = y[selected_0_indices].tolist()+y[selected_1_indices].tolist()
    # count_class_0 = len(selected_0_indices)
    # count_class_1 = len(selected_1_indices)
    count_class_0 = total_labels.count(0)
    count_class_1 = total_labels.count(1)
    temp_X = np.vstack((X[selected_0_indices], X[selected_1_indices]))
    # temp_y = np.hstack(([0]*count_class_0, [1]*count_class_1))
    temp_y = np.hstack((y[selected_0_indices], y[selected_1_indices]))
    temp_y_classname = np.hstack(([benign_y_name]*count_class_0, [attack_y_name]*count_class_1))
    print(f'No. of security analyst-labelled samples: {temp_X.shape[0]} (0:{count_class_0}, 1:{count_class_1})')
    # exit()
    # owl_analyst_labelled_count_class_0 += len(selected_0_indices)
    # owl_analyst_labelled_count_class_1 += len(selected_1_indices)

    owl_analyst_labelled_count_class_0 += count_class_0
    owl_analyst_labelled_count_class_1 += count_class_1

    labeled_X = np.vstack((labeled_X, temp_X)) if labeled_X is not None else temp_X
    labeled_y = np.hstack((labeled_y, temp_y)) if labeled_y is not None else temp_y
    labeled_y_classname = np.hstack((labeled_y_classname, temp_y_classname)) if labeled_y_classname is not None else temp_y_classname
    # labeled_indicies = np.hstack((labeled_indicies, np.hstack((selected_0_indices, selected_1_indices))))
    labeled_indicies = np.hstack((selected_0_indices, selected_1_indices))
    print(f'Total no. of labelled samples: {labeled_X.shape[0]}')

    unlabeled_indicies = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    X_unlab = X[unlabeled_indicies]
    y_unlab = y[unlabeled_indicies]
    y_classname_unlab = y_classname[unlabeled_indicies]
    print(f'No. of unlabelled samples: {X_unlab.shape}\n')

    labeled_indicies = labeled_indicies.astype(int)
    unlabeled_indicies = unlabeled_indicies.astype(int)

    return labeled_X,labeled_y,labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies



def owl_data_labeling_strategy_const_labels(X, y, y_classname,task_id, analyst_labels = 100,unseen_task=True):
    global owl_self_labelled_count_class_0, owl_self_labelled_count_class_1
    global owl_analyst_labelled_count_class_0, owl_analyst_labelled_count_class_1
    global truth_agreement_fraction_0,truth_agreement_fraction_1
    global avg_CI

    print(f'X shape: {X.shape}')
    dummy_target_label = torch.zeros(X.shape[0])
    data_loader = torch.utils.data.DataLoader(dataset(X,dummy_target_label),
                                            batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                            num_workers=0)
        
    predictions,predicted_labels = None,None

    if mlps == 1:
        models = [student_model1]
    elif mlps == 2:
        models = [student_model1,student_model2]
    else:
        models = [student_model1,student_model2,student_supervised]
    
    for data, _ in data_loader:
        class_probs = [] 
        with torch.no_grad():
            for model in models:
                outputs = torch.softmax(model(data.to(device)), dim=1)
                class_probs.append(outputs)
        
            pred = torch.stack(class_probs).mean(dim=0)#[:,1].reshape(target.shape)
            if predictions is None:
                predictions = pred
                predicted_labels = torch.argmax(pred,dim=1)
            else:
                predictions = torch.cat((predictions,pred),dim=0)   
                predicted_labels = torch.cat((predicted_labels,torch.argmax(pred,dim=1)),dim=0) 

    # Step 2: Extracting the high confidence samples for class 0 and class 1 respectively 
    class_0_indices = ((predicted_labels == 0).nonzero(as_tuple=False)[:, 0]).detach().cpu().numpy()
    class_1_indices = ((predicted_labels == 1).nonzero(as_tuple=False)[:, 0]).detach().cpu().numpy()
    
    print(f'Number of predicted 0s: {len(class_0_indices)}')
    print(f'Number of predicted 1s: {len(class_1_indices)}')

    total_samples = X.shape[0]    
    est_class_1_samples = int(total_samples*avg_CI)    
    est_class_0_samples = total_samples - est_class_1_samples
    print(f'Estimated no. of class 0 samples = {est_class_0_samples}')
    print(f'Estimated no. of class 1 samples = {est_class_1_samples}')
    
    sorted_pred_class_0 = torch.sort(predictions[class_0_indices, 0], dim=0, descending=True)    
    top_class_0_indices = sorted_pred_class_0[1][sorted_pred_class_0[0] > 0.8].detach().cpu()
    if len(top_class_0_indices) > int(labels_ratio*est_class_0_samples):
        top_class_0_indices = top_class_0_indices[:int(labels_ratio*est_class_0_samples)]
    print(f'(few) Highest confidence prediction values (class 0): {sorted_pred_class_0[0][:4]}')

    # print(predictions[class_1_indices, 1])
    sorted_pred_class_1 = torch.sort(predictions[class_1_indices, 1], dim=0, descending=True)
    top_class_1_indices = (sorted_pred_class_1[1][sorted_pred_class_1[0] > 0.5]).detach().cpu()
    if len(top_class_1_indices) > int(labels_ratio*est_class_1_samples):
        top_class_1_indices = top_class_1_indices[:int(labels_ratio*est_class_1_samples)]
    print(f'(few) Highest confidence prediction values (class 1): {sorted_pred_class_1[0][:4]}')
    
    labeled_X,labeled_y,labeled_y_classname = None, None, None
    X_unlab = None
    unlabeled_indicies, labeled_indicies = np.array([]), np.array([])
    member_inference_class_0 = 0 # dummy assignment
    member_inference_class_1 = 1 # dummy assignment
    selection_count_0, selection_count_1 = 0, 0
    n_agreements_0, n_agreements_1 = 0, 0
    curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1 = 0, 0

    # Computing the sample means for each class in memory
    sample_means = None
    associated_label = []

    class_in_memory = np.unique(memory_y_name)
    # print(f'\nclasses in memory: {class_in_memory}')

    for class_idx in class_in_memory:
        indices = np.where(memory_y_name == int(class_idx))[0]
        if str(int(class_idx)) in minorityclass_ids:
            associated_label.append(1)
        else:
            associated_label.append(0)

        if sample_means is None:
            sample_means = torch.mean(torch.tensor(memory_X[indices]), dim=0).unsqueeze(0)
        else:
            sample_means = torch.cat((sample_means, torch.mean(torch.tensor(memory_X[indices]), dim=0).unsqueeze(0)), dim=0)
    associated_label = np.array(associated_label)

    # Setting unique y_classname labels for the new unseen labelled data (for buffer memory storage purpose)
    if unseen_task:
        attack_y_name = np.max(class_in_memory) + 1
        benign_y_name = np.max(class_in_memory) + 2
        print(f'New classes added to memory: {attack_y_name}, {benign_y_name}')

    if len(top_class_0_indices) > 1:

        top_class_0_data = X[class_0_indices[top_class_0_indices]]
        top_class_0_truth = y[class_0_indices[top_class_0_indices]]
        top_class_0_y_classname = y_classname[class_0_indices[top_class_0_indices]]

        # Member inference of the top samples based on distance from buffer memory samples
        start_inference = time.time()
        cos_dist = cdist(model.forward_encoder((torch.from_numpy((top_class_0_data))).to(device)).detach().cpu().numpy(), model.forward_encoder(torch.from_numpy(memory_X[np.where(memory_y ==0)]).to(device)).detach().cpu().numpy(),'cosine')   
        maj_labels = []
        row_indices_to_keep = []
        percentage_mode_value_contributors = []
        Avg_sample_support = []
        Avg_sample_support_counter = 0
        filtered_indices = cos_dist < (cos_dist_ip*math.exp(-temperature * labels_ratio)*1e-4)
        print(top_class_0_data.shape,filtered_indices.shape)
        row_indices_to_keep = np.where(np.any(filtered_indices, axis=1))[0]
        rows_to_keep = np.any(filtered_indices, axis=1)
        for i in range(filtered_indices.shape[0]):
            valid_indices = np.where(filtered_indices[i])[0]
            if valid_indices.size > 0:
                Avg_sample_support_counter += 1
                Avg_sample_support.append(valid_indices.size)
                Mode_value_and_count = stats.mode(memory_y[valid_indices])
                Mode_value_percentage = (Mode_value_and_count[1]/valid_indices.size)*100
                if Mode_value_percentage > mode_value:
                    maj_labels.append(Mode_value_and_count[0])
                    percentage_mode_value_contributors.append(Mode_value_percentage)
                else:
                    maj_labels.append(1)
                # rows_to_keep.append(True)
            else:
                maj_labels.append(1)#Adding a flipped label for class 0 samples as no confident labels (cost dis <0.2) found in the memory    
                # rowrows_to_keeps_to_keep.append(False)
        print(len(maj_labels))      
        # exit()  
        maj_labels = np.array(maj_labels)
        member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
        # member_inference_class_0 = np.asarray(maj_labels)
        print("Average number of sample support for Attack is", stats.tmean(Avg_sample_support), stats.tstd(Avg_sample_support))
        print("Percentage of samples contributed to each Attack sample is",stats.tmean(percentage_mode_value_contributors),stats.tstd(percentage_mode_value_contributors))
        # top_class_0_indices = top_class_0_indices[rows_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_0_truth = top_class_0_truth[rows_to_keep]#removes the indices whose cosine distance > 0.2           

       

        
        end_inference = time.time()
        print(f'\nNumber of class 0 agreements (between model and member inference): {np.sum(member_inference_class_0 == 0)}/{len(member_inference_class_0)} - ({np.sum(member_inference_class_0 == 0)*100./len(member_inference_class_0):.3f}%)')
        print(f'Number of class 0 common agreements with ground truth: {np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)}/{np.sum(member_inference_class_0 == 0)} - ({np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)*100./np.sum(member_inference_class_0 == 0):.3f}%)')
        print(f'Time taken for member inference = {end_inference - start_inference}seconds')

        n_agreements_0 = np.sum(member_inference_class_0 == 0)
        curr_truth_agreement_fraction_0 = np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)/np.sum(member_inference_class_0 == 0)
        if math.isnan(curr_truth_agreement_fraction_0):
            curr_truth_agreement_fraction_0 = 0

        # 0-(self)labelled data
        if unseen_task:
            if n_agreements_0 > 0:
                if truth_agreement_fraction_0 is None or math.isnan(truth_agreement_fraction_0):
                    truth_agreement_fraction_0 = 1
                selection_count_0 = int(n_agreements_0*truth_agreement_fraction_0)
                selected_0_indices = np.random.choice(top_class_0_indices[member_inference_class_0 == 0], size=selection_count_0, replace=False)
                labeled_indicies = np.hstack((labeled_indicies, selected_0_indices))

                labeled_X = np.vstack((labeled_X, X[selected_0_indices])) if labeled_X is not None else X[selected_0_indices]
                labeled_y = np.hstack((labeled_y, [0]*selection_count_0)) if labeled_y is not None else [0]*selection_count_0
                labeled_y_classname = np.hstack((labeled_y_classname, [benign_y_name]*selection_count_0)) if labeled_y_classname is not None else [benign_y_name]*selection_count_0
                print(f'No. of self-labelled samples (class 0): {selection_count_0}')

                owl_self_labelled_count_class_0 += selection_count_0
            else:
                print('No. of self-labelled samples (class 0): 0')

    if len(top_class_1_indices) > 1:#!= 0:
    
        top_class_1_data = X[class_1_indices[top_class_1_indices]]
        top_class_1_truth = y[class_1_indices[top_class_1_indices]]
        top_class_1_y_classname = y_classname[class_1_indices[top_class_1_indices]]
 
        # Member inference of the top samples based on distance from sample means
        start_inference = time.time()
        # cos_dist = cdist(top_class_1_data, memory_X,'cosine')   
        cos_dist = cdist(model.forward_encoder(torch.from_numpy(top_class_1_data).to(device)).detach().cpu().numpy(), model.forward_encoder(torch.from_numpy(memory_X[np.where(memory_y ==1)]).to(device)).detach().cpu().numpy(),'cosine')
        # sorted_indices_temp = np.argsort(cos_dist, axis=1)[::-1]
        # top_k_indices = sorted_indices_temp[:, :1000]
        # # Create boolean mask with True for top k indices, False otherwise
        # mask = np.zeros_like(cos_dist, dtype=bool)
        # mask[np.arange(len(cos_dist))[:, None], top_k_indices] = True        
        # filtered_indices = mask
        # # filtered_indices = cos_dist <max_value
        # row_indices_to_keep = np.any(filtered_indices, axis=1)
        # filtered_arr = filtered_indices[row_indices_to_keep]  
        # top_class_1_indices = top_class_1_indices[row_indices_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_1_truth = top_class_1_truth[row_indices_to_keep]#removes the indices whose cosine distance > 0.2        
        # maj_labels = []
        # for row in filtered_arr:
        #     maj_labels.append(stats.mode(memory_y[row])[0])
        # maj_labels = np.array(maj_labels)    
        # maj_labels = []
        # for row in filtered_arr:
        #     maj_labels.append(stats.mode(memory_y[row])[0])
        # maj_labels = np.array(maj_labels)          
        # member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
        # member_inference_class_0 = np.asarray(maj_labels)
        maj_labels = []
        row_indices_to_keep = []
        percentage_mode_value_contributors = []
        Avg_sample_support = []
        Avg_sample_support_counter = 0
        filtered_indices = cos_dist < (cos_dist_ip*math.exp(-temperature * labels_ratio)*1e-4)
        print(top_class_1_data.shape,filtered_indices.shape)
        row_indices_to_keep = np.where(np.any(filtered_indices, axis=1))[0]
        rows_to_keep = np.any(filtered_indices, axis=1)
        for i in range(filtered_indices.shape[0]):
            valid_indices = np.where(filtered_indices[i])[0]
            if valid_indices.size > 0:
                Avg_sample_support_counter += 1
                Avg_sample_support.append(valid_indices.size)
                Mode_value_and_count = stats.mode(memory_y[valid_indices])
                Mode_value_percentage = (Mode_value_and_count[1]/valid_indices.size)*100
                if Mode_value_percentage > mode_value:
                    maj_labels.append(Mode_value_and_count[0])
                    percentage_mode_value_contributors.append(Mode_value_percentage)
                else:
                    maj_labels.append(0)    

                # rows_to_keep.append(True)
            else:
                maj_labels.append(0)#Adding a flipped label for class 0 samples as no confident labels found in the memory    
            #     rows_to_keep.append(False)
        print(len(maj_labels))    
        maj_labels = np.array(maj_labels)
        member_inference_class_1 = np.asarray(maj_labels.ravel().tolist())
        print("Average number of sample support for Attack is", stats.tmean(Avg_sample_support), stats.tstd(Avg_sample_support))
        print("Percentage of samples contributed to each Attack sample is",stats.tmean(percentage_mode_value_contributors),stats.tstd(percentage_mode_value_contributors))
        # member_inference_class_1 = np.asarray(maj_labels)
        # top_class_1_indices = top_class_1_indices[rows_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_1_truth = top_class_1_truth[rows_to_keep]#removes the indices whose cosine distance > 0.2 
        # member_inference_class_1 = np.asarray(maj_labels.ravel().tolist())
        end_inference = time.time()
        print(f'\nNumber of class 1 agreements (between model and member inference): {np.sum(member_inference_class_1 == 1)}/{len(member_inference_class_1)} - ({np.sum(member_inference_class_1 == 1)*100./len(member_inference_class_1):.3f})%')
        print(f'Number of class 1 common agreements with ground truth: {np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)}/{np.sum(member_inference_class_1 == 1)} - ({np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)*100./np.sum(member_inference_class_1 == 1):.3f}%)')
        print(f'Time taken for member inference = {end_inference - start_inference}seconds')

        n_agreements_1 = np.sum(member_inference_class_1 == 1)
        curr_truth_agreement_fraction_1 = np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)/np.sum(member_inference_class_1 == 1)
        if math.isnan(curr_truth_agreement_fraction_1):
            curr_truth_agreement_fraction_1 = 0

        # 1-(self)labelled data
        if unseen_task:
            if n_agreements_1 > 0:
                if truth_agreement_fraction_1 is None or math.isnan(truth_agreement_fraction_1):
                    truth_agreement_fraction_1 = 1
                
                selection_count_1 = int(n_agreements_1*truth_agreement_fraction_1)
                selected_1_indices = np.random.choice(top_class_1_indices[member_inference_class_1 == 1], size=selection_count_1, replace=False)
                labeled_indicies = np.hstack((labeled_indicies, selected_1_indices))

                labeled_X = np.vstack((labeled_X, X[selected_1_indices])) if labeled_X is not None else X[selected_1_indices]
                labeled_y = np.hstack((labeled_y, [1]*selection_count_1)) if labeled_y is not None else [1]*selection_count_1
                labeled_y_classname = np.hstack((labeled_y_classname, [attack_y_name]*selection_count_1)) if labeled_y_classname is not None else [attack_y_name]*selection_count_1
                
                print(f'No. of self-labelled samples (class 1): {selection_count_1}')
                owl_self_labelled_count_class_1 += selection_count_1
            else:
                print('No. of self-labelled samples (class 1): 0')

    if not unseen_task:
        return [curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1]
    
    print(f'\nTotal no. of self-labeled samples = {selection_count_0 + selection_count_1} (0: {selection_count_0}, 1: {selection_count_1})')

    # Get security analyst to label the remaining high confidence samples
   
    
    remaining_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    y_rem = y[remaining_indices]

    # feat_unlab = model.forward_encoder(torch.from_numpy(X[remaining_indices]).to(device))
    # feat_unlab_normalized = F.normalize(feat_unlab, p=2, dim=1)
    # mem_samples_normalized = F.normalize(model.forward_encoder(torch.from_numpy(memory_X).to(device)),p=2,dim=1)
    # cos_dist_feat_unlab = cdist(feat_unlab_normalized.detach().cpu().numpy(), mem_samples_normalized.detach().cpu().numpy(),'cosine')
    # avg_cos_dis = cos_dist_feat_unlab.mean(axis=1)
    # sorted_indices = np.argsort(avg_cos_dis)[::-1]
    


    # top_selected_indices = sorted_indices[:min(len(remaining_indices),analyst_labels)]
    # true_indices = y[top_selected_indices]
    # zero_indices = np.where(true_indices == 0)[0]
    # zero_indices = sorted_indices[zero_indices]
    
    # one_indices = np.where(true_indices == 1)[0]
    # one_indices = sorted_indices[one_indices]
   

    # remaining_indices = np.setdiff1d(remaining_indices, (zero_indices.tolist()+one_indices.tolist()))
    

    
    
    # selected_0_indices = zero_indices
    # selected_1_indices = one_indices
    # count_class_0 = len(selected_0_indices)
    # count_class_1 = len(selected_1_indices)
    # temp_X = np.vstack((X[selected_0_indices], X[selected_1_indices]))
    # temp_y = np.hstack(([0]*count_class_0, [1]*count_class_1))
    # temp_y_classname = np.hstack(([benign_y_name]*count_class_0, [attack_y_name]*count_class_1))
    # print(f'No. of security analyst-labelled samples: {temp_X.shape[0]} (0:{len(selected_0_indices)}, 1:{len(selected_1_indices)})')

    # owl_analyst_labelled_count_class_0 += len(selected_0_indices)
    # owl_analyst_labelled_count_class_1 += len(selected_1_indices)

    # labeled_X = np.vstack((labeled_X, temp_X)) if labeled_X is not None else temp_X
    # labeled_y = np.hstack((labeled_y, temp_y)) if labeled_y is not None else temp_y
    # labeled_y_classname = np.hstack((labeled_y_classname, temp_y_classname)) if labeled_y_classname is not None else temp_y_classname
    # labeled_indicies = np.hstack((labeled_indicies, np.hstack((selected_0_indices, selected_1_indices))))
    # print(f'Total no. of labelled samples: {labeled_X.shape[0]}')

    # unlabeled_indicies = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    # X_unlab = X[unlabeled_indicies]
    # y_unlab = y[unlabeled_indicies]
    # y_classname_unlab = y_classname[unlabeled_indicies]
    # print(f'No. of unlabelled samples: {X_unlab.shape}\n')

    # labeled_indicies = labeled_indicies.astype(int)
    # unlabeled_indicies = unlabeled_indicies.astype(int)

    # return labeled_X,labeled_y,labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies

    # remaining_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    

    feat_unlab = model.forward_encoder(torch.from_numpy(X[remaining_indices]).to(device))
    feat_unlab_normalized = F.normalize(feat_unlab, p=2, dim=1)
    mem_samples_normalized = F.normalize(model.forward_encoder(torch.from_numpy(memory_X[np.where(memory_y==0)[0]]).to(device)),p=2,dim=1)
    cos_dist_feat_unlab = cdist(feat_unlab_normalized.detach().cpu().numpy(), mem_samples_normalized.detach().cpu().numpy(),'cosine')
    avg_cos_dis = cos_dist_feat_unlab.mean(axis=1)
    sorted_indices = np.argsort(avg_cos_dis)#[::-1]
    # y_sorted_indices = remaining_indices[sorted_indices]
    # true_indices = y[sorted_indices]


    top_selected_indices_0 = sorted_indices[:min(len(remaining_indices),int(analyst_labels*0.5))]
    # true_indices = y[top_selected_indices]
    # zero_indices = np.where(true_indices == 0)[0]
    # zero_indices = sorted_indices[zero_indices]
    zero_indices = top_selected_indices_0
    print(y[zero_indices])
    # print(y[sorted_indices[zero_indices]])


    mem_samples_normalized = F.normalize(model.forward_encoder(torch.from_numpy(memory_X[np.where(memory_y==1)[0]]).to(device)),p=2,dim=1)
    cos_dist_feat_unlab = cdist(feat_unlab_normalized.detach().cpu().numpy(), mem_samples_normalized.detach().cpu().numpy(),'cosine')
    avg_cos_dis = cos_dist_feat_unlab.mean(axis=1)
    sorted_indices = np.argsort(avg_cos_dis)#[::-1]
    
    # top_selected_indices_1 = sorted_indices[:min(len(remaining_indices),int(analyst_labels*0.5))]

    non_matching_indices = [idx for idx in sorted_indices if idx not in zero_indices]
    top_selected_indices_1 = non_matching_indices[:min(len(non_matching_indices), int(analyst_labels * 0.5))]

    # one_indices = np.where(true_indices == 1)[0]
    # one_indices = sorted_indices[one_indices]
    one_indices = top_selected_indices_1
    # print(y[sorted_indices[one_indices]])
    # print(y[sorted_indices[one_indices]])
    print(y[one_indices])

    # remaining_indices = np.setdiff1d(remaining_indices, (zero_indices.tolist()+one_indices.tolist()))
    remaining_indices = np.setdiff1d(remaining_indices, (zero_indices.tolist()+one_indices))
   
    
    selected_0_indices = zero_indices
    selected_1_indices = one_indices
    total_labels = y[selected_0_indices].tolist()+y[selected_1_indices].tolist()
    # count_class_0 = len(selected_0_indices)
    # count_class_1 = len(selected_1_indices)
    count_class_0 = total_labels.count(0)
    count_class_1 = total_labels.count(1)
    temp_X = np.vstack((X[selected_0_indices], X[selected_1_indices]))
    # temp_y = np.hstack(([0]*count_class_0, [1]*count_class_1))
    temp_y = np.hstack((y[selected_0_indices], y[selected_1_indices]))
    temp_y_classname = np.hstack(([benign_y_name]*count_class_0, [attack_y_name]*count_class_1))
    print(f'No. of security analyst-labelled samples: {temp_X.shape[0]} (0:{count_class_0}, 1:{count_class_1})')
    # exit()
    # owl_analyst_labelled_count_class_0 += len(selected_0_indices)
    # owl_analyst_labelled_count_class_1 += len(selected_1_indices)

    owl_analyst_labelled_count_class_0 += count_class_0
    owl_analyst_labelled_count_class_1 += count_class_1

    labeled_X = np.vstack((labeled_X, temp_X)) if labeled_X is not None else temp_X
    labeled_y = np.hstack((labeled_y, temp_y)) if labeled_y is not None else temp_y
    labeled_y_classname = np.hstack((labeled_y_classname, temp_y_classname)) if labeled_y_classname is not None else temp_y_classname
    # labeled_indicies = np.hstack((labeled_indicies, np.hstack((selected_0_indices, selected_1_indices))))
    labeled_indicies = np.hstack((selected_0_indices, selected_1_indices))
    print(f'Total no. of labelled samples: {labeled_X.shape[0]}')

    unlabeled_indicies = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    X_unlab = X[unlabeled_indicies]
    y_unlab = y[unlabeled_indicies]
    y_classname_unlab = y_classname[unlabeled_indicies]
    print(f'No. of unlabelled samples: {X_unlab.shape}\n')

    labeled_indicies = labeled_indicies.astype(int)
    unlabeled_indicies = unlabeled_indicies.astype(int)

    return labeled_X,labeled_y,labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies

# def train_new_method(str_train_model,tasks,task_class_ids,task_id,feature_list,threshold,X_val,y_val,bool_reorganize_memory,owl_data_labeling=False):
    
#     global memory_X, memory_y, memory_y_name,local_count,global_count,local_store,input_shape,memory_size,task_num
#     global classes_so_far,full,global_priority_list,local_priority_list,memory_population_time,replay_size
#     global memory_population_time,epochs,grad_norm_dict,temp_norm
#     global student_optimizer1,student_optimizer2,student_supervised_optimizer
#     global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised, encoder
#     global truth_agreement_fraction_0, truth_agreement_fraction_1 
#     global avg_CI, CI_list 
#     if str_train_model == "student1":
#         model = student_model1
#         opt = student_optimizer1
#         teacher_model = teacher_model1
#     elif str_train_model == "student2":
#         model = student_model2
#         opt = student_optimizer2
#         teacher_model = teacher_model2
#     elif str_train_model == "student_supervised":
#         model = student_supervised
#         opt = student_supervised_optimizer    
#         teacher_model = teacher_supervised    

#     grad_norm_list = []
#     valid_loader = torch.utils.data.DataLoader(dataset(X_val,y_val),
#                                                batch_size=batch_size,
#                                             #    sampler=valid_sampler,
#                                                num_workers=0)
#     feature_mat = []
#     X,y,y_classname = tasks[0][0],tasks[0][1],tasks[0][2]
#     y_large,y_small = max(np.sum(y == 0),np.sum(y == 1)),min(np.sum(y == 0),np.sum(y == 1))
#     print("majority class",y_large)
#     print("minority class",y_small)
#     print("class imbalance ratio",y_small/(y_large+y_small))
#     # print("computed class imbalce ratio", clustering_class_imbalance(X))
#     unique_y_classname = np.unique(y_classname)
#     if unique_y_classname[0]%2 == 0:
#         attack_y_name = unique_y_classname[0]
#         benign_y_name = unique_y_classname[1]
#     else:
#         attack_y_name = unique_y_classname[1]
#         benign_y_name = unique_y_classname[0]

#     # if task_id > 0:
#     #     compute_otdd(task_id, X, memory_X, memory_y_name, attack_y_name, benign_y_name)

#     task_size = X.shape[0]
#     if owl_data_labeling == False:

#         if task_id == 0:
#             labeled_indicies,unlabeled_indicies=split_a_task(tasks,0.99,task_class_ids)
#         else:
#             labeled_indicies,unlabeled_indicies=split_a_task(tasks,labels_ratio,task_class_ids)
#         labeled_X,labeled_y,labeled_y_classname = X[labeled_indicies],y[labeled_indicies],y_classname[labeled_indicies]
#         X_unlab,y_unlab,y_unlabclassname = X[unlabeled_indicies],y[unlabeled_indicies],y_classname[unlabeled_indicies]
        
#         # Computing class imbalance in the labeled samples
#         maj_class_count,min_class_count = max(np.sum(labeled_y == 0),np.sum(labeled_y == 1)),min(np.sum(labeled_y == 0),np.sum(labeled_y == 1))
#         CI_list.append(min_class_count/(maj_class_count+min_class_count))
#         # CI_list.append(maj_class_count/(min_class_count))
#         # CI_list.append(np.sum(labeled_y == 0)/(np.sum(labeled_y == 1) + np.sum(labeled_y == 0)))
#         # print("majority samples",maj_class_count)
#         # print("minority samples",min_class_count)
#         # CI_list.append(np.sum(labeled_y == 0)/(np.sum(labeled_y == 1) + np.sum(labeled_y == 0)))
#         print(f'Class Imbalance for task {task_id} (% of class 0 samples)= {CI_list[-1]}\n')
#         avg_CI = np.mean(CI_list)

#         # Computing class imbalance ratio for the task (0:1)
#         if task_id > 0:
#             task_truth_agreement_fractions = owl_data_labeling_strategy_const_labels(X, y, y_classname, unseen_task=False,task_id=task_id)
#             print(f'\nCurrent task truth agreement fractions = {task_truth_agreement_fractions}')
#             print(truth_agreement_fraction_0, truth_agreement_fraction_1)
#             ## accumulation
#             if truth_agreement_fraction_0 is None:
#                 truth_agreement_fraction_0 = task_truth_agreement_fractions[0] if task_truth_agreement_fractions[0] != 0 else None
#             else:
#                 truth_agreement_fraction_0 = beta*truth_agreement_fraction_0 + (1 - beta)*task_truth_agreement_fractions[0] if task_truth_agreement_fractions[0] != 0 else truth_agreement_fraction_0
            
#             if truth_agreement_fraction_1 is None:
#                 truth_agreement_fraction_1 = task_truth_agreement_fractions[1] if task_truth_agreement_fractions[1] != 0 else None
#             else:
#                 truth_agreement_fraction_1 = beta*truth_agreement_fraction_1 + (1 - beta)*task_truth_agreement_fractions[1] if task_truth_agreement_fractions[1] != 0 else truth_agreement_fraction_1

#             # truth_agreement_fraction_0, truth_agreement_fraction_1 = min(0.5,truth_agreement_fraction_0),min(0.5,truth_agreement_fraction_1)
#             print(truth_agreement_fraction_0, truth_agreement_fraction_1)
#         # #     print()


#     else:
#         labeled_X, labeled_y, labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies = owl_data_labeling_strategy_const_labels(X, y, y_classname, unseen_task=True,task_id=task_id)
        
#         # labeled_X_class_0,labeled_X_class_1,X_unlab = open_world_data_labeling(X)
#         # labeled_X = np.concatenate((labeled_X_class_0,labeled_X_class_1),axis=0)
#         # labeled_y = np.concatenate((np.zeros((labeled_X_class_0.shape[0],), dtype=np.int64),np.ones((labeled_X_class_1.shape[0],), dtype=np.int64)),axis=0)
#         # labeled_y_classname = np.concatenate((np.full((labeled_X_class_0.shape[0],),fill_value=benign_y_name,dtype=np.int64),np.full((labeled_X_class_1.shape[0],),fill_value=attack_y_name,dtype=np.int64)),axis=0)
#         # random_indices = np.random.permutation(len(labeled_X))
#         # labeled_X,labeled_y,labeled_y_classname = X[random_indices],y[random_indices],y_classname[random_indices]
#         # labeled_indicies = [lab_idx_num for lab_idx_num in range(labeled_X.shape[0])]
#         # unlabeled_indicies = [unlab_idx_num+len(labeled_indicies) for unlab_idx_num in range(X_unlab.shape[0])]
#         # labeled_indicies,unlabeled_indicies=open_world_data_labeling(X)
#         # labeled_X,labeled_y,labeled_y_classname = X[labeled_indicies],y[labeled_indicies],y_classname[labeled_indicies]
#         # X_unlab,y_unlab,y_unlabclassname = X[unlabeled_indicies],y[unlabeled_indicies],y_classname[unlabeled_indicies]
        
#         # labeled_X_class_0,labeled_X_class_1,X_unlab = open_world_data_labeling(X)
#         # print("expected class zero labels",labeled_X_class_0.shape)
#         # print("actual class zero labels,",Counter(y[labeled_X_class_0]))
#         # print("top 20 class 0 labels",y[labeled_X_class_0[0:20]])
#         # print("expected class one labels",labeled_X_class_1.shape)
#         # print("actual class one labels,",Counter(y[labeled_X_class_1]))
    
    
#     if task_id > 0:
              
#             mem_batch_size = floor(batch_size*b_m)
#             rem_batch_size = batch_size-mem_batch_size
#             # task_size = X.shape[0] + memory_X.shape[0] 
#             labeled_batch_size = floor(rem_batch_size*labels_ratio)
#             unlabeled_batch_size = rem_batch_size - (labeled_batch_size)
#             no_of_batches = floor(len(labeled_indicies)/labeled_batch_size)
#             no_of_unlab_batches = floor(len(unlabeled_indicies)/unlabeled_batch_size)
#             p = np.random.permutation(labeled_X.shape[0])
#             labeled_X,labeled_y,labeled_y_classname = labeled_X[p,:],labeled_y[p],labeled_y_classname[p]
            
#     else:
#         # initialize_buffermemory(labeled_task,memory_size)
#         task_size = X.shape[0]    
#         # labeled_batch_size = floor(batch_size*0.99)
#         labeled_batch_size = floor(batch_size*labels_ratio)
#         unlabeled_batch_size = batch_size-labeled_batch_size
#         no_of_batches = floor(task_size/batch_size)

#     if bool_gpm:
#         for i in range(len(feature_list)):
#             Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
#             feature_mat.append(Uf)    
    

#     ###Buffer memory organization
#     temp_x,temp_y,temp_yname = labeled_X,labeled_y,labeled_y_classname
#     if task_id > 0 and bool_reorganize_memory:
#         mem_start_time = time.time()
#         if str(mem_strat) == "replace":
            
#             tasks[0] = temp_x,temp_y,temp_yname
#             lab_samples_in_memory = split_a_task(tasks,lab_samp_in_mem_ratio)
#             tasks[0] = temp_x[lab_samples_in_memory[0],:],temp_y[lab_samples_in_memory[0]],temp_yname[lab_samples_in_memory[0]]
#             initialize_buffermemory(tasks=tasks,mem_size=memory_size)
#         elif str(mem_strat) == "equal":
            
#             memory_X, memory_y, memory_y_name = memory_update_equal_allocation2(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.15,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)
#         else:
            
#             memory_X, memory_y, memory_y_name = memory_update_equal_allocation(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.85,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)

#         mem_finish_time = time.time()
#         memory_population_time += mem_finish_time-mem_start_time


#     ##Training encoder for self-supervision
#     print("************Training the Encoder***********")
#     # encoder = train_encoder (encoder, X_train, y_train, optimizer, total_epochs, batch_size, is_tensor=True)


#     # prog_bar = tqdm(range(no_of_batches))
#     # for batch_idx in prog_bar:
#     # to track the training loss as the model trains
#     train_losses = []
#     # to track the validation loss as the model trains
#     valid_losses = []
#     # to track the average training loss per epoch as the model trains
#     avg_train_losses = []
#     # to track the average validation loss per epoch as the model trains
#     avg_valid_losses = [] 
#     check_point_file_name = "checkpoint"+str(os.getpid())+".pt"
#     check_point_file_name_norm = "checkpoint"+str(os.getpid())+"grad_norm"+".pt"
#     early_stopping = EarlyStopping(patience=3, verbose=True,path=check_point_file_name)
#     gradient_rejection = GradientRejection(patience=2, verbose=True,path=check_point_file_name_norm)
#     scheduler = StepLR(opt, step_size=1, gamma=0.96)
#     for epoch in range(epochs):
#         # print("epoch",epoch)
#         # scheduler.step()
#         prog_bar = tqdm(range(no_of_batches))
#         for batch_idx in prog_bar:
#             model.train()        
#         # for epoch in range(epochs):
#             with torch.no_grad():
#                 if task_id > 0 and batch_idx < no_of_unlab_batches:
#                     unlabeled_X = torch.from_numpy(X_unlab[batch_idx*unlabeled_batch_size:batch_idx*unlabeled_batch_size+unlabeled_batch_size]).to(device)
#                 else:
#                     rand_indices = list(random.sample(range(X_unlab.shape[0]),min(unlabeled_batch_size,X_unlab.shape[0])))
#                     unlabeled_X = torch.from_numpy(X_unlab[rand_indices]).to(device)
#                 if image_resolution is not None:
#                     unlabeled_X = unlabeled_X.reshape(image_resolution)
#                 unlabeled_pred = torch.softmax(model(unlabeled_X),dim=1).detach()
#             lab_X = labeled_X[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]  
#             lab_y = labeled_y[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]
#             task_lab_X,task_lab_y = lab_X,lab_y
#             if task_id > 0:
                
#                 mem_batch = sample_batch_from_memory(floor(batch_size*b_m),minority_alloc=batch_minority_alloc)
#                 if mem_batch is not None and mem_batch[0].shape[0] > 0:
                    
#                     lab_X = np.concatenate((lab_X,mem_batch[0]), axis=0)  
#                     # temp_mem_X = torch.from_numpy(mem_batch[0]).to(device)
#                     # if image_resolution is not None:
#                     #     temp_mem_X = temp_mem_X.reshape(image_resolution)
#                     # temp_mem_y = torch.argmax(teacher_model(temp_mem_X),dim=1).detach().cpu().numpy().squeeze()
#                     # lab_y = np.concatenate((lab_y,temp_mem_y), axis=0)
#                     lab_y = np.concatenate((lab_y,mem_batch[1]), axis=0)
#             lab_X = torch.from_numpy(lab_X).to(device)
#             if image_resolution is not None:
#                     lab_X = lab_X.reshape(image_resolution)
                   
#             # print(model(lab_X))
#             y_pred = torch.softmax(model(lab_X),dim=1).squeeze()#.to(device)                      
#             lab_y = torch.from_numpy(lab_y).to(device).to(dtype=torch.long)#.reshape(y_pred.shape)

#             # lab_y = F.one_hot(lab_y, 2)
#             sup_loss = loss_fn(y_pred.float(),F.one_hot(lab_y.to(dtype=torch.long), 2).float())#to(device)
#             # sup_loss = loss_fn(y_pred,lab_y.float())
#             total_loss = sup_loss
#             distil_loss = 0
#             distil_loss = torch.as_tensor(distil_loss).to(device)
#             opt.zero_grad()
#             if task_id > 0:
#                 if str_train_model!="student_supervised":
#                     #computing the distillation loss
#                     # distil_loss_list = compute_distill_loss(unlabeled_pred,unlabeled_X)
#                     # distil_loss = distil_loss_list[0]

#                     # distil_loss = compute_distill_loss_self_supervision(p_m=0.3, K=3,unlabeled_x=unlabeled_X,encoder_model=encoder)
#                     total_loss = total_loss 
#                     # total_loss = total_loss +  alpha *distil_loss  
#                     # lab_X = torch.cat((lab_X,unlabeled_X),0)
#                     # lab_y = torch.cat((lab_y,distil_loss_list[1]),0)

#                 contrast_loss = 0
                
#                 if bool_closs:
#                     # positives, negatives = construct_positive_negative_samples(lab_X, lab_y)  
#                     positives, negatives = construct_positive_negative_samples_from_memory(task_lab_y)
#                     anchor_representations = model(torch.from_numpy(task_lab_X).to(device)) ### Get Encoder Representations
#                     positive_representations = model(positives) ### Get Encoder Representations
#                     negative_representations = model(negatives) ### Get Encoder Representations
#                     contrast_loss = contrastive_loss(anchor_representations, positive_representations, negative_representations)
#                 total_loss = total_loss+contrast_loss
#                 # print(y_pred)
#                 # print("total_loss",total_loss)
#                 if bool_gpm:
#                     total_loss.backward()
#                     # for i in range(len(feature_list)):
#                     #     Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
#                     #     feature_mat.append(Uf)
#                     bn_counter = 0
#                     for k, (m,params) in enumerate(model.named_parameters()):
#                         # print(params.grad)
#                         # print(m)
#                         if 'bn' not in m:
#                             k -= bn_counter
#                             sz =  params.grad.data.size(0)
#                             params.grad.data = torch.mul((params.grad.data - torch.mul(torch.mm(params.grad.data.view(sz,-1),\
#                                                     feature_mat[k]).view(params.size()),1)), (1))  
#                         else:
#                             bn_counter += 1    

#             else:       
#                 total_loss.backward()

            
            
#             opt.step() 
#             # teacher_model.load_state_dict(model.state_dict(), strict=False)
#             # gradient_rejection(model=model)
#             # if gradient_rejection.early_stop:
#             #     torch.save(model.state_dict(), check_point_file_name_norm)
#             train_losses.append(total_loss.item())

#             y_pred = y_pred.detach().cpu().numpy()
#             lab_y = lab_y.detach().cpu().numpy()
            
#             # lr_precision, lr_recall, _ = precision_recall_curve(lab_y, y_pred,pos_label=1)
#             # lr_auc_outlier =  auc(lr_recall, lr_precision)
            
        

#             # lr_precision, lr_recall, _ = precision_recall_curve(lab_y, [1-x for x in y_pred],pos_label=0)
#             # lr_auc_inliers =  auc(lr_recall, lr_precision)   
#             # prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f} - PR-AUC(inliers): {:.2f} - PR_auc(outlier)_curve {:.3f}'.format(
#             #      total_loss.item(), sup_loss.item(), distil_loss.item(), lr_auc_inliers,lr_auc_outlier ))
#             # r_auc = roc_auc_score(lab_y, y_pred)
#             # prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f}'.format(
#             #      total_loss, sup_loss, distil_loss))
#             prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f}'.format(
#                  total_loss.item(), sup_loss.item(), distil_loss.item()))
        
#         model.eval() # prep model for evaluation
#         val_pred,val_gt = [],[]
#         for data, target in valid_loader:
#             # pred = torch.argmax(model(data.to(device)),dim=1).reshape(target.shape)
#             pred = model(data.to(device))[:,1].reshape(target.shape)
#             y_pred = pred.detach().cpu().numpy().tolist()
#             val_pred.extend(y_pred)
#             val_gt.extend(target.detach().cpu().numpy().tolist())
#         lr_precision, lr_recall, _ = precision_recall_curve(val_gt, [x for x in val_pred], pos_label=1.)
#         lr_auc_minority =  auc(lr_recall, lr_precision)
#         # lr_precision, lr_recall, _ = precision_recall_curve(val_gt, val_pred, pos_label=1.)
#         # lr_auc_majority=  auc(lr_recall, lr_precision)
#         lr_auc = lr_auc_minority#[lr_auc_minority,lr_auc_majority]
#         # lr_auc = f1_score(val_gt,val_pred)
#             # calculate the loss
#             # loss = loss_fn(pred, target.to(device))
#             # record validation loss
#             # valid_losses.append(loss.item())
#         # valid_losses.append(np.nan_to_num(lr_auc))
#         # print training/validation statistics 
#         # calculate average loss over an epoch
#         train_loss = np.average(train_losses)
#         # valid_loss = np.average(valid_losses)
#         avg_train_losses.append(train_loss)
#         # avg_valid_losses.append(valid_loss)
#         epoch_len = len(str(epochs))
        
#         print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
#                      f'train_loss: {train_loss:.5f} ' +
#                      f'PR-AUC (I): {lr_auc:.5f}')
        
#         print(print_msg)
        
#         # clear lists to track next epoch
#         train_losses = []
#         valid_losses = []
        
#         # early_stopping needs the validation loss to check if it has decresed, 
#         # and if it has, it will make a checkpoint of the current model
#         early_stopping(lr_auc, model)
#         if early_stopping.counter <1:
#             scheduler.step()

#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
#     # load the last checkpoint with the best model
#     model.load_state_dict(torch.load(check_point_file_name))
#     teacher_model.load_state_dict(torch.load(check_point_file_name))

#     # temp_x,temp_y,temp_yname = X[labeled_indicies,:],y[labeled_indicies],y_classname[labeled_indicies]
    
#     # temp_x,temp_y,temp_yname = X[unlabeled_indicies,:],y[unlabeled_indicies],y_classname[unlabeled_indicies]
    

#     # temp_x,temp_y,temp_yname = labeled_X,labeled_y,labeled_y_classname
#     # if task_id > 0 and bool_reorganize_memory:
#     #     mem_start_time = time.time()
#     #     if str(mem_strat) == "replace":
            
#     #         tasks[0] = temp_x,temp_y,temp_yname
#     #         lab_samples_in_memory = split_a_task(tasks,lab_samp_in_mem_ratio)
#     #         tasks[0] = temp_x[lab_samples_in_memory[0],:],temp_y[lab_samples_in_memory[0]],temp_yname[lab_samples_in_memory[0]]
#     #         initialize_buffermemory(tasks=tasks,mem_size=memory_size)
#     #     elif str(mem_strat) == "equal":
            
#     #         memory_X, memory_y, memory_y_name = memory_update_equal_allocation2(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.15,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)
#     #     else:
            
#     #         memory_X, memory_y, memory_y_name = memory_update_equal_allocation(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.85,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)

#     #     mem_finish_time = time.time()
#     #     memory_population_time += mem_finish_time-mem_start_time

#     # mat_list = []    
#     temp_x,temp_y,temp_yname = X[labeled_indicies,:],y[labeled_indicies],y_classname[labeled_indicies]
#     # mat_list = get_representation_matrix (model, device, temp_x, temp_y)
#     if bool_gpm:
#         mat_list = get_representation_matrix (model, device, temp_x, temp_y,rand_samples=no_of_rand_samples)
#         feature_list = update_GPM(model, mat_list, threshold, feature_list)
#     else:
#         feature_list = []

#     # grad_norm_dict[task_id] = grad_norm_list   
#     # print(grad_norm_dict)
#     if os.path.exists(check_point_file_name):
#         os.remove(check_point_file_name)
#     if os.path.exists(check_point_file_name_norm):
#         os.remove(check_point_file_name_norm) 

#     # print(f'Buffer memory size for task {task_id}: {memory_X.shape}')
    
#     return feature_list

     
def train(str_train_model,tasks,task_class_ids,task_id,feature_list,threshold,X_val,y_val,bool_reorganize_memory,owl_data_labeling=False):
    
    global memory_X, memory_y, memory_y_name,local_count,global_count,local_store,input_shape,memory_size,task_num
    global classes_so_far,full,global_priority_list,local_priority_list,memory_population_time,replay_size
    global memory_population_time,epochs,grad_norm_dict,temp_norm
    global student_optimizer1,student_optimizer2,student_supervised_optimizer
    global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised
    global truth_agreement_fraction_0, truth_agreement_fraction_1 
    global avg_CI, CI_list 

    if str_train_model == "student1":
        model = student_model1
        opt = student_optimizer1
        teacher_model = teacher_model1
    elif str_train_model == "student2":
        model = student_model2
        opt = student_optimizer2
        teacher_model = teacher_model2
    elif str_train_model == "student_supervised":
        model = student_supervised
        opt = student_supervised_optimizer    
        teacher_model = teacher_supervised    
    
    grad_norm_list = []

    valid_loader = torch.utils.data.DataLoader(dataset(X_val,y_val),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=0)
    feature_mat = []
    X,y,y_classname = tasks[0][0],tasks[0][1],tasks[0][2]
    y_large,y_small = max(np.sum(y == 0),np.sum(y == 1)),min(np.sum(y == 0),np.sum(y == 1))
    print("majority class",y_large)
    print("minority class",y_small)
    print("class imbalance ratio",y_small/(y_large+y_small))
    # print("computed class imbalce ratio", clustering_class_imbalance(X))
    unique_y_classname = np.unique(y_classname)
    if unique_y_classname[0]%2 == 0:
        attack_y_name = unique_y_classname[0]
        benign_y_name = unique_y_classname[1]
    else:
        attack_y_name = unique_y_classname[1]
        benign_y_name = unique_y_classname[0]

    # if task_id > 0:
    #     compute_otdd(task_id, X, memory_X, memory_y_name, attack_y_name, benign_y_name)

    task_size = X.shape[0]
    if owl_data_labeling == False:

        # if task_id == 0:
        if task_id == 0:    
            labeled_indicies,unlabeled_indicies=split_a_task(tasks,0.99,task_class_ids)
        else:
            labeled_indicies,unlabeled_indicies=split_a_task(tasks,labels_ratio,task_class_ids)
        labeled_X,labeled_y,labeled_y_classname = X[labeled_indicies],y[labeled_indicies],y_classname[labeled_indicies]
        X_unlab,y_unlab,y_unlabclassname = X[unlabeled_indicies],y[unlabeled_indicies],y_classname[unlabeled_indicies]
        
        # Computing class imbalance in the labeled samples
        maj_class_count,min_class_count = max(np.sum(labeled_y == 0),np.sum(labeled_y == 1)),min(np.sum(labeled_y == 0),np.sum(labeled_y == 1))
        CI_list.append(min_class_count/(maj_class_count+min_class_count))
        # CI_list.append(maj_class_count/(min_class_count))
        # CI_list.append(np.sum(labeled_y == 0)/(np.sum(labeled_y == 1) + np.sum(labeled_y == 0)))
        # print("majority samples",maj_class_count)
        # print("minority samples",min_class_count)
        # CI_list.append(np.sum(labeled_y == 0)/(np.sum(labeled_y == 1) + np.sum(labeled_y == 0)))
        print(f'Class Imbalance for task {task_id} (% of class 0 samples)= {CI_list[-1]}\n')
        avg_CI = np.mean(CI_list)

        # Computing class imbalance ratio for the task (0:1)
        if task_id > 0:
            task_truth_agreement_fractions = owl_data_labeling_strategy_const_labels2(X, y, y_classname, unseen_task=False,task_id=task_id,analyst_labels=analyst_labels)
            print(f'\nCurrent task truth agreement fractions = {task_truth_agreement_fractions}')
            print(truth_agreement_fraction_0, truth_agreement_fraction_1)
            ## accumulation
            if truth_agreement_fraction_0 is None:
                truth_agreement_fraction_0 = task_truth_agreement_fractions[0] if task_truth_agreement_fractions[0] != 0 else None
            else:
                truth_agreement_fraction_0 = beta*truth_agreement_fraction_0 + (1 - beta)*task_truth_agreement_fractions[0] if task_truth_agreement_fractions[0] != 0 else truth_agreement_fraction_0
            
            if truth_agreement_fraction_1 is None:
                truth_agreement_fraction_1 = task_truth_agreement_fractions[1] if task_truth_agreement_fractions[1] != 0 else None
            else:
                truth_agreement_fraction_1 = beta*truth_agreement_fraction_1 + (1 - beta)*task_truth_agreement_fractions[1] if task_truth_agreement_fractions[1] != 0 else truth_agreement_fraction_1

            # truth_agreement_fraction_0, truth_agreement_fraction_1 = min(0.5,truth_agreement_fraction_0),min(0.5,truth_agreement_fraction_1)
            print(truth_agreement_fraction_0, truth_agreement_fraction_1)
        # #     print()


    else:
        labeled_X, labeled_y, labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies = owl_data_labeling_strategy_const_labels2(X, y, y_classname, unseen_task=True,task_id=task_id,analyst_labels=analyst_labels)
    
        # labeled_X_class_0,labeled_X_class_1,X_unlab = open_world_data_labeling(X)
        # labeled_X = np.concatenate((labeled_X_class_0,labeled_X_class_1),axis=0)
        # labeled_y = np.concatenate((np.zeros((labeled_X_class_0.shape[0],), dtype=np.int64),np.ones((labeled_X_class_1.shape[0],), dtype=np.int64)),axis=0)
        # labeled_y_classname = np.concatenate((np.full((labeled_X_class_0.shape[0],),fill_value=benign_y_name,dtype=np.int64),np.full((labeled_X_class_1.shape[0],),fill_value=attack_y_name,dtype=np.int64)),axis=0)
        # random_indices = np.random.permutation(len(labeled_X))
        # labeled_X,labeled_y,labeled_y_classname = X[random_indices],y[random_indices],y_classname[random_indices]
        # labeled_indicies = [lab_idx_num for lab_idx_num in range(labeled_X.shape[0])]
        # unlabeled_indicies = [unlab_idx_num+len(labeled_indicies) for unlab_idx_num in range(X_unlab.shape[0])]
        # labeled_indicies,unlabeled_indicies=open_world_data_labeling(X)
        # labeled_X,labeled_y,labeled_y_classname = X[labeled_indicies],y[labeled_indicies],y_classname[labeled_indicies]
        # X_unlab,y_unlab,y_unlabclassname = X[unlabeled_indicies],y[unlabeled_indicies],y_classname[unlabeled_indicies]
        
        # labeled_X_class_0,labeled_X_class_1,X_unlab = open_world_data_labeling(X)
        # print("expected class zero labels",labeled_X_class_0.shape)
        # print("actual class zero labels,",Counter(y[labeled_X_class_0]))
        # print("top 20 class 0 labels",y[labeled_X_class_0[0:20]])
        # print("expected class one labels",labeled_X_class_1.shape)
        # print("actual class one labels,",Counter(y[labeled_X_class_1]))
    
    ### Calculating Uncertainity ------------------------ ------------------------ ------------------------ ------------------------ ------------------------ ------------------------
    uncertainity_unlabelled_data = X[unlabeled_indicies]
    uncertainity_unlabelled_data_loader = torch.utils.data.DataLoader(uncertainity_unlabelled_data, batch_size=100, shuffle=False, num_workers=1)
    uncertainity = get_uncertainity(model,device,uncertainity_unlabelled_data_loader)
    # if owl_data_labeling ==False:
    #     ce = MarginLoss(m=-1*uncertainity)
    # else:
    #     ce = MarginLoss(m=-0*uncertainity)    
    ce = MarginLoss(m=-0*uncertainity)    
    #------------------------ ------------------------ ------------------------ ------------------------ ------------------------ ------------------------ ----------------------------
    if task_id > 0:
            mem_batch_size = floor(batch_size*b_m)
            rem_batch_size = batch_size-mem_batch_size
            # task_size = X.shape[0] + memory_X.shape[0] 
            labeled_batch_size = floor(rem_batch_size*labels_ratio)
            unlabeled_batch_size = rem_batch_size - (labeled_batch_size)
            no_of_labeled_batches = floor(len(labeled_indicies)/labeled_batch_size)
            # no_of_batches = floor(len(labeled_indicies)/labeled_batch_size)
            no_of_unlab_batches = floor(len(unlabeled_indicies)/unlabeled_batch_size)
            p = np.random.permutation(labeled_X.shape[0])
            labeled_X,labeled_y,labeled_y_classname = labeled_X[p,:],labeled_y[p],labeled_y_classname[p]
            no_of_batches = max(no_of_labeled_batches,no_of_unlab_batches)
            print(f"mem_batch:{mem_batch_size}_{labeled_batch_size}_{unlabeled_batch_size},labeled batchs:{no_of_batches} and unlabaled batches:{no_of_unlab_batches}")
            # exit()
            
    else:
        # initialize_buffermemory(labeled_task,memory_size)
        task_size = X.shape[0]    
        # labeled_batch_size = floor(batch_size*0.99)
        mem_batch_size = floor(batch_size*b_m)
        rem_batch_size = batch_size-mem_batch_size
        labeled_batch_size = rem_batch_size
        # labeled_batch_size = floor(batch_size*labels_ratio)
        # unlabeled_batch_size = batch_size-labeled_batch_size
        # no_of_batches = floor(task_size/batch_size)
        no_of_batches = floor(task_size/rem_batch_size)
        unlabeled_batch_size=2
        no_of_labeled_batches = no_of_batches

    if bool_gpm:
        for i in range(len(feature_list)):
            Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
            # print("UFF is here ", Uf)
            feature_mat.append(Uf)    
    

    ###Buffer memory organization
    temp_x,temp_y,temp_yname = labeled_X,labeled_y,labeled_y_classname
    if task_id > 0 and bool_reorganize_memory:
        mem_start_time = time.time()
        if str(mem_strat) == "replace":
            
            tasks[0] = temp_x,temp_y,temp_yname
            lab_samples_in_memory = split_a_task(tasks,lab_samp_in_mem_ratio)
            tasks[0] = temp_x[lab_samples_in_memory[0],:],temp_y[lab_samples_in_memory[0]],temp_yname[lab_samples_in_memory[0]]
            initialize_buffermemory(tasks=tasks,mem_size=memory_size)
        elif str(mem_strat) == "equal":
            
            memory_X, memory_y, memory_y_name = memory_update_equal_allocation2(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.15,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)
        else:
            
            memory_X, memory_y, memory_y_name = memory_update_equal_allocation(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.85,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)

        mem_finish_time = time.time()
        memory_population_time += mem_finish_time-mem_start_time


    ###Training encoder for self-supervision
    # print("************Training the Encoder***********")
    # encoder = vime_self(device,X_unlab, p_m=0.3, alpha=2.0, parameters={'epochs': 5, 'batch_size': 32})
    # encoder = encoder.eval()


    # prog_bar = tqdm(range(no_of_batches))
    # for batch_idx in prog_bar:
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    ce_losses = []
    bce_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    avg_rejected_samples = []
    check_point_file_name = "checkpoint"+str(os.getpid())+".pt"
    check_point_file_name_norm = "checkpoint"+str(os.getpid())+"grad_norm"+".pt"
    if ds == 'bodmas':
        early_stopping = EarlyStopping(patience=5, verbose=True,path=check_point_file_name)
    else:
        early_stopping = EarlyStopping(patience=5, verbose=True,delta=0.001,path=check_point_file_name)
 
    gradient_rejection = GradientRejection(patience=2, verbose=True,path=check_point_file_name_norm)
    scheduler = StepLR(opt, step_size=1, gamma=0.96)
    
    for epoch in range(epochs):
        # print("epoch",epoch)
        # scheduler.step()
        prog_bar = tqdm(range(no_of_batches))
        basis = None
        batch_rejected_samples = 0 
        for batch_idx in prog_bar:
            
            model.train()        
        # for epoch in range(epochs):
            with torch.no_grad():
                if task_id > 0 and batch_idx < no_of_unlab_batches:
                    unlabeled_X = torch.from_numpy(X_unlab[batch_idx*unlabeled_batch_size:batch_idx*unlabeled_batch_size+unlabeled_batch_size]).to(device)
                else:
                    rand_indices = list(random.sample(range(X_unlab.shape[0]),min(unlabeled_batch_size,X_unlab.shape[0])))
                    unlabeled_X = torch.from_numpy(X_unlab[rand_indices]).to(device)
                # if image_resolution is not None and image_resolution is not None:
                #     unlabeled_X = unlabeled_X.reshape(image_resolution)
                # unlabeled_pred = torch.softmax(model(unlabeled_X),dim=1).detach()
                
                if task_id >= 0 and batch_idx < no_of_labeled_batches:
                    lab_X = labeled_X[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]  
                    lab_y = labeled_y[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]
                else:
                    rand_indices = list(random.sample(range(labeled_X.shape[0]),min(labeled_batch_size,labeled_X.shape[0])))
                    # indices_1 = list(np.where(labeled_y == 1)[0] )
                    # rand_indices = list(random.sample(indices_1,min(labeled_batch_size,len(indices_1))))
                    # if len(rand_indices)<labeled_batch_size:
                    #     indices_0 = np.where(labeled_y == 0)[0]
                    #     num_zeros = labeled_batch_size - len(rand_indices)
                    #     rand_indices_0 = list(random.sample(list(indices_0), num_zeros))
                    #     rand_indices = rand_indices+rand_indices_0
                        


                     
                    lab_X = labeled_X[rand_indices]  
                    lab_y = labeled_y[rand_indices]

                    # Find indices of samples with label 1
                    # indices_1 = np.where(labeled_y == 1)[0] 
                    # # Find indices of samples with label 0
                    # indices_0 = np.where(labeled_y == 0)[0]
                    # # Sample indices for label 1 (up to the desired batch size)
                    # num_ones = min(labeled_batch_size, len(indices_1)) 
                    # if num_ones < 1:#int(labeled_batch_size/2):
                    #     mem_indices_1 = np.where(memory_y == 1)[0]
                    #     num_ones = min(int(labeled_batch_size*0.9), len(mem_indices_1)) 
                    #     rand_indices_1 = list(random.sample(list(mem_indices_1), num_ones))
                    #     memory_lab_X_1 = memory_X[rand_indices_1]
                    #     memory_lab_y_1 = memory_y[rand_indices_1]
                    #     num_zeros = labeled_batch_size - num_ones 
                    #     rand_indices_0 = list(random.sample(list(indices_0), num_zeros))
                    #     lab_X = labeled_X[rand_indices_0]  
                    #     lab_y = labeled_y[rand_indices_0]
                    #     lab_X = np.concatenate((lab_X,memory_lab_X_1), axis=0)
                    #     lab_y = np.concatenate((lab_y,memory_lab_y_1), axis=0)
                        

                    # else:
                    #     rand_indices_1 = list(random.sample(list(indices_1), num_ones))

                    #     # Calculate the remaining slots for label 0
                    #     num_zeros = labeled_batch_size - num_ones 

                    #     # Sample indices for label 0
                    #     rand_indices_0 = list(random.sample(list(indices_0), num_zeros))

                    #     # Combine indices
                    #     # rand_indices = np.concatenate((rand_indices_1, rand_indices_0))
                    #     rand_indices = rand_indices_0+rand_indices_1
                    #     lab_X = labeled_X[rand_indices]  
                    #     lab_y = labeled_y[rand_indices]    


            # lab_X = labeled_X[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]  
            # lab_y = labeled_y[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]
            # task_lab_X,task_lab_y = lab_X,lab_y
            if task_id >= 0:
                
                mem_batch = sample_batch_from_memory(floor(batch_size*b_m),minority_alloc=batch_minority_alloc)
                if mem_batch is not None and mem_batch[0].shape[0] > 0:
                    
                    lab_X = np.concatenate((lab_X,mem_batch[0]), axis=0)  
                    # temp_mem_X = torch.from_numpy(mem_batch[0]).to(device)
                    # if image_resolution is not None:
                    #     temp_mem_X = temp_mem_X.reshape(image_resolution)
                    # temp_mem_y = torch.argmax(teacher_model(temp_mem_X),dim=1).detach().cpu().numpy().squeeze()
                    
                    # lab_y = np.concatenate((lab_y,temp_mem_y), axis=0)
                    
                    lab_y = np.concatenate((lab_y,mem_batch[1]), axis=0)
            lab_X = torch.from_numpy(lab_X).to(device)
            # print(model(lab_X))
            y_pred = torch.softmax(model(lab_X),dim=1).squeeze()#.to(device)                      
            lab_y = torch.from_numpy(lab_y).to(device).to(dtype=torch.long)#.reshape(y_pred.shape)
            
            # lab_y = F.one_hot(lab_y, 2)
            sup_loss = loss_fn(y_pred.float(),F.one_hot(lab_y.to(dtype=torch.long), 2).float())#to(device)
            # sup_loss = loss_fn(y_pred,lab_y.float())
            # total_loss = sup_loss
            #New Method Losses  ---------------------------------------------------------------------------------------------------------
            # Step 1: Create Two set of Data :
            labeled_len = len(lab_X)
            unlabeled_data = unlabeled_X
            x1 = torch.cat([lab_X,unlabeled_data])    
            output1,feat = model(x1),model.forward_encoder(x1)   
            prob1 = torch.softmax(output1,dim=1).squeeze()
            # `prob2 = torch.softmax(output2,dim=1).squeeze() 
            # feat_detached = feat.detach()
            # feat_norm = feat_detached / torch.norm(feat_detached,2,1,keepdim=True)
            # cosine_dist = torch.mm(feat_norm,feat_norm.t())`

            
            ## For the Label Part use the ground truth to find the positive Pairs
            pos_pairs = []
            target_np  = lab_y.cpu().numpy()
            for i in range(labeled_len):
                t_i = target_np[i]
                idx = np.where(target_np == t_i)[0]
                if len(idx) ==1 :
                    pos_pairs.append(idx[0])
                else:
                    ran_idx = np.random.choice(idx,1)
                    while ran_idx ==i :
                        ran_idx = np.random.choice(idx,1)
                    pos_pairs.append(int(ran_idx))
            ## For the labelled part use the cosine similarity to find the positive pairs
            ## New Projection Step
            if batch_idx  == 0 :
                memory_torch = torch.cat([torch.from_numpy(memory_X).to(device)])
                mem_feat = model.forward_encoder(memory_torch)
                mem_feat = mem_feat.detach()
                U, S, Vh = torch.linalg.svd(mem_feat, full_matrices=False)  # GPU-compatible SVD
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio.detach().cpu().numpy())<0.95)
                basis = Vh[:r]
            # Normalize mem_feat directly on the GPU
            bce_loss = torch.tensor(0, dtype=torch.int32)
            if task_id > 0:
                mem_feat_norm = F.normalize(mem_feat, p=2, dim=1)
            # Projection of Unlabelled onto the that space
                feat_unlab = model.forward_encoder(unlabeled_data)
                projected_unlabelled = feat_unlab @ basis.T @ basis
            # Calculating the cosine_similarity
            # cosine_similarities_matrix = []
            # for proj_sample in projected_unlabelled:
            #     similarities = cosine_similarity([proj_sample], mem_feat.detach().cpu().numpy())  # Similarity to each memory_X sample
            #     cosine_similarities_matrix.append(similarities.flatten())
            # # Convert to a matrix for easier viewing
            # cosine_similarities_matrix = np.array(cosine_similarities_matrix)
            # indices_with_high_similarity = apply_soft_threshold(cosine_similarities_matrix,initial_threshold=0.95)
            # avg_representations = []
            # # Iterate over each set of high similarity indices
            # for indices in indices_with_high_similarity:
            #     # Retrieve the memory features corresponding to these indices
            #     selected_features = mem_feat[indices].detach().cpu().numpy()
            #     avg_representation = np.mean(selected_features, axis=0)
            #     avg_representations.append(avg_representation)
            # # Convert the list of average representations to a NumPy array
            # # print(avg_representations)
            # avg_representations = torch.from_numpy(np.array(avg_representations)).to(device)
                proj_unlab_norm = F.normalize(projected_unlabelled, p=2, dim=1)
            

            # Calculate cosine similarities using matrix multiplication (vectorized)
                memory_torch = torch.cat([torch.from_numpy(memory_X).to(device)])
                mem_feat = model.forward_encoder(memory_torch).detach()
                mem_feat =  mem_feat @ basis.T @ basis
                mem_feat_norm = F.normalize(mem_feat, p=2, dim=1)
                cosine_similarities_matrix = 1-torch.mm(proj_unlab_norm, mem_feat_norm.t()).detach().cpu().numpy()
            # Apply soft threshold to get indices with high similarity
                # if owl_data_labeling == True:
                #     indices_with_high_similarity,valid_indices = apply_soft_threshold(cosine_similarities_matrix, initial_threshold=0,min_threshold=6e-3)
                # else:
                #     indices_with_high_similarity,valid_indices = apply_soft_threshold(cosine_similarities_matrix, initial_threshold=0,min_threshold=upper_thresh*math.exp(-1*temperature2 * labels_ratio))
                    
                indices_with_high_similarity,valid_indices = apply_soft_threshold(cosine_similarities_matrix, initial_threshold=0,min_threshold=upper_thresh*math.exp(-1*temperature2 * labels_ratio))
                # indices_with_high_similarity,valid_indices = apply_soft_threshold(cosine_similarities_matrix, initial_threshold=0,min_threshold=(upper_thresh*math.exp(-10 * labels_ratio)))#*math.exp(-1.5 * labels_ratio))
                filtered_indices, majority_label,max_similar_indices = major_representations(memory_y,indices_with_high_similarity,cosine_similarities_matrix)
                excluded_data = [unlabeled_data[i] for i in range(len(unlabeled_data)) if i not in valid_indices]
                rejected_samples  = len(unlabeled_data) - len(valid_indices)
            # excluded_data_ind = [i for i in range(len(unlabeled_data)) if i not in valid_indices]
            # if len(excluded_data_ind) > 0:
            #     unlabaled_data_cos_sim = cosine_similarities_matrix[excluded_data_ind,:]
            #     print("rejected samples avg cosine dist",np.min(unlabaled_data_cos_sim,axis=1),np.max(unlabaled_data_cos_sim,axis=1))
            #     unlabaled_data_cos_sim = cosine_similarities_matrix[valid_indices,:]
            #     print("accepted avg cosine dist",np.min(unlabaled_data_cos_sim,axis=1),np.max(unlabaled_data_cos_sim,axis=1))
                # exit()
            # print(excluded_data,rejected_samples)
                batch_rejected_samples += rejected_samples
            # Efficiently compute the average representations for each set of high-similarity indices
                start_time = time.time()
            # avg_representations = []
            # for indices in filtered_indices:
            #     selected_features = mem_feat[indices]  # Retrieve the memory features corresponding to these indices
            #     avg_representation = torch.mean(selected_features, dim=0)  # Compute mean along the specified axis
            #     avg_representations.append(avg_representation)
            # avg_representations = torch.stack(avg_representations).to(device)
            # # print(f"Time taken by old method is {time.time() - start_time} and size of indices_with_high_similarity is {len(avg_representations)} and {max(len(inner_list) for inner_list in avg_representations)} ")
            # prob_avg_rep = model.forward_classifier(avg_representations) 
            # max_similar_feat = []
            # for indices in filtered_indices:
            #     selected_features = mem_feat[indices]  # Retrieve the memory features corresponding to these indices
            #     max_similar_feat.append(selected_features)
            # # print(f"Time taken by old method is {time.time() - start_time} and size of indices_with_high_similarity is {len(avg_representations)} and {max(len(inner_list) for inner_list in avg_representations)} ")
            # max_similar_feat = torch.stack(max_similar_feat).to(device)
                max_similar_feat = mem_feat[max_similar_indices]
                prob_avg_rep  = model.forward_classifier(max_similar_feat)
                prob_avg_rep = torch.softmax(prob_avg_rep,dim=1).squeeze()#.to(device)                      
            # Labeled probabs
                pos_pair_probs = prob1[pos_pairs,:]
            # Unlabeled Probabs 
            # print(pos_pair_probs.shape,prob_avg_rep.shape)
            # print(pos_pair_probs,prob_avg_rep)
                pos_pair_probs = torch.concat([pos_pair_probs,prob_avg_rep if prob_avg_rep.dim()>1 else prob_avg_rep.unsqueeze(0)])
                pos_main_probs = torch.concat([prob1[pos_pairs,:],prob1[labeled_len:,:][valid_indices,:]])
            # print("Are there NaNs in prob1?", torch.isnan(prob1).any())
            # print("Are there NaNs in prob_avg_rep?", torch.isnan(prob_avg_rep).any())
                pos_sim = torch.bmm(pos_main_probs.view(pos_main_probs.size(0), 1, -1),  # prob1.size(0) gives batch size
                            pos_pair_probs.view(pos_pair_probs.size(0), -1, 1)  # pos_pair_probs.size(0) gives batch size
                            ).squeeze()
                ones  = torch.ones_like(pos_sim)
                bce_loss = loss_fn(pos_sim,ones)            
            # unlabeled_cosine_dist = cosine_dist[labeled_len:,:]
            # _, pos_idx = torch.topk(unlabeled_cosine_dist,2,dim=1)
            # pos_idx = pos_idx[:,1].cpu().numpy().flatten().tolist()
            # pos_pairs.extend(pos_idx)
            
            # pos_pair_probs = prob2[pos_pairs,:]
            # pos_sim = torch.bmm(prob1.view(prob1.size(0), 1, -1),  # prob1.size(0) gives batch size
            #                 pos_pair_probs.view(pos_pair_probs.size(0), -1, 1)  # pos_pair_probs.size(0) gives batch size
            #                 ).squeeze()
            # if (pos_sim > 1).any():
            #     print("pos_sim contains values greater than 1.")
            #     # Count how many values are greater than 1
            #     count = (pos_sim > 1).sum().item()
            #     print(f"Number of values greater than 1: {count}")
            # else:
            #     print("All values in pos_sim are within the valid range (<= 1).")
            # print("prob1 min and max:", prob1.min().item(), prob1.max().item())
            # print("prob_avg_rep min and max:", prob_avg_rep.min().item(), prob_avg_rep.max().item())
            # print("pos_pair_probs min and max:", pos_pair_probs.min().item(), pos_pair_probs.max().item())
            # ones  = torch.ones_like(pos_sim)
            # bce_loss = loss_fn(pos_sim,ones)
            # ce_loss = ce(output1[:labeled_len],lab_y[:labeled_len])
            # print(lab_y[:labeled_len])            
            ce_loss_fucntion = nn.CrossEntropyLoss()
            ce_loss = ce_loss_fucntion(output1[:labeled_len],lab_y[:labeled_len])
            entropy_loss = entropy(torch.mean(prob1[labeled_len:],0))
            # entropy_loss = F.kl_div(torch.mean(prob1[labeled_len:],0).log(), (torch.tensor([0.1,0.9]).log()).to(device), log_target=True)
            total_loss = -entropy_loss*0+ ce_loss + bce_loss
            # controller = math.exp(-1.5 * labels_ratio)
            # if owl_data_labeling == False:
            #     total_loss = -entropy_loss*0+ ce_loss + controller*bce_loss
            # else:
            #      total_loss = -entropy_loss*0+ ce_loss   
            #---------------------------------------------------------------------------------------------------------
            distil_loss = 0
            distil_loss = torch.as_tensor(distil_loss).to(device)
            opt.zero_grad()
            if task_id > 0:
                if bool_gpm:
                    total_loss.backward()
                    bn_counter = 0
                    for k, (m,params) in enumerate(model.named_parameters()):
                        # print(params.grad.data.shape)
                        # print(m)
                        # print("featmat",feature_mat[k].shape)
                        if 'bn' not in m:
                            k -= bn_counter
                            # print(params.grad.data.shape)
                            # print("featmat",feature_mat[k].shape)                            
                            sz =  params.grad.data.size(0)
                            # print("sz=",sz)
                            projection =  torch.mul(torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[k]).view(params.size()),1)
                            params.grad.data = torch.mul((params.grad.data  - projection ), (1))
                        else:
                            bn_counter += 1    
            else:       
                total_loss.backward()

            opt.step() 
            # teacher_model.load_state_dict(model.state_dict(), strict=False)
            # gradient_rejection(model=model)
            # if gradient_rejection.early_stop:
            #     torch.save(model.state_dict(), check_point_file_name_norm)
            train_losses.append(total_loss.item())
            ce_losses.append(ce_loss.item())
            bce_losses.append(bce_loss.item())

            y_pred = y_pred.detach().cpu().numpy()
            lab_y = lab_y.detach().cpu().numpy()
            
            # lr_precision, lr_recall, _ = precision_recall_curve(lab_y, y_pred,pos_label=1)
            # lr_auc_outlier =  auc(lr_recall, lr_precision)
            
        

            # lr_precision, lr_recall, _ = precision_recall_curve(lab_y, [1-x for x in y_pred],pos_label=0)
            # lr_auc_inliers =  auc(lr_recall, lr_precision)   
            # prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f} - PR-AUC(inliers): {:.2f} - PR_auc(outlier)_curve {:.3f}'.format(
            #      total_loss.item(), sup_loss.item(), distil_loss.item(), lr_auc_inliers,lr_auc_outlier ))
            # r_auc = roc_auc_score(lab_y, y_pred)
            # prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f}'.format(
            #      total_loss, sup_loss, distil_loss))
            prog_bar.set_description('loss: {:.5f} - entropy_loss : {:.5f} bce_loss: {:.5f} ce_loss:{:.5f}'.format(
                 total_loss.item(),entropy_loss.item(),bce_loss.item(),ce_loss.item()))

            
        
        model.eval() # prep model for evaluation
        val_pred,val_gt = [],[]
        for data, target in valid_loader:
            # pred = torch.argmax(model(data.to(device)),dim=1).reshape(target.shape)
            pred = model(data.to(device))[:,1].reshape(target.shape)
            y_pred = pred.detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_gt.extend(target.detach().cpu().numpy().tolist())
        lr_precision, lr_recall, _ = precision_recall_curve(val_gt, [x for x in val_pred], pos_label=1.)
        lr_auc_minority =  auc(lr_recall, lr_precision)
        # lr_precision, lr_recall, _ = precision_recall_curve(val_gt, val_pred, pos_label=1.)
        # lr_auc_majority=  auc(lr_recall, lr_precision)
        lr_auc = lr_auc_minority#[lr_auc_minority,lr_auc_majority]
        # lr_auc = f1_score(val_gt,val_pred)
            # calculate the loss
            # loss = loss_fn(pred, target.to(device))
            # record validation loss
            # valid_losses.append(loss.item())
        # valid_losses.append(np.nan_to_num(lr_auc))
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        global_train_loss.append(train_loss)
        global_ce_loss.append(np.average(ce_losses))
        global_bce_loss.append(np.average(bce_losses))
        # valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        # avg_valid_losses.append(valid_loss)
        epoch_len = len(str(epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'PR-AUC (I): {lr_auc:.5f}')
        
        print(print_msg)
        avg_rejected_samples.append(batch_rejected_samples)
    #     plt.figure()
    #     window_size = 5  # Adjust this value as needed
    #     weights = np.ones(window_size) / window_size 
    #     # print(len(train_losses),len(ce_losses))
    #     # exit()
    #     smoothed_train_losses = np.convolve(train_losses, weights, mode='valid')
    #     smoothed_ce_losses = np.convolve(ce_losses, weights, mode='valid')
    #     smoothed_bce_losses = np.convolve(bce_losses, weights, mode='valid')
    #     plt.plot(range(1, len(smoothed_train_losses) + 1), smoothed_train_losses, marker='.', label=f'Train Loss (Task {task_id + 1}),labeled ratio{labels_ratio*100}')
    #     plt.plot(range(1, len(smoothed_ce_losses) + 1), smoothed_ce_losses, marker='.', label=f'CE Loss (Task {task_id + 1}),labeled ratio{labels_ratio*100}')
    #     plt.plot(range(1, len(smoothed_bce_losses) + 1), smoothed_bce_losses, marker='.', label=f'BCE Loss (Task {task_id + 1}),labeled ratio{labels_ratio*100}')

    #     # plt.plot(range(1, len(train_losses) + 1), train_losses, marker='.', label=f'Train Loss (Task {task_id + 1})')
    #     # plt.plot(range(1, len(ce_losses) + 1), ce_losses, marker='s', label=f'CE Loss (Task {task_id + 1})')
    #     # plt.plot(range(1, len(bce_losses) + 1), bce_losses, marker='*', label=f'BCE Loss (Task {task_id + 1})')
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Loss")
    #     plt.ylim(0, 1)
    #     plt.title(f"Loss Curve for Task {task_id + 1}")
    #     plt.legend()
    #     plt.grid(True)
    # # directory_path = "./plots/loss/new_method/"+str(label)+"/labelratio/"+str(int(labels_ratio*100))+"/"+str(seed)+"/task"+task_id+"/"
    #     directory_path = f"./plots/loss/new_method/{label}/labelratio/{int(labels_ratio*100)}/{seed}/task{task_id}/"
    #     if not os.path.exists(directory_path):
    #             os.makedirs(directory_path)
                
    #     plt.savefig(os.path.join(directory_path, "loss_curve.pdf")) 
    #     plt.close()
        # clear lists to track next epoch
        train_losses = []
        ce_losses = []
        bce_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(lr_auc, model)
        if early_stopping.counter <1:
            scheduler.step()

        if early_stopping.early_stop:
            print("Early stopping")
            break
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(check_point_file_name))
    teacher_model.load_state_dict(torch.load(check_point_file_name))
    

    # temp_x,temp_y,temp_yname = X[labeled_indicies,:],y[labeled_indicies],y_classname[labeled_indicies]
    
    # temp_x,temp_y,temp_yname = X[unlabeled_indicies,:],y[unlabeled_indicies],y_classname[unlabeled_indicies]
    

    # temp_x,temp_y,temp_yname = labeled_X,labeled_y,labeled_y_classname
    # if task_id > 0 and bool_reorganize_memory:
    #     mem_start_time = time.time()
    #     if str(mem_strat) == "replace":
            
    #         tasks[0] = temp_x,temp_y,temp_yname
    #         lab_samples_in_memory = split_a_task(tasks,lab_samp_in_mem_ratio)
    #         tasks[0] = temp_x[lab_samples_in_memory[0],:],temp_y[lab_samples_in_memory[0]],temp_yname[lab_samples_in_memory[0]]
    #         initialize_buffermemory(tasks=tasks,mem_size=memory_size)
    #     elif str(mem_strat) == "equal":
            
    #         memory_X, memory_y, memory_y_name = memory_update_equal_allocation2(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.15,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)
    #     else:
            
    #         memory_X, memory_y, memory_y_name = memory_update_equal_allocation(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.85,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)

    #     mem_finish_time = time.time()
    #     memory_population_time += mem_finish_time-mem_start_time

    # mat_list = []    
    temp_x,temp_y,temp_yname = X[labeled_indicies,:],y[labeled_indicies],y_classname[labeled_indicies]
    # mat_list = get_representation_matrix (model, device, temp_x, temp_y)
    for i in model.act.keys():
        if ds in ['api_graph', 'androzoo']:
            threshold.append(np.random.choice([0.1,0.10,0.10,0.10,0.1,0.1,0.1],1)[0])
        else:
            threshold.append(np.random.choice([0.95,0.99,0.99,0.98,0.99,0.99,0.99],1)[0])
        
    if bool_gpm:
        mat_list = get_representation_matrix (model, device, temp_x, temp_y,rand_samples=no_of_rand_samples)
        feature_list = update_GPM(model, mat_list, threshold, feature_list)
    else:
        feature_list = []

    # grad_norm_dict[task_id] = grad_norm_list   
    # print(grad_norm_dict)
    if os.path.exists(check_point_file_name):
        os.remove(check_point_file_name)
    if os.path.exists(check_point_file_name_norm):
        os.remove(check_point_file_name_norm) 

    task_change_indices.append(len(global_train_loss))    

    # print(f'Buffer memory size for task {task_id}: {memory_X.shape}')
    print(f"Total Rejected Samples per task : {np.sum(avg_rejected_samples)}")
    print(f"Average Rejected Samples per task : {np.mean(avg_rejected_samples)}")
    if owl_data_labeling == False:
        avg_rej_samples_per_seen_task.append(float(np.mean(avg_rejected_samples)))
    else:
        avg_rej_samples_per_unseen_task.append(float(np.mean(avg_rejected_samples)))
    return feature_list

def get_whole_test_set():
    global test_x,test_y,task_num,teacher_model,model,task_order,X_test,y_test
    test_x,test_y = [],[]
    task_order1 = task_order
    # random.shuffle(task_order1)
    for task_id,task in enumerate(task_order1):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        # print("loading task:",task_id)     
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=bool_encode_benign,bool_encode_anomaly=bool_encode_anomaly,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=load_whole_train_data)
        test_x.extend([X_test])
        test_y.extend([y_test])

    X_test,y_test = np.concatenate( test_x, axis=0 ),np.concatenate( test_y, axis=0 )

def taskwise_lazytrain():
    global test_x,test_y,task_num,task_order,auc_result,val_x_all_tasks,val_y_all_tasks
    global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised
    global owl_self_labelled_count_class_0, owl_self_labelled_count_class_1
    global owl_analyst_labelled_count_class_0, owl_analyst_labelled_count_class_1
    global avg_CI, CI_list
    global truth_agreement_fraction_0, truth_agreement_fraction_1
    # global consecutive_otdd

    # random.shuffle(task_order)
    print("task order",task_order)
    threshold  = []
    feature_list_student1,feature_list_student2,feature_list_student_supervised =[],[],[]

    train_order = task_order[:training_cutoff]
    test_order = task_order[training_cutoff:]
    print(f'\nTraining on first {training_cutoff} tasks...')
    for task_id,task in enumerate(train_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("\nloading task:",task_id)     
        input_shape,tasks,X_test,y_test,X_val,y_val = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=bool_encode_benign,bool_encode_anomaly=bool_encode_anomaly,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=load_whole_train_data)
        # print(tasks)

        

        val_x_all_tasks.extend([X_val])
        val_y_all_tasks.extend([y_val])
        # val_x_all_tasks,val_y_all_tasks = [X_val],[y_val]
        print("validation dataset size",X_val.shape)

        test_x.extend([X_test])
        test_y.extend([y_test])
        print("Training task:",task_id)
        task_num = task_id
        if task_num == int(0):
            initialize_buffermemory(tasks=tasks,mem_size=memory_size)
        if mlps == 1:
            feature_list_student1 =train("student1",tasks,task_class_ids,task_id,feature_list_student1,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),True,False)
                    
        elif mlps == 2:
            feature_list_student1 =train("student1",tasks,task_class_ids,task_id,feature_list_student1,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),True,False)
            feature_list_student2 =train("student2",tasks,task_class_ids,task_id,feature_list_student2,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),False,False)
            
        
        else:
            feature_list_student1 =train("student1",tasks,task_class_ids,task_id,feature_list_student1,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),True,False)
            feature_list_student2 =train("student2",tasks,task_class_ids,task_id,feature_list_student2,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),False,False)
            feature_list_student_supervised =train("student_supervised",tasks,task_class_ids,task_id,feature_list_student_supervised,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),False,False)
        
        # teacher_model.load_state_dict(model.state_dict())
    benchmar_results = testing(training_cutoff=training_cutoff,type="unseen" ,task_id=task_id ,seen_data=False,run_order="per_task",curr_train_task=task_id)   
    # testing(training_cutoff=training_cutoff, seen_data=True) 
    # testing(training_cutoff=training_cutoff, seen_data=False)    
    print(f'\nOpen world setting training from task {training_cutoff} onwards...')
    truth_agreement_fraction_0 = max(truth_agreement_fraction_0, 0.5)
    truth_agreement_fraction_1 = max(truth_agreement_fraction_1, 0.5)
    print(f'Agreement fraction: class 0 = {truth_agreement_fraction_0}, class 1 = {truth_agreement_fraction_1}')
    
    avg_CI = np.mean(CI_list)
    print(f'Average CI over training tasks (% of class 0 samples)= {avg_CI}')

    for task_id,task in enumerate(test_order,start=training_cutoff):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("\nloading task:",task_id)     
        input_shape,tasks,X_test,y_test,X_val,y_val = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=bool_encode_benign,bool_encode_anomaly=bool_encode_anomaly,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=load_whole_train_data)
        
        
        val_x_all_tasks.extend([X_val])
        val_y_all_tasks.extend([y_val])
        # val_x_all_tasks.extend([X_val])
        # val_y_all_tasks.extend([y_val])
        # val_x_all_tasks,val_y_all_tasks = [X_val],[y_val]
        print("validation dataset size",X_val.shape)

        test_x.extend([X_test])
        test_y.extend([y_test])
        print("Training task:",task_id)
        task_num = task_id
        if task_num == int(0):
            initialize_buffermemory(tasks=tasks,mem_size=memory_size)

        if mlps == 1:
            feature_list_student1 =train("student1",tasks,task_class_ids,task_id,feature_list_student1,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),True,True)
                    
        elif mlps == 2:
            feature_list_student1 =train("student1",tasks,task_class_ids,task_id,feature_list_student1,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),True,True)
            feature_list_student2 =train("student2",tasks,task_class_ids,task_id,feature_list_student2,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),False,True)
            
        
        else:
            feature_list_student1 =train("student1",tasks,task_class_ids,task_id,feature_list_student1,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),True,True)
            feature_list_student2 =train("student2",tasks,task_class_ids,task_id,feature_list_student2,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),False,True)
           #feature_list_student_supervised =train("student_supervised",tasks,task_class_ids,task_id,feature_list_student_supervised,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),False,False)
        # teacher_model.load_state_dict(model.state_dict())
        # teacher_model.load_state_dict(model.state_dict())
    # testing(training_cutoff=training_cutoff, seen_data=True) 
    # testing(training_cutoff=training_cutoff, seen_data=False)  
    test_set_results = []
    with open(temp_filename, 'w') as fp:
        test_set_results.extend([testing(training_cutoff=training_cutoff,type="seen" ,task_id=task_id ,seen_data=True),testing(training_cutoff=training_cutoff, type="unseen" ,task_id=task_id ,seen_data=False),str(owl_self_labelled_count_class_0), str(owl_self_labelled_count_class_1),str(owl_analyst_labelled_count_class_0), str(owl_analyst_labelled_count_class_1),testing(training_cutoff=len(task_order), type="all" ,task_id=task_id ,seen_data=True),sum(avg_rej_samples_per_seen_task)/len(avg_rej_samples_per_seen_task),sum(avg_rej_samples_per_unseen_task)/len(avg_rej_samples_per_unseen_task) ])
        test_set_results.extend([benchmar_results[2],benchmar_results[3]])
        auc_result[str(args.seed)] = test_set_results
        json.dump(auc_result, fp)

    # print('\nOTDD values for consecutive tasks:')
    # for i, val in enumerate(consecutive_otdd):
    #     print(f'Task ({i},{i + 1}): {val}')
    # print()

    plt.figure()
    window_size = 5  # Adjust this value as needed
    weights = np.ones(window_size) / window_size 
    # print(len(train_losses),len(ce_losses))
    # exit()
    smoothed_train_losses = np.convolve(global_train_loss, weights, mode='valid')
    smoothed_ce_losses = np.convolve(global_ce_loss, weights, mode='valid')
    smoothed_bce_losses = np.convolve(global_bce_loss, weights, mode='valid')
    plt.plot(range(1, len(smoothed_train_losses) + 1), smoothed_train_losses, marker='.', label=f'Train Loss all Task ,labeled ratio {labels_ratio*100}')
    plt.plot(range(1, len(smoothed_ce_losses) + 1), smoothed_ce_losses, marker='.', label=f'CE Loss all Tasks,labeled ratio {labels_ratio*100}')
    plt.plot(range(1, len(smoothed_bce_losses) + 1), smoothed_bce_losses, marker='.', label=f'BCE Loss all Tasks,labeled ratio {labels_ratio*100}')

    for index in task_change_indices:
        plt.axvline(x=index, color='red', linestyle='--', linewidth=1)

# Add labels for each task segment
    for i, index in enumerate(task_change_indices):
        plt.text(index + 0.5, plt.ylim()[1] * 0.9, f'T{i + 1}', color='black',fontsize=5)

    # plt.plot(range(1, len(train_losses) + 1), train_losses, marker='.', label=f'Train Loss (Task {task_id + 1})')
    # plt.plot(range(1, len(ce_losses) + 1), ce_losses, marker='s', label=f'CE Loss (Task {task_id + 1})')
    # plt.plot(range(1, len(bce_losses) + 1), bce_losses, marker='*', label=f'BCE Loss (Task {task_id + 1})')
    plt.xlabel("Loss values idx")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.title(f"Loss Curve for all Tasks, dataset: {label}, seed: {seed}, labeled ratio: {labels_ratio}")
    plt.legend()
    plt.grid(True)
    # directory_path = "./plots/loss/new_method/"+str(label)+"/labelratio/"+str(int(labels_ratio*100))+"/"+str(seed)+"/task"+task_id+"/"
    directory_path = f"./plots/loss/new_method/{label}/"
    if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
    plt.savefig(os.path.join(directory_path, f"loss_curve_labeled ratio_{labels_ratio}_seed:{seed}.pdf")) 
    plt.close()

    print('************** OWL labelling stats ****************')
    print(f'Total number of self-labelled samples = {owl_self_labelled_count_class_0 + owl_self_labelled_count_class_1}')
    print(f'Class 0 count = {owl_self_labelled_count_class_0}')
    print(f'Class 1 count = {owl_self_labelled_count_class_1}')

    print(f'Total number of analyst-labelled samples = {owl_analyst_labelled_count_class_0 + owl_analyst_labelled_count_class_1}')
    print(f'Class 0 count = {owl_analyst_labelled_count_class_0}')
    print(f'Class 1 count = {owl_analyst_labelled_count_class_1}')

    print(f'Average rejected samples={avg_rej_samples_per_seen_task},{sum(avg_rej_samples_per_seen_task)/len(avg_rej_samples_per_seen_task)}')
    print(f'Average rejected samples={avg_rej_samples_per_unseen_task},{sum(avg_rej_samples_per_unseen_task)/len(avg_rej_samples_per_unseen_task)}')
    print(f"Benchmarkin results: Benign{benchmar_results[2]} and Attack{benchmar_results[3]}")


        
def testing(training_cutoff, task_id,type,run_order="",curr_train_task = 0,seen_data=False):

    dataset_loadtime=0
    global teacher_model1,teacher_model2,teacher_supervised
    global student_model1,student_model2,student_supervised

    

    
    if mlps == 1:
        models = [student_model1]
    elif mlps == 2:
        models = [student_model1,student_model2]
    else:
        models = [student_model1,student_model2,student_supervised]
    
    # ensemble_main.eval()
    
    task_CI_pnt = []
    test_CI_pnt =[]
    prauc_in_pnt = []
    prauc_out_pnt = []
    en_prauc_in_pnt = []
    en_prauc_out_pnt = []
    metric_results=[]

    if seen_data:
        testing_tasks = task_order[:training_cutoff]
        start_id = 0
    else:
        testing_tasks  = task_order[training_cutoff:]
        start_id = training_cutoff
        

    # for task_id,task in enumerate(task_order):#[training_cutoff:], start = training_cutoff):
    # for task_id,task in enumerate(task_order[training_cutoff:], start = training_cutoff):
    for task_id,task in enumerate(testing_tasks, start = start_id):
        
        
        task_class_ids = []
        task_minorityclass_ids = []
        
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        start = time.time()
        # input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=0,bool_encode_anomaly=1,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=True)
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=bool_encode_benign,bool_encode_anomaly=bool_encode_anomaly,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=False)
        
        if seen_data:
            features,target_label = X_test,y_test
        else:
            features,target_label = X_test,y_test
            # features,target_label = tasks[0][0],tasks[0][1]
        # print(f'testing on task {task_id}: {features.shape}')
        
        dataset_loadtime += time.time()-start
        
        
        # valid_loader = torch.utils.data.DataLoader(dataset(tasks[0][0],tasks[0][1]),
        #                                        batch_size=batch_size,
        #                                     #    sampler=valid_sampler,
        #                                        num_workers=0)     
        valid_loader = torch.utils.data.DataLoader(dataset(features,target_label),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=0)
        
        val_pred, en_val_pred, val_actual = [],[], []
        for data, target in valid_loader:
            class_probs = [] 
            with torch.no_grad():
                for model in models:
                    outputs = torch.softmax(model(data.to(device)), dim=1)
                    class_probs.append(outputs)

            pred = torch.stack(class_probs).mean(dim=0)[:,1].reshape(target.shape)
            
            # pred = (model(data.to(device))[:,1]).reshape(target.shape)
            en_pred = pred
            # en_pred = ensemble_main(data.to(device)).reshape(target.shape)
            y_pred = pred.detach().cpu().numpy().tolist()
            en_y_pred = en_pred.detach().cpu().numpy().tolist()

            val_pred.extend(y_pred)
            en_val_pred.extend(en_y_pred)
            val_actual.extend(target.detach().cpu().numpy().tolist())         

        
        train_y = tasks[0][1]

        # print(f'test set size:= {len(val_actual)} with {len([i for i in val_actual if int(i) == 1])} attacks')
        # print(f'train set size:= {len(tasks[0][1])} with {len([i for i in tasks[0][1] if int(i) ==1])} attacks')
        
        # train_CI = len([i for i in tasks[0][1] if round(i) == 1])/(len(tasks[0][1])-len([i for i in tasks[0][1] if round(i) == 1]))
        # test_CI = len([i for i in val_actual if round(i) == 1])/(len(val_actual)-len([i for i in val_actual if round(i) == 1]))
        # print(f'test CI: {test_CI}')
        # print(f'train CI: {train_CI}')
        
        precision, recall, thresholds = precision_recall_curve(val_actual, val_pred,pos_label=1.0)
        auc_precision_recall_1 = auc(recall, precision)
        precision, recall, thresholds = precision_recall_curve(val_actual, en_val_pred)
        en_auc_precision_recall_1 = auc(recall, precision)

        precision, recall, thresholds = precision_recall_curve(val_actual, [1-val for val in val_pred], pos_label=0.)
        auc_precision_recall_0 = auc(recall, precision)
        precision, recall, thresholds = precision_recall_curve(val_actual, [1-val for val in en_val_pred], pos_label=0.)
        en_auc_precision_recall_0 = auc(recall, precision)

        #when number of 1s > 0s then the 1 is the inliers and0 is the outliers
        # auc_precision_recall_in = auc_precision_recall_0 if test_CI < 1 else auc_precision_recall_1
        # auc_precision_recall_out = auc_precision_recall_1 if test_CI < 1 else auc_precision_recall_0

        # en_auc_precision_recall_in = en_auc_precision_recall_0 if test_CI < 1 else en_auc_precision_recall_1
        # en_auc_precision_recall_out = en_auc_precision_recall_1 if test_CI < 1 else en_auc_precision_recall_0

        # task_CI_pnt.append(train_CI)
        # test_CI_pnt.append(test_CI)
        # prauc_in_pnt.append(auc_precision_recall_in)
        # prauc_out_pnt.append(auc_precision_recall_out)
        # en_prauc_in_pnt.append(en_auc_precision_recall_in)
        # en_prauc_out_pnt.append(en_auc_precision_recall_out)
        prauc_in_pnt.append(auc_precision_recall_0)
        prauc_out_pnt.append(auc_precision_recall_1)
        en_prauc_in_pnt.append(en_auc_precision_recall_0)
        en_prauc_out_pnt.append(en_auc_precision_recall_1)
        metric_results.append(compute_results_new(y_test=val_actual, lr_probs=np.array(val_pred),name=ds,seed=seed,task_id=task_id,type=type,run_order=run_order,curr_train_task =curr_train_task))
        # print(f'prauc inliers: {auc_precision_recall_in}')        
        # print(f'prauc outliers: {auc_precision_recall_out}')                     
        # print('')
    
    N = len(testing_tasks) #number of test tasks
    prauc_in_aut  = 0
    prauc_out_aut = 0
    if N<2:
        print('not printing AUT values since it requires atleast 2 test tasks')
        return [prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N]
    for i in range(N-1):
        prauc_in_aut+= (prauc_in_pnt[i]+prauc_in_pnt[i+1])/(2)
        prauc_out_aut+=(prauc_out_pnt[i]+prauc_out_pnt[i+1])/(2)
    prauc_in_aut  = prauc_in_aut/(N-1)
    prauc_out_aut = prauc_out_aut/(N-1)
    
    print(f'AUT(prauc inliers,{N}) := {prauc_in_aut}')
    print(f'AUT(prauc outliers,{N}) := {prauc_out_aut}')

    # print('\npnt table:')
    pnt_table = [
        # ['task_CI']+ task_CI_pnt, 
        # ['test_CI'] + test_CI_pnt,
        ['prauc Benign traffic'] + prauc_in_pnt, 
        ['prauc Attack traffic'] + prauc_out_pnt
    ]
    # print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
    
    # print('\npnt table for Ensemble models:')
    # pnt_table = [
    #     # ['task_CI']+ task_CI_pnt, 
    #     # ['test_CI'] + test_CI_pnt,
    #     ['prauc Benign traffic'] + en_prauc_in_pnt, 
    #     ['prauc Attack traffic'] + en_prauc_out_pnt
    # ]
    # print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
    print(f'dataset loading time: {dataset_loadtime}s\n')
    
    # print('Here json dump of the data for easy unparsing')
    # print(f'#pnt_table#{json.dumps(pnt_table)}#end_pnt_table#')
    # print(f'#train_order#{json.dumps(train_order)}#end_train_order#')
    return [prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N]
  

def evaluate_on_sub_testset(test_x,test_y):
    test_x,test_y = np.concatenate( test_x, axis=0 ),np.concatenate( test_y, axis=0 )
    model.eval()
    print("computing the results")
    offset = 25000
    for idx in range(0,test_x.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(test_x[idx1:idx2,:].astype(float)).to(device)
        if image_resolution is not None:
                    X_test1 = X_test1.reshape(image_resolution)
        temp = model(X_test1.float()).detach().cpu().numpy()
        if idx1==0:
            yhat = temp
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
    compute_results(test_y,yhat)
    model.train()

def evaluate_on_testset():
    
    global X_test,y_test
    if pth_testset is not None:
        X_test,y_test = load_teset(pth_testset,testset_class_ids,label)
    yhat = None    
    model.eval()
    print("computing the results")
    offset = 25000000
    # offset = 25000
    for idx in range(0,X_test.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float)).to(device)
        if image_resolution is not None:
                    X_test1 = X_test1.reshape(image_resolution)
        # temp = torch.argmax(model(X_test1.float()),dim=1).detach().cpu().numpy()
        temp = (model(X_test1.float())[:,1]).detach().cpu().numpy()
        if idx1==0:
            yhat = temp
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
    return compute_results(y_test,yhat)
    # print("test sample counters are",Counter(y_test))

import numpy as np

def select_random_indices_for_classes(labels, class_label_1, class_label_2, num_samples_per_class1=1000,num_samples_per_class2=100):
    indices_class_label_1 = np.where(labels == class_label_1)[0]
    indices_class_label_2 = np.where(labels == class_label_2)[0]
    print("classs 0",len(indices_class_label_1))
    print("classs 0",len(indices_class_label_2))

    random_indices_class_label_1 = np.random.choice(indices_class_label_1, num_samples_per_class1, replace=False).tolist()
    random_indices_class_label_2 = np.random.choice(indices_class_label_2, num_samples_per_class2, replace=False).tolist()
    random_indices_class_label_1.extend(random_indices_class_label_2)
    return random_indices_class_label_1




def tsne_visualize(seed,labels_ratio=0.1,batch_minority=0.5,rand_samples=100,ppt=50):
    global X_test,y_test
    test_embeddings = torch.zeros((0,10), dtype=torch.float32)
    if pth_testset is not None:
        X_test,y_test = load_teset(pth_testset,testset_class_ids,label)
    yhat = None    
    model.eval()
    indices = select_random_indices_for_classes(y_test,0,1,10000,100000)
    print(indices)
    X_test,y_test = X_test[indices],y_test[indices]
    print("computing the results")
    offset = 25000
    for idx in range(0,X_test.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float)).to(device)
        if image_resolution is not None:
                    X_test1 = X_test1.reshape(image_resolution)
        temp = model(X_test1.float()).detach().cpu().numpy()
        embeddings = model.act['hidden6']
        if idx1==0:
            yhat = temp
            
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
        test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)    
    test_embeddings = np.array(test_embeddings)
    dir_struct = {0:"tsne",1:"caring",2:str(label)}    
    dir_struct[3 ]= "_lab_ratio_"+str(labels_ratio)+"_minorty_"+str(batch_minority)+"_rand_samp_"+str(rand_samples)+"_seed"+str(seed)
    for pt in [5,10,25,50,100,150,200,350,500,750,1000,1500,2000,2500,5000,10000,20000,40000,50000]:
        plot_tsne(y_test,yhat,test_embeddings,dir_struct,pt)


def start_execution(dataset_name,l_rate,w_decay):
    global input_shape,tasks,X_test,y_test,test_x,test_y,val_x_all_tasks,val_y_all_tasks
    # print(f"{analyst_labels} is the no of labels for the security analysts.---a-sdf-asdf;jasdfljkaksdf")
    start_time=time.time()
    load_metadata(dataset_name,l_rate,w_decay)
    # load_model_metadata()
    # print(model)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("number of parameters is",pytorch_total_params)
    print(f'Is lazy training: {is_lazy_training}')
    if is_lazy_training:
        test_x,test_y = [],[]
        val_x_all_tasks,val_y_all_tasks = [],[]
        # get_whole_test_set()
        taskwise_lazytrain()
        # plot_grdient_norm_line_graph()
        X_test,y_test = np.concatenate( test_x, axis=0 ),np.concatenate( test_y, axis=0 )
        

    else:
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,class_ids,minorityclass_ids,tasks_list,task2_list,task_order,bool_encode_benign=False,bool_encode_anomaly=True,label=label,bool_create_tasks_avalanche=False)
        initialize_buffermemory(tasks=tasks,mem_size=memory_size)
        print('Total no.of tasks', len(tasks))
        # update_buffermemory_counter(memorysamples=memory_y_name)
        # update_mem_samples_indexdict(memorysamples=memory_y_name)
        train(tasks=tasks)
    print("Total execution time is--- %s seconds ---" % (time.time() - start_time))
    print("Total memory population time is--- %s seconds ---" % (memory_population_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--ds', type=str, default="ids17", metavar='S',help='dataset name')
    parser.add_argument('--gpu', type=int, default=1, metavar='S',help='gpu id (default: 0)')
    parser.add_argument('--filename', type=str,default="temp", metavar='S',help='json file name')
    parser.add_argument('--b_m', type=float, default=0.2, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='S',help='batch memory ratio(default: 0.001)')
    parser.add_argument('--wd', type=float, default= 1e-3, metavar='S',help='batch memory ratio(default: 0.01)')
    parser.add_argument('--label_ratio', type=float, default=0.2, metavar='S',help='labeled ratio (default: 0.1)')
    parser.add_argument('--nps', type=int, metavar='S',default=10000,help='number of projection samples(default: 100)')
    parser.add_argument('--bma', type=float, metavar='S',default=0.9,help='batch minority allocation(default: 0.8)')
    parser.add_argument('--alpha', type=float, metavar='S',default=9,help='distill loss multiplier(default: 9)')
    parser.add_argument('--lab_samp_in_mem_ratio', type=float, metavar='S',default=1.0,help='Percentage of labeled samples to store in memory(default: 1.0)')
    parser.add_argument('--bool_gpm', type=str, metavar='S',default="True",help='Enables gradient projections(default: True)')
    parser.add_argument('--mem_strat', type=str, metavar='S',default="equal",help='Buffer memory strategy(default: full initialization)')
    parser.add_argument('--training_cutoff', type=int, default=3, metavar='S',help='train the model for first n tasks and test for time decay on the rest')
    parser.add_argument('--bool_closs', type=str, metavar='S',default="False",help='Enables using contrastive loss(default: False)')
    parser.add_argument('--mlps', type=int, metavar='S',default=1,help='Number of learners (MLPs)default: 1)')
    parser.add_argument('--cos_dist', type=float, metavar='S',default=0.15,help='cosine distance for OWL(default: 0.3)')
    parser.add_argument('--mode_val', type=int, metavar='S',default=80,help='Mode value for OWL (default: 99)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N', help='number of training epochs/task (default: 10)')
    parser.add_argument('--beta', type=float, metavar='S',default=0.1,help='hyperparameter for accumulation of agreement fraction')
    parser.add_argument('--upper_thresh', type=float, metavar='S',default=0.2,help='Sets the upper threshold for while aligning similarity (default:0.2)')
    parser.add_argument('--temp', type=float, metavar='S',default=20,help='temperature (default:20)')
    parser.add_argument('--temp2', type=float, metavar='S',default=1.5,help='temperature2 (default:1.5)')
    parser.add_argument('--analyst_labels', type=int, metavar='S',default=100,help='No of labels from analysts (default:50)')

    args = parser.parse_args()
    set_seed(args.seed)
    get_gpu(args.gpu)
    print("seed is",args.seed)
    global analyst_labels,labels_ratio,no_of_rand_samples,l_rate,w_decay,batch_minority_allocation,b_m,alpha,lab_samp_in_mem_ratio,bool_gpm,mem_strat,temp_filename,auc_result,seed,bool_closs,mlps,training_cutoff, epochs, ds, beta,cos_dist_ip, mode_value,upper_thresh,temperature,temperature2
    epochs = args.n_epochs
    b_m = float(args.b_m)
    labels_ratio=float(args.label_ratio)
    no_of_rand_samples = int(args.nps)
    batch_minority_alloc = float(args.bma)
    alpha = float(args.alpha)
    analyst_labels = int(args.analyst_labels)
    l_rate = float(args.lr) 
    w_decay = float(args.wd)
    lab_samp_in_mem_ratio = float(args.lab_samp_in_mem_ratio)
    bool_gpm = eval(args.bool_gpm)
    bool_closs = eval(args.bool_closs)
    mem_strat = str(args.mem_strat)
    mlps = int(args.mlps)
    mode_value = int(args.mode_val)
    cos_dist_ip = float(args.cos_dist)
    training_cutoff = int(args.training_cutoff)
    seed = args.seed
    temperature = float(args.temp)
    temperature2 = float(args.temp2)

    ppt = 25
    ds = args.ds
    beta = args.beta
    upper_thresh = float(args.upper_thresh)
    print("{:<20}  {:<20}".format('Argument','Value'))
    print("*"*80)
    for arg in vars(args):
        print("{:<20}  {:<20}".format(arg, getattr(args, arg)))
    print("*"*80)    
    auc_result= {}
    temp_filename = str(args.filename)
    start_execution(args.ds,l_rate,w_decay)     
    print("*"*80)

