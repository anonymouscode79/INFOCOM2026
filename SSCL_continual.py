

from turtle import st
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torch.optim.lr_scheduler import StepLR
from scipy.spatial.distance import cdist
from scipy import stats
# from pytorchtools import EarlyStopping
import subprocess
import os
import numpy as np
import pandas as pd
import math
from baselines.active_learning.models import CAE, MLPClassifier, SimpleEncClassifier
from baselines.active_learning.sample_selector import LocalPseudoLossSelector, OODSelector
from baselines.active_learning.train import evaluate_model, evaluate_model_cl, train_classifier, train_encoder,init_weights
from baselines.continual_learning.model_loader import MODELNAME, MalwareDetectionModel, Model_class
from avalanche.benchmarks import tensors_benchmark
from utils.customdataloader import load_dataset,Tempdataset,compute_total_minority_testsamples,get_inputshape,load_teset
from utils.buffermemory import memory_update,retrieve_replaysamples,memory_update_equal_allocation,memory_update_equal_allocation2,memory_update_equal_allocation3
from utils.metrics import compute_results_new
from utils.utils import log,create_directories,trigger_logging,set_seed,get_gpu,load_model,EarlyStopping,GradientRejection
from utils.config.configurations import cfg
from utils.metadata import initialize_metadata




import time
import random
from math import floor
from collections import Counter
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
from sklearn.metrics import f1_score
from tqdm import tqdm
import itertools
import argparse
import json
import multiprocessing as mp
# mp.set_start_method('spawn')
from tabulate import tabulate


from torchmetrics import Accuracy
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

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

import torch.nn as nn
from collections import OrderedDict
from torch.nn.init import kaiming_uniform_
from torch.nn import Module,Linear,ReLU,Sigmoid
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
from avalanche.benchmarks.utils import AvalancheDataset
selector,family_info,is_supervised = None,None,None
selector_analyst_attack = []
selector_analyst_benign = []
method =None
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
def apply_soft_threshold(cosine_similarities_matrix, initial_threshold=0.9, threshold_step=0.05, min_threshold=0.6):
    indices_with_high_similarity = []
    valid_indices =[]
    for i in range(len(cosine_similarities_matrix)):
        threshold = initial_threshold
        while threshold >= min_threshold:
            indices = np.where(cosine_similarities_matrix[i] > threshold)[0]  
            if len(indices) > 0:
                indices_with_high_similarity.append(indices)
                valid_indices.append(i)
                break  # Move to the next unlabeled sample once we find at least one index
            threshold -= threshold_step  # Reduce the threshold if no indices are found
    return indices_with_high_similarity,valid_indices
def major_representations(memory_y,indices_with_high_similarity) :
    filtered_indices = []
    majority_labels = []

    for group_indices in indices_with_high_similarity:
        # Get the labels for the current group of indices
        group_labels = memory_y[group_indices]
        # Find the majority label
        unique_labels, counts = np.unique(group_labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        majority_labels.append(majority_label)

        # Filter indices with the majority label
        filtered_group = group_indices[group_labels == majority_label]
        filtered_indices.append(filtered_group)


    return filtered_indices, majority_labels

rejected_samples_per_task_list = []
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
def set_cl_strategy_name(strategy_id):
    if strategy_id == 0:
        cfg.clstrategy = "CBRS"            
    elif strategy_id == 1:
        cfg.clstrategy = "ECBRS"
    elif strategy_id == 2:
        cfg.clstrategy = "ECBRS_taskaware"    
     
         
def load_model_metadata(w_decay):
    log("loading model parameter")
    global student_model1,student_model2,student_supervised,student_optimizer1,student_optimizer2,student_supervised_optimizer,loss_fn,train_acc_metric,input_shape
    global teacher_model1,teacher_model2,teacher_supervised

    w_d = w_decay
    student_model1 = APIGRAPH_FC(inputsize=input_shape)
    print(student_model1)
    teacher_model1 = APIGRAPH_FC(inputsize=input_shape)
    student_optimizer1 = torch.optim.SGD(student_model1.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=w_d)
    loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    # train_acc_metric = Accuracy(task='multiclass',                                           
    #                                  num_classes=2).to(device)

          

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




class dataset(Dataset):

    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
 
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.length


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




def split_a_task(y_classname,lab_ratio,task_class_ids=None):
    global batch_size
    labeled_indices,unlabeled_indices = [],[]
    # print("task classes",task_class_ids)
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




def sample_batch_from_memory(mem_batchsize,minority_alloc):
    if mem_batchsize > 0:
        majority_class_idices,minority_class_indices = [],[]
        global memory_X,memory_y,memory_y_name,minorityclass_ids
        minority_classes = [int(class_idx) for class_idx in minorityclass_ids]
        unique_class = np.unique(memory_y_name).tolist()
        majority_class = list(set(unique_class)-set(minority_classes))

        if mem_strat == "replace":
            majority_class = [0]
            for class_idx in majority_class:
                indices = (np.where(memory_y == int(class_idx))[0]).tolist()
                majority_class_idices.extend(indices) 
    
        else:
            for class_idx in majority_class:
                indices = (np.where(memory_y_name == int(class_idx))[0]).tolist()
                majority_class_idices.extend(indices)

    
        minority_class_indices = list(set(range(0,memory_X.shape[0]))-set(majority_class_idices))
        minority_offset = floor(mem_batchsize*minority_alloc)
        majority_offset = mem_batchsize-minority_offset
        select_indices = min(minority_offset,len(minority_class_indices))
        select_indices = max(1, select_indices)
        minority_class_indices = random.sample(minority_class_indices,select_indices)
        select_indices = min(majority_offset,len(majority_class_idices))
        select_indices = max(1, select_indices)
        # print(majority_class_idices,select_indices)
        # print(len(majority_class_idices),select_indices)
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
    print(f'\nclasses in memory: {class_in_memory}')

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
    print(f'\nclasses in memory: {class_in_memory}')

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
    print(f'\nclasses in memory: {class_in_memory}')

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
        # cos_dist = cdist(top_class_0_data, memory_X,'cosine')
        cos_dist = cdist(model.forward_encoder(torch.from_numpy(top_class_0_data).to(device)).detach().cpu().numpy(), model.forward_encoder(torch.from_numpy(memory_X).to(device)).detach().cpu().numpy(),'cosine')   
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
        cos_dist = cdist(model.forward_encoder(torch.from_numpy(top_class_1_data).to(device)).detach().cpu().numpy(), model.forward_encoder(torch.from_numpy(memory_X).to(device)).detach().cpu().numpy(),'cosine')
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
                owl_self_labelled_count_class_1 += selection_count_1
            else:
                print('No. of self-labelled samples (class 1): 0')

    if not unseen_task:
        return [curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1]
    
    print(f'\nTotal no. of self-labeled samples = {selection_count_0 + selection_count_1} (0: {selection_count_0}, 1: {selection_count_1})')

    # Get security analyst to label the remaining high confidence samples
   
    
    remaining_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    y_rem = y[remaining_indices]
   
    remaining_0_indices = remaining_indices[np.where(y_rem == 0)[0]] # remaining indices where y == 0 
    remaining_1_indices = remaining_indices[np.where(y_rem == 1)[0]] # remaining indices where y == 1
    ## First past both the unlabeled and mem_x through the model and then take cosine distance. Take average and sort .
    feat_unlabl_0 = student_model1.forward_encoder(torch.from_numpy(X[remaining_0_indices]).to(device))
    feat_unlabl_1 = student_model1.forward_encoder(torch.from_numpy(X[remaining_1_indices]).to(device))
    # obtain the positive and negative samples
    mem_0_indices = np.where(memory_y ==0)
    mem_1_indices = np.where(memory_y ==1)

    cos_dist_0  = cdist(feat_unlabl_0.detach().cpu().numpy(), model.forward_encoder(torch.from_numpy(memory_X[mem_0_indices]).to(device)).detach().cpu().numpy(),'cosine')
    cos_dist_1  = cdist(feat_unlabl_1.detach().cpu().numpy(), model.forward_encoder(torch.from_numpy(memory_X[mem_1_indices]).to(device)).detach().cpu().numpy(),'cosine')
    avg_cos_dist_0 = cos_dist_0.mean(axis=1)  # Average cosine distance for each top_class_0_data row
    avg_cos_dist_1 = cos_dist_1.mean(axis=1)  # Average cosine distance for each top_class_1_data row
    # Sort the indices based on the average values (ascending order)
    sorted_indices_0 = np.argsort(avg_cos_dist_0)[::-1]  # Indices for top_class_0_data sorted by min to max average cosine distance
    sorted_indices_1 = np.argsort(avg_cos_dist_1)[::-1]  # Indices for top_class_1_data sorted by min to max average cosine distance

    # Apply the sorting to the original indices
    sorted_top_class_0_indices = remaining_0_indices[sorted_indices_0]
    sorted_top_class_1_indices = remaining_1_indices[sorted_indices_1]

    selected_0_indices = sorted_top_class_0_indices [:min(len(sorted_top_class_0_indices), int(0.1*(analyst_labels)))  ]
    selected_1_indices = sorted_top_class_1_indices [:min(len(sorted_top_class_1_indices), int(0.9*(analyst_labels))) ]
    count_class_0 = len(selected_0_indices)
    count_class_1 = len(selected_1_indices)
    temp_X = np.vstack((X[selected_0_indices], X[selected_1_indices]))
    temp_y = np.hstack(([0]*count_class_0, [1]*count_class_1))
    temp_y_classname = np.hstack(([benign_y_name]*count_class_0, [attack_y_name]*count_class_1))
    print(len(remaining_0_indices), count_class_0)
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


def train_new_method(str_train_model,tasks,task_class_ids,task_id,feature_list,threshold,X_val,y_val,bool_reorganize_memory,owl_data_labeling=False):
    
    global memory_X, memory_y, memory_y_name,local_count,global_count,local_store,input_shape,memory_size,task_num
    global classes_so_far,full,global_priority_list,local_priority_list,memory_population_time,replay_size
    global memory_population_time,epochs,grad_norm_dict,temp_norm
    global student_optimizer1,student_optimizer2,student_supervised_optimizer
    global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised, encoder
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

        if task_id == 0:
            labeled_indicies,unlabeled_indicies=split_a_task(y_classname,0.99,task_class_ids)
        else:
            labeled_indicies,unlabeled_indicies=split_a_task(y_classname,labels_ratio,task_class_ids)
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
            task_truth_agreement_fractions = owl_data_labeling_strategy_const_labels(X, y, y_classname, unseen_task=False,task_id=task_id)
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
        labeled_X, labeled_y, labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies = owl_data_labeling_strategy_const_labels(X, y, y_classname, unseen_task=True,task_id=task_id)
        
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
    
    
    if task_id > 0:
              
            mem_batch_size = floor(batch_size*b_m)
            rem_batch_size = batch_size-mem_batch_size
            # task_size = X.shape[0] + memory_X.shape[0] 
            labeled_batch_size = floor(rem_batch_size*labels_ratio)
            unlabeled_batch_size = rem_batch_size - (labeled_batch_size)
            no_of_batches = floor(len(labeled_indicies)/labeled_batch_size)
            no_of_unlab_batches = floor(len(unlabeled_indicies)/unlabeled_batch_size)
            p = np.random.permutation(labeled_X.shape[0])
            labeled_X,labeled_y,labeled_y_classname = labeled_X[p,:],labeled_y[p],labeled_y_classname[p]
            
    else:
        # initialize_buffermemory(labeled_task,memory_size)
        task_size = X.shape[0]    
        # labeled_batch_size = floor(batch_size*0.99)
        labeled_batch_size = floor(batch_size*labels_ratio)
        unlabeled_batch_size = batch_size-labeled_batch_size
        no_of_batches = floor(task_size/batch_size)

    if bool_gpm:
        for i in range(len(feature_list)):
            Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
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


    ##Training encoder for self-supervision
    print("************Training the Encoder***********")
    # encoder = train_encoder (encoder, X_train, y_train, optimizer, total_epochs, batch_size, is_tensor=True)


    # prog_bar = tqdm(range(no_of_batches))
    # for batch_idx in prog_bar:
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    check_point_file_name = "checkpoint"+str(os.getpid())+".pt"
    check_point_file_name_norm = "checkpoint"+str(os.getpid())+"grad_norm"+".pt"
    early_stopping = EarlyStopping(patience=3, verbose=True,path=check_point_file_name)
    gradient_rejection = GradientRejection(patience=2, verbose=True,path=check_point_file_name_norm)
    scheduler = StepLR(opt, step_size=1, gamma=0.96)
    for epoch in range(epochs):
        # print("epoch",epoch)
        # scheduler.step()
        prog_bar = tqdm(range(no_of_batches))
        for batch_idx in prog_bar:
            model.train()        
        # for epoch in range(epochs):
            with torch.no_grad():
                if task_id > 0 and batch_idx < no_of_unlab_batches:
                    unlabeled_X = torch.from_numpy(X_unlab[batch_idx*unlabeled_batch_size:batch_idx*unlabeled_batch_size+unlabeled_batch_size]).to(device)
                else:
                    rand_indices = list(random.sample(range(X_unlab.shape[0]),min(unlabeled_batch_size,X_unlab.shape[0])))
                    unlabeled_X = torch.from_numpy(X_unlab[rand_indices]).to(device)
                if image_resolution is not None:
                    unlabeled_X = unlabeled_X.reshape(image_resolution)
                unlabeled_pred = torch.softmax(model(unlabeled_X),dim=1).detach()
            lab_X = labeled_X[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]  
            lab_y = labeled_y[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]
            task_lab_X,task_lab_y = lab_X,lab_y
            if task_id > 0:
                
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
            if image_resolution is not None:
                    lab_X = lab_X.reshape(image_resolution)
                   
            # print(model(lab_X))
            y_pred = torch.softmax(model(lab_X),dim=1).squeeze()#.to(device)                      
            lab_y = torch.from_numpy(lab_y).to(device).to(dtype=torch.long)#.reshape(y_pred.shape)

            # lab_y = F.one_hot(lab_y, 2)
            sup_loss = loss_fn(y_pred.float(),F.one_hot(lab_y.to(dtype=torch.long), 2).float())#to(device)
            # sup_loss = loss_fn(y_pred,lab_y.float())
            total_loss = sup_loss
            distil_loss = 0
            distil_loss = torch.as_tensor(distil_loss).to(device)
            opt.zero_grad()
            if task_id > 0:
                if str_train_model!="student_supervised":
                    #computing the distillation loss
                    # distil_loss_list = compute_distill_loss(unlabeled_pred,unlabeled_X)
                    # distil_loss = distil_loss_list[0]

                    # distil_loss = compute_distill_loss_self_supervision(p_m=0.3, K=3,unlabeled_x=unlabeled_X,encoder_model=encoder)
                    total_loss = total_loss 
                    # total_loss = total_loss +  alpha *distil_loss  
                    # lab_X = torch.cat((lab_X,unlabeled_X),0)
                    # lab_y = torch.cat((lab_y,distil_loss_list[1]),0)

                contrast_loss = 0
                
                if bool_closs:
                    # positives, negatives = construct_positive_negative_samples(lab_X, lab_y)  
                    positives, negatives = construct_positive_negative_samples_from_memory(task_lab_y)
                    anchor_representations = model(torch.from_numpy(task_lab_X).to(device)) ### Get Encoder Representations
                    positive_representations = model(positives) ### Get Encoder Representations
                    negative_representations = model(negatives) ### Get Encoder Representations
                    contrast_loss = contrastive_loss(anchor_representations, positive_representations, negative_representations)
                total_loss = total_loss+contrast_loss
                # print(y_pred)
                # print("total_loss",total_loss)
                if bool_gpm:
                    total_loss.backward()
                    # for i in range(len(feature_list)):
                    #     Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
                    #     feature_mat.append(Uf)
                    bn_counter = 0
                    for k, (m,params) in enumerate(model.named_parameters()):
                        # print(params.grad)
                        # print(m)
                        if 'bn' not in m:
                            k -= bn_counter
                            sz =  params.grad.data.size(0)
                            params.grad.data = torch.mul((params.grad.data - torch.mul(torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[k]).view(params.size()),1)), (1))  
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
            prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f}'.format(
                 total_loss.item(), sup_loss.item(), distil_loss.item()))
        
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
        # valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        # avg_valid_losses.append(valid_loss)
        epoch_len = len(str(epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'PR-AUC (I): {lr_auc:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
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

    # print(f'Buffer memory size for task {task_id}: {memory_X.shape}')
    
    return feature_list

     
def train(str_train_model,tasks,task_class_ids,task_id,feature_list,threshold,X_val,y_val,bool_reorganize_memory,owl_data_labeling=False):
    
    global memory_X, memory_y, memory_y_name,local_count,global_count,local_store,input_shape,memory_size,task_num
    global classes_so_far,full,global_priority_list,local_priority_list,memory_population_time,replay_size
    global memory_population_time,epochs,grad_norm_dict,temp_norm
    global student_optimizer1,student_optimizer2,student_supervised_optimizer
    global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised
    global truth_agreement_fraction_0, truth_agreement_fraction_1 
    global avg_CI, CI_list 
    global rejected_samples_per_task_list
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

        if task_id == 0:
            labeled_indicies,unlabeled_indicies=split_a_task(y_classname,0.99,task_class_ids)
        else:
            labeled_indicies,unlabeled_indicies=split_a_task(y_classname,labels_ratio,task_class_ids)
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
            task_truth_agreement_fractions = owl_data_labeling_strategy_const_labels(X, y, y_classname, unseen_task=False,task_id=task_id)
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
        labeled_X, labeled_y, labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies = owl_data_labeling_strategy_const_labels(X, y, y_classname, unseen_task=True,task_id=task_id)

    
    ### Calculating Uncertainity ------------------------ ------------------------ ------------------------ ------------------------ ------------------------ ------------------------
    uncertainity_unlabelled_data = X[unlabeled_indicies]
    uncertainity_unlabelled_data_loader = torch.utils.data.DataLoader(uncertainity_unlabelled_data, batch_size=100, shuffle=False, num_workers=1)
    uncertainity = get_uncertainity(model,device,uncertainity_unlabelled_data_loader)
    ce = MarginLoss(m=-0.1*uncertainity)
    #------------------------ ------------------------ ------------------------ ------------------------ ------------------------ ------------------------ ----------------------------
    if task_id > 0:
            mem_batch_size = floor(batch_size*b_m)
            rem_batch_size = batch_size-mem_batch_size
            # task_size = X.shape[0] + memory_X.shape[0] 
            labeled_batch_size = floor(rem_batch_size*labels_ratio)
            unlabeled_batch_size = rem_batch_size - (labeled_batch_size)
            no_of_batches = floor(len(labeled_indicies)/labeled_batch_size)
            no_of_unlab_batches = floor(len(unlabeled_indicies)/unlabeled_batch_size)
            p = np.random.permutation(labeled_X.shape[0])
            labeled_X,labeled_y,labeled_y_classname = labeled_X[p,:],labeled_y[p],labeled_y_classname[p]
            
    else:
        # initialize_buffermemory(labeled_task,memory_size)
        task_size = X.shape[0]    
        # labeled_batch_size = floor(batch_size*0.99)
        labeled_batch_size = floor(batch_size*labels_ratio)
        unlabeled_batch_size = batch_size-labeled_batch_size
        no_of_batches = floor(task_size/batch_size)

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
            if task_id == 5:
                print(temp_x.shape,np.unique(temp_y, return_counts=True))
            memory_X, memory_y, memory_y_name = temp_x,temp_y,temp_yname
            # tasks[0] = temp_x,temp_y,temp_yname
            # lab_samples_in_memory = split_a_task(tasks,lab_samp_in_mem_ratio)
            # tasks[0] = temp_x[lab_samples_in_memory[0],:],temp_y[lab_samples_in_memory[0]],temp_yname[lab_samples_in_memory[0]]
            # initialize_buffermemory(tasks=tasks,mem_size=memory_size)
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
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    # to track no of samples with no close samples in memo
    avg_rejected_samples = []
    os.makedirs("models", exist_ok=True)
    check_point_file_name = "models/checkpoint"+str(os.getpid())+".pt"
    check_point_file_name_norm = "models/checkpoint"+str(os.getpid())+"grad_norm"+".pt"
    early_stopping = EarlyStopping(patience=5, verbose=True,path=check_point_file_name)
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
                if image_resolution is not None:
                    unlabeled_X = unlabeled_X.reshape(image_resolution)
                unlabeled_pred = torch.softmax(model(unlabeled_X),dim=1).detach()
            
            lab_X = labeled_X[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]  
            lab_y = labeled_y[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]
            task_lab_X,task_lab_y = lab_X,lab_y
            if task_id > 0:
                # if task_id == 5:
                #     print(floor(batch_size*b_m))
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
                basis = Vh.T # Basis of the vector space of mem_feat
            # Normalize mem_feat directly on the GPU
            mem_feat_norm = F.normalize(mem_feat, p=2, dim=1)
            # Projection of Unlabelled onto the that space
            feat_unlab = model.forward_encoder(unlabeled_data)
            projected_unlabelled = feat_unlab @ basis @ basis.T
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
            cosine_similarities_matrix = torch.mm(proj_unlab_norm, mem_feat_norm.t()).detach().cpu().numpy()

            # Apply soft threshold to get indices with high similarity
            indices_with_high_similarity,valid_indices = apply_soft_threshold(cosine_similarities_matrix, initial_threshold=0.95)
            filtered_indices, majority_label = major_representations(memory_y,indices_with_high_similarity)
            excluded_data = [unlabeled_data[i] for i in range(len(unlabeled_data)) if i not in valid_indices]
            rejected_samples  = len(unlabeled_data) - len(valid_indices)
            # print(excluded_data,rejected_samples)
            batch_rejected_samples += rejected_samples
            # print(f"Length of all indices : {len(indices_with_high_similarity[0])} and filtered ones : {len(filtered_indices[0])} and label is {majority_label[0]}")
            # Efficiently compute the average representations for each set of high-similarity indices
            start_time = time.time()
            avg_representations = []
            for indices in filtered_indices:
                selected_features = mem_feat[indices]  # Retrieve the memory features corresponding to these indices
                avg_representation = torch.mean(selected_features, dim=0)  # Compute mean along the specified axis
                avg_representations.append(avg_representation)
            avg_representations = torch.stack(avg_representations).to(device)
            # print(f"Time taken by old method is {time.time() - start_time} and size of indices_with_high_similarity is {len(avg_representations)} and {max(len(inner_list) for inner_list in avg_representations)} ")
            prob_avg_rep = model.forward_classifier(avg_representations)
            prob_avg_rep = torch.softmax(prob_avg_rep,dim=1).squeeze()#.to(device)                      
            # Labeled probabs
            pos_pair_probs = prob1[pos_pairs,:]
            # Unlabeled Probabs 
            pos_pair_probs = torch.concat([pos_pair_probs,prob_avg_rep])
            pos_main_probs = torch.concat([prob1[pos_pairs,:],prob1[labeled_len:,:][valid_indices,:]])
            # print("Are there NaNs in prob1?", torch.isnan(prob1).any())
            # print("Are there NaNs in prob_avg_rep?", torch.isnan(prob_avg_rep).any())
            pos_sim = torch.bmm(pos_main_probs.view(pos_main_probs.size(0), 1, -1),  # prob1.size(0) gives batch size
                            pos_pair_probs.view(pos_pair_probs.size(0), -1, 1)  # pos_pair_probs.size(0) gives batch size
                            ).squeeze()
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
            ones  = torch.ones_like(pos_sim)
            bce_loss = loss_fn(pos_sim,ones)
            ce_loss = ce(output1[:labeled_len],lab_y[:labeled_len])
            # entropy_loss = entropy(torch.mean(prob1,0))
            entropy_loss = entropy(torch.mean(prob1[labeled_len:],0))
            total_loss = - entropy_loss + ce_loss +  bce_loss
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
        # valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        # avg_valid_losses.append(valid_loss)
        epoch_len = len(str(epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'PR-AUC (I): {lr_auc:.5f}')
        
        print(print_msg)
        avg_rejected_samples.append(batch_rejected_samples)
        # clear lists to track next epoch
        train_losses = []
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
        threshold.append(np.random.choice([0.95,0.99,0.99,0.98,0.99,0.99,0.99],1)[0])
    if bool_gpm:
        mat_list = get_representation_matrix (model, device, temp_x, temp_y,rand_samples=no_of_rand_samples)
        feature_list = update_GPM(model, mat_list, threshold, feature_list)
    else:
        feature_list = []

    # grad_norm_dict[task_id] = grad_norm_list   
    # print(grad_norm_dict)

    if os.path.exists(check_point_file_name_norm):
        os.remove(check_point_file_name_norm) 

    # print(f'Buffer memory size for task {task_id}: {memory_X.shape}')
    print(f"Average Rejected Samples per task : {np.mean(avg_rejected_samples)}")
    rejected_samples_per_task_list.append(np.mean(avg_rejected_samples))
    return feature_list
def train_baseline_hcl(supervised = True,threshold = 0.8):
    global input_shape
    # Step 1 : Load Tasks : Seen Task and Unseen Tasks
    train_order = task_order[:training_cutoff]
    test_order = task_order[training_cutoff:]
    # Step 2 : Select Model
    if type_selector =="pseudo-loss" :
        enc_dims = [input_shape,512,384,256,128]
        mlp_dims = [128,100,100,2]
        encoder = SimpleEncClassifier(enc_dims=enc_dims,mlp_dims=mlp_dims).to(device)
        encoder.apply(init_weights)
    elif type_selector == "cade":
        enc_dims = [input_shape,512,128,32,7]
        mlp_dims = [7,100,100,2]
        encoder = CAE(enc_dims)
        encoder.apply(init_weights)
        classifier = MLPClassifier(mlp_dims)
        optimizer_mlp = torch.optim.Adam(classifier.parameters(), lr=l_rate)
    print(f"Starting the Trainig for {training_cutoff} tasks...")
    print("-"*20)
    X_train,y_train_bin,y_train = np.array([]),np.array([]),np.array([])
    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=l_rate,betas=(0.9, 0.999),weight_decay=w_decay)
    for task_id , task in enumerate(train_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("\nloading task:",task_id)     
        input_shape,tasks,X_test,y_test,X_val,y_val = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=bool_encode_benign,bool_encode_anomaly=bool_encode_anomaly,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=load_whole_train_data)
        valid_loader = torch.utils.data.DataLoader(dataset(np.concatenate([X_val,X_test]),np.concatenate([y_val,y_test])),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=0)
        val_x_all_tasks.extend([X_val])
        val_y_all_tasks.extend([y_val])
        # val_x_all_tasks,val_y_all_tasks = [X_val],[y_val]
        print("validation dataset size",X_val.shape)
        test_x.extend([X_test])
        test_y.extend([y_test])
        print("Training task:",task_id)
        task_num = task_id
        X,y_bin,y_classname,y = tasks[0]

        if task_id == 0:
            X_train,y_train_bin,y_train = X,y_bin,y
        else:
             X_train,y_train_bin,y_train = np.concatenate([X_train,X]),np.concatenate([y_train_bin,y_bin]),np.concatenate([y_train,y])
        if not supervised :
                labeled_indicies,unlabeled_indices=split_a_task(y_classname,labels_ratio,task_class_ids)
                X_labeled,y_bin_labeled,y_classname_labeled,y_labeled = X[labeled_indicies],y_bin[labeled_indicies],y_classname[labeled_indicies],y[[labeled_indicies]]
                X_unlabeled = X[unlabeled_indices]
    
                # Predict probabilities for unlabeled data using the encoder model
                if type_selector =="cade":
                    probas = classifier.predict_proba(torch.from_numpy(X_unlabeled ).to(device))
                elif type_selector == "pseudo-loss":
                    probas = encoder.predict_proba(torch.from_numpy(X_unlabeled ).to(device))
                pseudo_labels = probas.argmax(axis=1).cpu()  # Predicted labels
                confidences, _ = torch.max(probas, dim=1)
                
                # Filter pseudo-labels based on confidence threshold
                high_confidence_mask = confidences >= threshold
                high_confidence_mask = high_confidence_mask.cpu()
                X_pseudo = X_unlabeled[high_confidence_mask]
                pseudo_labels_high_conf = pseudo_labels[high_confidence_mask]
                X = np.concatenate([X_labeled,X_pseudo])
                y_bin = np.concatenate([y_bin_labeled,pseudo_labels_high_conf])
                y_classname = np.concatenate([y_classname_labeled,y_classname[unlabeled_indices][high_confidence_mask] ])
                y = np.concatenate([y_labeled,y[unlabeled_indices][high_confidence_mask]])

        
        if type_selector == "cade":
            train_encoder(encoder,X_train=X,y_train=y,y_train_binary=y_bin,optimizer=optimizer_enc,total_epochs=50,model_path="models/cade",device=device,
                      batch_size=batch_size,valid_loader=valid_loader,family_info=family_info,type_selector=type_selector,lambda_=0.1,margin=10)
            X_enc = encoder.encode(torch.from_numpy(X).float().to(device)).detach().cpu().numpy()
            X_enc_val = encoder.encode(torch.from_numpy(np.concatenate([X_val,X_test] )).float().to(device)).detach().cpu().numpy()
            valid_loader_enc = torch.utils.data.DataLoader(dataset(X_enc_val,np.concatenate([y_val,y_test])),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=4)
            train_classifier(classifier,X_train=X_enc,y_train = y_bin, optimizer=optimizer_mlp ,total_epochs=50,batch_size=batch_size,model_path="models/cade",valid_loader=valid_loader_enc,device=device)
        elif type_selector == "pseudo-loss":
            train_encoder(encoder,X_train=X,y_train=y,y_train_binary=y_bin,optimizer=optimizer_enc,total_epochs=50,model_path="models/hcl",device=device,
                      batch_size=batch_size,valid_loader=valid_loader,family_info=family_info,type_selector=type_selector,lambda_=100)
    ### For the Unseen Tasks

    ## Sample SELECTOR
    if type_selector =="pseudo-loss" :
        selector = LocalPseudoLossSelector(encoder)
    elif type_selector == "cade":
        selector = OODSelector(encoder,device)
    prev_train_size = X_train.shape[0]
    for task_id, task in enumerate(test_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("\nLoading task:", task_id)
        
        # Load data
        X_test, y_test_bin, y_test = load_task( path=pth, task=task)
        valid_loader = torch.utils.data.DataLoader(
            dataset(X_test, y_test_bin),
            batch_size=batch_size,
            num_workers=0
        )
        
        # Initial evaluation
        print("Evaluating before training...")
        # metrics_before = evaluate_model(encoder, valid_loader, device, None)
        
        

        # Sample selection
        if type_selector == "pseudo-loss":
            # Predict test data
            y_test_pred = encoder.predict(torch.from_numpy(X_test).to(device).float()).cpu().detach().numpy()
            test_offset = prev_train_size
            sample_indices, sample_scores = selector.select_samples(
                X_train, y_train, y_train_bin,
                X_test, y_test_pred,
                10,
                test_offset,
                y_test,
                100,
                device=device,
                batch_size=batch_size
            )
        elif type_selector == "cade":
            sample_indices, sample_scores = selector.select_samples(X_train, y_train, X_test, 100)
        
        # Add selected samples to the training set
        X_train = np.concatenate([ X_test[sample_indices]])
        y_train_bin = np.concatenate([ y_test_bin[sample_indices]])
        y_train = np.concatenate([ y_test[sample_indices]])
        # Gathering Data
        selector_analyst_attack.append(int((y_test_bin[sample_indices] ==1).sum()))
        selector_analyst_benign.append(int((y_test_bin[sample_indices] ==0).sum()))
        # Remove selected samples from the test set
        X_test = np.delete(X_test, sample_indices, axis=0)
        y_test_bin = np.delete(y_test_bin, sample_indices, axis=0)
        y_test = np.delete(y_test, sample_indices, axis=0)
        
        valid_loader = torch.utils.data.DataLoader(
            dataset(X_test, y_test_bin),
            batch_size=batch_size,
            num_workers=0
        )
        # Train encoder
        if type_selector == "cade":
            train_encoder(encoder,X_train=X_train,y_train=y_train,y_train_binary=y_train_bin,optimizer=optimizer_enc,total_epochs=50,model_path="models/cade",device=device,
                      batch_size=batch_size,valid_loader=valid_loader,family_info=family_info,type_selector=type_selector,lambda_=0.1,margin=10)
            X_enc = encoder.encode(torch.from_numpy(X_train).float().to(device)).detach().cpu().numpy()
            X_enc_val = encoder.encode(torch.from_numpy(X_test).float().to(device)).detach().cpu().numpy()
            valid_loader_enc = torch.utils.data.DataLoader(dataset(X_enc_val,y_test_bin),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=4)
            train_classifier(classifier,X_train=X_enc,y_train = y_train_bin, optimizer=optimizer_mlp ,total_epochs=50,batch_size=batch_size,device=device,model_path="models/cade",valid_loader=valid_loader_enc)
        elif type_selector == "pseudo-loss":
            train_encoder(encoder,X_train=X_train,y_train=y_train,y_train_binary=y_train_bin,optimizer=optimizer_enc,total_epochs=50,model_path="models/hcl",device=device,
                      batch_size=batch_size,valid_loader=valid_loader,family_info=family_info,type_selector=type_selector,lambda_=100)
        
        # # Final evaluation
        # print("Evaluating after training...")
        # metrics_after = evaluate_model(encoder, valid_loader, device, None)
        # # Compare performance
        # pr_auc_before = metrics_before["PR-AUC (Attack)"]
        # pr_auc_after = metrics_after["PR-AUC (Attack)"]
        
        # if pr_auc_after > pr_auc_before:
        #     print(f"Performance improved after training! PR-AUC increased from {pr_auc_before:.5f} to {pr_auc_after:.5f}")
        # elif pr_auc_after < pr_auc_before:
        #     print(f"Performance decreased after training. PR-AUC decreased from {pr_auc_before:.5f} to {pr_auc_after:.5f}")
        # else:
        #     print(f"Performance remained the same after training. PR-AUC: {pr_auc_before:.5f}")
    test_set_results = []
    with open(temp_filename, 'w') as fp:
        if type_selector == "cade":
            test_set_results.extend([testing(classifier=classifier,path= f"output/baseline/{type_selector}",training_cutoff=training_cutoff,type="seen" ,task_id=task_id ,seen_data=True,model=encoder),testing(classifier=classifier,path= f"output/baseline/{type_selector}",training_cutoff=training_cutoff, type="unseen" ,task_id=task_id ,seen_data=False,model=encoder),str(owl_self_labelled_count_class_0), str(owl_self_labelled_count_class_1),str(owl_analyst_labelled_count_class_0), str(owl_analyst_labelled_count_class_1),testing(classifier=classifier,path= f"output/baseline/{type_selector}",training_cutoff=len(task_order), type="all" ,task_id=task_id ,seen_data=True,model=encoder) ])
        elif type_selector == "pseudo-loss": 
            test_set_results.extend([testing(path= f"output/baseline/{type_selector}",training_cutoff=training_cutoff,type="seen" ,task_id=task_id ,seen_data=True,model=encoder),testing(path= f"output/baseline/{type_selector}",training_cutoff=training_cutoff, type="unseen" ,task_id=task_id ,seen_data=False,model=encoder),str(owl_self_labelled_count_class_0), str(owl_self_labelled_count_class_1),str(owl_analyst_labelled_count_class_0), str(owl_analyst_labelled_count_class_1),testing(path= f"output/baseline/{type_selector}",training_cutoff=len(task_order), type="all" ,task_id=task_id ,seen_data=True,model=encoder) ])
        auc_result[str(args.seed)] = test_set_results
        json.dump(auc_result, fp)  
def load_task(path="../data/data_processed/test/bodmas/",task=""):
    print(task)
    X_pos =  np.load(path+task[1] +'.npy')[:,:-1]
    X_neg = np.load(path+task[0]+'.npy')[:,:-1]
    y_family = np.load(path+task[1]+'_labels.npy')
    y_pos = [1] * len(X_pos)
    y_neg = [0] * len(X_neg)
    y_neg_family = [0] * len(X_neg)
    X ,y ,y_family = np.concatenate([X_pos,X_neg]),np.concatenate([y_pos,y_neg]) , np.concatenate([y_family,y_neg_family])
    indices = np.arange(len(X))
    random_indices = np.random.permutation(indices)
    print(f"Concatenated X shape: {X.shape}")
    print(f"Concatenated y shape: {y.shape}")
    print(f"Concatenated y_family shape: {y_family.shape}")
    return X[random_indices],y[random_indices],y_family[random_indices]
def train_baseline_hcl_good(supervised = True,threshold = 0.8):
    global input_shape
    # Step 1 : Load Tasks
    train_order = task_order[:training_cutoff]
    test_order = task_order[training_cutoff:]
    # Step 2 : Select Model
    if type_selector =="pseudo-loss" :
        enc_dims = [input_shape,512,384,256,128]
        mlp_dims = [128,100,100,2]
        encoder = SimpleEncClassifier(enc_dims=enc_dims,mlp_dims=mlp_dims).to(device)
        encoder.apply(init_weights)
    elif type_selector == "cade":
        enc_dims = [input_shape,512,128,32,7]
        mlp_dims = [7,100,100,2]
        encoder = CAE(enc_dims)
        classifier = MLPClassifier(mlp_dims)
        optimizer_mlp = torch.optim.Adam(classifier.parameters(), lr=l_rate)
    print(f"Starting the Trainig for {training_cutoff} tasks...")
    print("-"*20)
    X_train,y_train_bin,y_train = np.array([]),np.array([]),np.array([])
    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=l_rate, betas=(0.9, 0.999), weight_decay=w_decay)
    for task_id , task in enumerate(train_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("\nloading task:",task_id)     
        input_shape,tasks,X_test,y_test,X_val,y_val = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=bool_encode_benign,bool_encode_anomaly=bool_encode_anomaly,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=load_whole_train_data)
        valid_loader = torch.utils.data.DataLoader(dataset(np.concatenate([X_val,X_test]),np.concatenate([y_val,y_test])),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=0)
        val_x_all_tasks.extend([X_val])
        val_y_all_tasks.extend([y_val])
        # val_x_all_tasks,val_y_all_tasks = [X_val],[y_val]
        print("validation dataset size",X_val.shape)
        test_x.extend([X_test])
        test_y.extend([y_test])
        print("Training task:",task_id)
        task_num = task_id
        X,y_bin,y_classname,y = tasks[0]
        if task_id == 0:
            X_train,y_train_bin,y_train = X,y_bin,y
        else:
             X_train,y_train_bin,y_train = np.concatenate([X_train,X]),np.concatenate([y_train_bin,y_bin]),np.concatenate([y_train,y])
        if not supervised :
                labeled_indicies,unlabeled_indices=split_a_task(y_classname,labels_ratio,task_class_ids)
                X_labeled,y_bin_labeled,y_classname_labeled,y_labeled = X[labeled_indicies],y_bin[labeled_indicies],y_classname[labeled_indicies],y[[labeled_indicies]]
                X_unlabeled = X[unlabeled_indices]
    
                # Predict probabilities for unlabeled data using the encoder model
                probas = encoder.predict_proba(torch.from_numpy(X_unlabeled ).to(device))
                pseudo_labels = probas.argmax(axis=1).cpu()  # Predicted labels
                confidences, _ = torch.max(probas, dim=1)
                
                # Filter pseudo-labels based on confidence threshold
                high_confidence_mask = confidences >= threshold
                high_confidence_mask = high_confidence_mask.cpu()
                X_pseudo = X_unlabeled[high_confidence_mask]
                pseudo_labels_high_conf = pseudo_labels[high_confidence_mask]
                X = np.concatenate([X_labeled,X_pseudo])
                y_bin = np.concatenate([y_bin_labeled,pseudo_labels_high_conf])
                y_classname = np.concatenate([y_classname_labeled,y_classname[unlabeled_indices][high_confidence_mask] ])
                y = np.concatenate([y_labeled,y[unlabeled_indices][high_confidence_mask]])

        if type_selector == "cade":
            train_encoder(encoder,X_train=X,y_train=y,y_train_binary=y_bin,optimizer=optimizer_enc,total_epochs=50,model_path="models/cade",device=device,
                      batch_size=batch_size,valid_loader=valid_loader,family_info=family_info,type_selector=type_selector,lambda_=0.1,margin=10)
            X_enc = encoder.encode(torch.from_numpy(X).float().to(device)).detach().cpu().numpy()
            X_enc_val = encoder.encode(torch.from_numpy(np.concatenate([X_val,X_test] )).float().to(device)).detach().cpu().numpy()
            valid_loader_enc = torch.utils.data.DataLoader(dataset(X_enc_val,np.concatenate([y_val,y_test])),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=4)
            train_classifier(classifier,X_train=X_enc,y_train = y_bin, optimizer=optimizer_mlp ,total_epochs=50,batch_size=batch_size,model_path="models/cade",valid_loader=valid_loader_enc,device=device)
        elif type_selector == "pseudo-loss":
            train_encoder(encoder,X_train=X,y_train=y,y_train_binary=y_bin,optimizer=optimizer_enc,total_epochs=50,model_path="models/hcl",device=device,
                      batch_size=batch_size,valid_loader=valid_loader,family_info=family_info,type_selector=type_selector,lambda_=100)
    ### For the Unseen Tasks

    ## Sample SELECTOR
    if type_selector =="pseudo-loss" :
        selector = LocalPseudoLossSelector(encoder)
    elif type_selector == "cade":
        selector = OODSelector(encoder,device)
    prev_train_size = X_train.shape[0]
    for task_id, task in enumerate(test_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("\nLoading task:", task_id)
        
        # Load data
        X_test, y_test_bin, y_test = load_task( path=pth, task=task)
        valid_loader = torch.utils.data.DataLoader(
            dataset(X_test, y_test_bin),
            batch_size=batch_size,
            num_workers=0
        )
        
        # Initial evaluation
        if type_selector == "pseudo-loss":
            print("Evaluating before training...")
            metrics_before = evaluate_model(encoder, valid_loader, device, None)
        elif type_selector == "cade":
            print("Evaluating before training...")
            X_enc_val = encoder.encode(torch.from_numpy(X_test).float().to(device)).detach().cpu().numpy()
            valid_loader_enc = torch.utils.data.DataLoader(dataset(X_enc_val,y_test_bin),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=4)
            metrics_before = evaluate_model(classifier, valid_loader_enc, device, None)
        
        # Predict test data
    
        # Sample selection
        if type_selector == "pseudo-loss":
            y_test_pred = encoder.predict(torch.from_numpy(X_test).to(device).float()).cpu().detach().numpy()
            test_offset = prev_train_size
            sample_indices, sample_scores = selector.select_samples(
                X_train, y_train, y_train_bin,
                X_test, y_test_pred,
                10,
                test_offset,
                y_test,
                100,
                device=device,
                batch_size=batch_size
            )
        elif type_selector == "cade":
            sample_indices, sample_scores = selector.select_samples(X_train, y_train, X_test, 100)
        
        # Add selected samples to the training set
        X_train = np.concatenate([ X_test[sample_indices]])
        y_train_bin = np.concatenate([ y_test_bin[sample_indices]])
        y_train = np.concatenate([ y_test[sample_indices]])
        # Gathering Data
        selector_analyst_attack.append(int((y_test_bin[sample_indices] ==1).sum()))
        selector_analyst_benign.append(int((y_test_bin[sample_indices] ==0).sum()))
        # Remove selected samples from the test set
        X_test = np.delete(X_test, sample_indices, axis=0)
        y_test_bin = np.delete(y_test_bin, sample_indices, axis=0)
        y_test = np.delete(y_test, sample_indices, axis=0)
        
        valid_loader = torch.utils.data.DataLoader(
            dataset(X_test, y_test_bin),
            batch_size=batch_size,
            num_workers=0
        )
        
        if type_selector == "cade":
            train_encoder(encoder,X_train=X_train,y_train=y_train,y_train_binary=y_train_bin,optimizer=optimizer_enc,total_epochs=50,model_path="models/cade",device=device,
                      batch_size=batch_size,valid_loader=valid_loader,family_info=family_info,type_selector=type_selector,lambda_=0.1,margin=10)
            X_enc = encoder.encode(torch.from_numpy(X_train).float().to(device)).detach().cpu().numpy()
            X_enc_val = encoder.encode(torch.from_numpy(X_test).float().to(device)).detach().cpu().numpy()
            valid_loader_enc = torch.utils.data.DataLoader(dataset(X_enc_val,y_test_bin),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=4)
            train_classifier(classifier,X_train=X_enc,y_train = y_train_bin, optimizer=optimizer_mlp ,total_epochs=50,batch_size=batch_size,device=device,model_path="models/cade",valid_loader=valid_loader_enc)
        elif type_selector == "pseudo-loss":
            train_encoder(encoder,X_train=X_train,y_train=y_train,y_train_binary=y_train_bin,optimizer=optimizer_enc,total_epochs=50,model_path="models/hcl",device=device,
                      batch_size=batch_size,valid_loader=valid_loader,family_info=family_info,type_selector=type_selector,lambda_=100)
        
        # Final evaluation
        if type_selector == "pseudo-loss":
            print("Evaluating after training...")
            metrics_after = evaluate_model(encoder, valid_loader, device, None)
        elif type_selector == "cade":
            print("Evaluating after training...")
            X_enc_val = encoder.encode(torch.from_numpy(X_test).float().to(device)).detach().cpu().numpy()
            valid_loader_enc = torch.utils.data.DataLoader(dataset(X_enc_val,y_test_bin),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=4)
            metrics_after = evaluate_model(classifier, valid_loader_enc, device, None)
        # Compare performance
        pr_auc_before = metrics_before["PR-AUC (Attack)"]
        pr_auc_after = metrics_after["PR-AUC (Attack)"]
        
        if pr_auc_after > pr_auc_before:
            print(f"Performance improved after training! PR-AUC increased from {pr_auc_before:.5f} to {pr_auc_after:.5f}")
        elif pr_auc_after < pr_auc_before:
            print(f"Performance decreased after training. PR-AUC decreased from {pr_auc_before:.5f} to {pr_auc_after:.5f}")
        else:
            print(f"Performance remained the same after training. PR-AUC: {pr_auc_before:.5f}")
    test_set_results = []
    with open(temp_filename, 'w') as fp:
        if type_selector == "cade":
            test_set_results.extend([testing(classifier=classifier,path= f"output/baseline/{type_selector}",training_cutoff=training_cutoff,type="seen" ,task_id=task_id ,seen_data=True,model=encoder),testing(classifier=classifier,path= f"output/baseline/{type_selector}",training_cutoff=training_cutoff, type="unseen" ,task_id=task_id ,seen_data=False,model=encoder),str(owl_self_labelled_count_class_0), str(owl_self_labelled_count_class_1),str(owl_analyst_labelled_count_class_0), str(owl_analyst_labelled_count_class_1),testing(classifier=classifier,path= f"output/baseline/{type_selector}",training_cutoff=len(task_order), type="all" ,task_id=task_id ,seen_data=True,model=encoder) ])
        elif type_selector == "pseudo-loss": 
            test_set_results.extend([testing(path= f"output/baseline/{type_selector}",training_cutoff=training_cutoff,type="seen" ,task_id=task_id ,seen_data=True,model=encoder),testing(path= f"output/baseline/{type_selector}",training_cutoff=training_cutoff, type="unseen" ,task_id=task_id ,seen_data=False,model=encoder),str(owl_self_labelled_count_class_0), str(owl_self_labelled_count_class_1),str(owl_analyst_labelled_count_class_0), str(owl_analyst_labelled_count_class_1),testing(path= f"output/baseline/{type_selector}",training_cutoff=len(task_order), type="all" ,task_id=task_id ,seen_data=True,model=encoder) ])
        auc_result[str(args.seed)] = test_set_results
        json.dump(auc_result, fp)  
def train_CL():
    global input_shape,student_model1
    # Step 1 : Load Tasks : Seen Task and Unseen Tasks
    train_order = task_order
    # Step 2 : Select The continual Learning Strategy 
    model_name = get_enum_from_string(method)
    loss_fn = torch.nn.CrossEntropyLoss()
    student_model1 = student_model1.float()
    startegy = Model_class(
            model_name=model_name,
            model=student_model1,
            optimizer=student_optimizer1,
            criterion=loss_fn,
            train_mb_size=64,
            eval_mb_size=64 ,
            device = device,
            epochs = 50
        )
    cl_strategy =startegy.load_model()
    for task_id,task in enumerate(train_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("loading task:",task_id)     
        input_shape,train_scenario = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=True,bool_encode_anomaly=False,label=label,bool_create_tasks_avalanche=True)
        print("Training task:",task_id)
        # print(train_scenario[1])
        # print(train_scenario[0])
        generic_scenario = create_avalanche_scenario([train_scenario[0],train_scenario[1]],task_labels=[0])
        print(train_scenario[0][0][0].shape)
        # Create a validation stream
        # Now you can pass this experience to the evaluation plugin or directly to the strategy
        # validation_stream = [validation_experience]
        # valid_loader = torch.utils.data.DataLoader(
        #             dataset(train_scenario[1][0][0], train_scenario[1][0][1]),
        #             batch_size=batch_size,
        #             num_workers=4
        #         )
        # train_loader = torch.utils.data.DataLoader(
        #             dataset(train_scenario[0][0][0], train_scenario[0][0][1]),
        #             batch_size=batch_size,
        #             num_workers=4
        #         )
        # test_loader =  torch.utils.data.DataLoader(
        #             dataset(train_scenario[2][0], train_scenario[3][0]),
        #             batch_size=batch_size,
        #             num_workers=4
        #         )
        for task_number, experience in enumerate(generic_scenario.train_stream):
                # print(experience)
                res = cl_strategy.train(experience,eval_streams=[generic_scenario.test_stream])
        student_model1.train()
    # testing(training_cutoff=training_cutoff,type="seen" ,task_id=task_id ,seen_data=True)
    test_set_results = []
    with open(temp_filename, 'w') as fp:
        test_set_results.extend([testing(training_cutoff=training_cutoff,type="seen" ,task_id=task_id ,seen_data=True),testing(training_cutoff=training_cutoff, type="unseen" ,task_id=task_id ,seen_data=False),str(owl_self_labelled_count_class_0), str(owl_self_labelled_count_class_1),str(owl_analyst_labelled_count_class_0), str(owl_analyst_labelled_count_class_1),testing(training_cutoff=len(task_order), type="all" ,task_id=task_id ,seen_data=True) ])
        auc_result[str(args.seed)] = test_set_results
        json.dump(auc_result, fp)
def create_avalanche_scenario(tasks_dataset,task_labels):
    # print(tasks_dataset[0],tasks_dataset[1])
    generic_scenario = tensors_benchmark(train_tensors = tasks_dataset[0],
                                           test_tensors = tasks_dataset[1],#tasks_dataset[1],
                                           task_labels = task_labels,
                                        )
    return generic_scenario  
              
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
        # print("validation y is : ", y_val)
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
    # testing(training_cutoff=training_cutoff,type="unseen" ,task_id=task_id ,seen_data=False,run_order="per_task",curr_train_task=task_id)   
    # testing(training_cutoff=training_cutoff, seen_data=True) 
    # testing(training_cutoff=training_cutoff, seen_data=False)    
    print(f'\nOpen world setting training from task {training_cutoff} onwards...')
    truth_agreement_fraction_0 = max(truth_agreement_fraction_0, 0.5)
    truth_agreement_fraction_1 = max(truth_agreement_fraction_1, 0.1)
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
        test_set_results.extend([testing(training_cutoff=training_cutoff,type="seen" ,task_id=task_id ,seen_data=True),testing(training_cutoff=training_cutoff, type="unseen" ,task_id=task_id ,seen_data=False),str(owl_self_labelled_count_class_0), str(owl_self_labelled_count_class_1),str(owl_analyst_labelled_count_class_0), str(owl_analyst_labelled_count_class_1),testing(training_cutoff=len(task_order), type="all" ,task_id=task_id ,seen_data=True) ])
        auc_result[str(args.seed)] = test_set_results
        json.dump(auc_result, fp)
    # print('\nOTDD values for consecutive tasks:')
    # for i, val in enumerate(consecutive_otdd):
    #     print(f'Task ({i},{i + 1}): {val}')
    # print()

    print('************** OWL labelling stats ****************')
    print(f'Total number of self-labelled samples = {owl_self_labelled_count_class_0 + owl_self_labelled_count_class_1}')
    print(f'Class 0 count = {owl_self_labelled_count_class_0}')
    print(f'Class 1 count = {owl_self_labelled_count_class_1}')

    print(f'Total number of analyst-labelled samples = {owl_analyst_labelled_count_class_0 + owl_analyst_labelled_count_class_1}')
    print(f'Class 0 count = {owl_analyst_labelled_count_class_0}')
    print(f'Class 1 count = {owl_analyst_labelled_count_class_1}')


class APIGRAPH_FC(nn.Module):
    def __init__(self,inputsize, softmax=False,dropout=0.2):
        super(APIGRAPH_FC, self).__init__()
        layers = []
        self.softmax=softmax
        self.act=OrderedDict()
        drop_out_ratio=dropout
        self.hidden1 = Linear(inputsize,100,bias=False)
        self.bn1 = nn.InstanceNorm1d(100)
        self.dropout1 = nn.Dropout(drop_out_ratio)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 250,bias=False)
        self.bn2 = nn.InstanceNorm1d(250)
        self.dropout2 = nn.Dropout(drop_out_ratio)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        self.hidden3 = Linear(250, 500,bias=False)
        self.bn3 = nn.InstanceNorm1d(500)
        self.dropout3 = nn.Dropout(drop_out_ratio)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden3.weight)
        self.act3 = ReLU()
        self.hidden4 = Linear(500, 150,bias=False)
        self.bn4 = nn.InstanceNorm1d(150)
        self.dropout4 = nn.Dropout(drop_out_ratio)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden4.weight)
        self.act4 = ReLU()
        self.hidden5 = Linear(150, 50,bias=False)
        self.bn5 = nn.InstanceNorm1d(50)
        self.dropout5 = nn.Dropout(drop_out_ratio)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = ReLU()
        self.hidden6 = Linear(50,2,bias=False)
        # Normalize weights
        # self.hidden6.weight.data = self.hidden6.weight.data / self.hidden6.weight.data.norm(dim=1, keepdim=True)
        # self.hidden6 = utils.weight_norm(self.hidden6, name='weight')
        kaiming_uniform_(self.hidden6.weight, nonlinearity='relu')
        self.act6 = ReLU()
        # self.act6 = Identity()

        # self.normedlayer= NormedLinear(50,2)
    def forward_encoder(self, X):
        X = X.float()
        self.act["hidden1"]  = X
        X = self.hidden1(X)
        X = self.bn1(X)
        X = self.act1(self.dropout1(X))
        self.act["hidden2"]  = X
        X = self.hidden2(X)
        X = self.bn2(X)
        X = self.act2(self.dropout2(X))
        self.act["hidden3"]  = X
        X = self.hidden3(X)
        X = self.bn3(X)
        X = self.act3(self.dropout3(X))
        self.act["hidden4"]  = X
        X = self.hidden4(X)
        X = self.bn4(X)
        X = self.act4(self.dropout4(X))
        self.act["hidden5"]  = X
        X = self.hidden5(X)
        X = self.bn5(X)
        X = self.act5(self.dropout5(X))
        self.act["hidden6"]  = X
        # X = 10*self.hidden6(F.normalize(X, dim=1))
        # X = 10*self.hidden6(F.normalize(X, dim=1))
        # X = self.act6(X)
        #Testing with normed last year
        #X = self.normedlayer(X)
        
        return X  # Encoded output

    def forward_classifier(self, encoded_X):
        # Normalize the encoded Output
        # norm_encoded=encoded_X/torch.norm(encoded_X,p=2,dim=1, keepdim=True)
        # # Normalize the Weights of the classifier
        # with torch.no_grad():
        #     weight_norm = torch.norm(self.hidden6.weight, p=2, dim=1, keepdim=True)
        #     normalized_weight = self.hidden6.weight / weight_norm  # Normalize the weights of hidden6
        # # Apply the normalized weights to the input
        # self.hidden6.weight.data = normalized_weight
        X = self.hidden6(encoded_X)
        if self.softmax:
            X = torch.softmax(encoded_X, dim=1).squeeze()
        return X

    def forward(self, X):
        encoded_X = self.forward_encoder(X)
        output = self.forward_classifier(encoded_X)
        return output
        
def testing(training_cutoff, task_id,type,run_order="",curr_train_task = 0,seen_data=False,path="output/new_method/",model=None,classifier=None):

    dataset_loadtime=0
    global teacher_model1,teacher_model2,teacher_supervised
    global student_model1,student_model2,student_supervised
    if model == None:
        if mlps == 1:
            models = [student_model1]
        elif mlps == 2:
            models = [student_model1,student_model2]
        else:
            models = [student_model1,student_model2,student_supervised]
    else :
        models=[model]
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
        print(f'testing on task {task_id}: {features.shape}')
        
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
        student_model1.eval()
        for data, target in valid_loader:
            class_probs = []
            with torch.no_grad():
                for m in models:
                    if model is not None:  # Skip models that are None
                        if type_selector == "cade":
                            data_enc = m.encode(data.to(device))
                            outputs = classifier.predict_proba(data_enc)
                        elif type_selector =="pseudo-loss":
                            outputs = m.predict_proba(data.to(device))  # Get predictions
                    else:
                        outputs = m(data.to(device))  # Get predictions
                        # Ensure outputs have at least two dimensions
                    if outputs.dim() == 1:  # Single sample case
                        outputs = outputs.unsqueeze(0)  # Add batch dimension

                        # Apply softmax along the correct dimension
                    outputs = torch.softmax(outputs, dim=1)
                    class_probs.append(outputs)

            # Stack the probabilities and compute the mean
            if len(class_probs) > 0:  # Ensure class_probs is not empty
                class_probs_tensor = torch.stack(class_probs).mean(dim=0)

                # Handle reshaping based on the target shape
                if class_probs_tensor.shape[0] == target.numel():  # Match batch size
                    pred = class_probs_tensor[:, 1].reshape(target.shape)
                else:
                    raise ValueError(
                        f"Mismatch in sizes: class_probs_tensor shape {class_probs_tensor.shape} "
                        f"does not align with target shape {target.shape}."
                    )
            else:
                raise ValueError("class_probs is empty. Ensure models are defined and provide valid outputs.")
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
        metric_results.append(compute_results_new(y_test=val_actual, lr_probs=np.array(val_pred),name=ds,seed=seed,task_id=task_id,type=type,run_order=run_order,curr_train_task =curr_train_task,path=path))
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

    print('\npnt table for SPIDER:')
    pnt_table = [
        # ['task_CI']+ task_CI_pnt, 
        # ['test_CI'] + test_CI_pnt,
        ['prauc Benign traffic'] + prauc_in_pnt, 
        ['prauc Attack traffic'] + prauc_out_pnt
    ]
    print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
    
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
        # train_baseline_hcl_good(supervised=is_supervised)
        train_CL()
        # taskwise_lazytrain()
        # plot_grdient_norm_line_graph()
        # X_test,y_test = np.concatenate( test_x, axis=0 ),np.concatenate( test_y, axis=0 )
    else:
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,class_ids,minorityclass_ids,tasks_list,task2_list,task_order,bool_encode_benign=False,bool_encode_anomaly=True,label=label,bool_create_tasks_avalanche=False)
        initialize_buffermemory(tasks=tasks,mem_size=memory_size)
        print('Total no.of tasks', len(tasks))
        # update_buffermemory_counter(memorysamples=memory_y_name)
        # update_mem_samples_indexdict(memorysamples=memory_y_name)
        train(tasks=tasks)
    print("Total execution time is--- %s seconds ---" % (time.time() - start_time))
    print("Total memory population time is--- %s seconds ---" % (memory_population_time))
def get_enum_from_string(model_name_str):
    """
    Converts a string to the corresponding MODELNAME enum.

    Args:
        model_name_str (str): The string representation of the model name.

    Returns:
        MODELNAME: The corresponding MODELNAME enum value.

    Raises:
        ValueError: If the input string does not match any enum.
    """
    try:
        return MODELNAME[model_name_str.upper()]
    except KeyError:
        raise ValueError(f"{model_name_str} is not a valid model name.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--ds', type=str, default="bodmas", metavar='S',help='dataset name')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)')
    parser.add_argument('--filename', type=str,default="temp", metavar='S',help='json file name')
    parser.add_argument('--b_m', type=float, default=0.2, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='S',help='batch memory ratio(default: 0.001)')
    parser.add_argument('--wd', type=float, default= 1e-3, metavar='S',help='batch memory ratio(default: 0.01)')
    parser.add_argument('--label_ratio', type=float, default=0.1, metavar='S',help='labeled ratio (default: 0.1)')
    parser.add_argument('--nps', type=int, metavar='S',default=10000,help='number of projection samples(default: 100)')
    parser.add_argument('--bma', type=float, metavar='S',default=0.3,help='batch minority allocation(default: 0.8)')
    parser.add_argument('--alpha', type=float, metavar='S',default=9,help='distill loss multiplier(default: 9)')
    parser.add_argument('--lab_samp_in_mem_ratio', type=float, metavar='S',default=0.1,help='Percentage of labeled samples to store in memory(default: 1.0)')
    parser.add_argument('--bool_gpm', type=str, metavar='S',default="True",help='Enables gradient projections(default: True)')
    parser.add_argument('--mem_strat', type=str, metavar='S',default="equal",help='Buffer memory strategy(default: full initialization)')
    parser.add_argument('--training_cutoff', type=int, default=5, metavar='S',help='train the model for first n tasks and test for time decay on the rest')
    parser.add_argument('--bool_closs', type=str, metavar='S',default="False",help='Enables using contrastive loss(default: False)')
    parser.add_argument('--mlps', type=int, metavar='S',default=1,help='Number of learners (MLPs)default: 1)')
    parser.add_argument('--cos_dist', type=float, metavar='S',default=0.3,help='cosine distance for OWL(default: 0.3)')
    parser.add_argument('--mode_val', type=int, metavar='S',default=99,help='Mode value for OWL (default: 99)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N', help='number of training epochs/task (default: 10)')
    parser.add_argument('--beta', type=float, metavar='S',default=0.1,help='hyperparameter for accumulation of agreement fraction')
    parser.add_argument('--analyst_labels', type=int, metavar='S',default=100,help='no of samples labelled by Analyst')
    parser.add_argument('--uncertainity', type=str, metavar='S',default="pseudo-loss",help='Sample selector')
    parser.add_argument('--family_info', type=bool, metavar='S',default=True,help='Include Family Info in HCL')
    parser.add_argument('--is_supervised',type=bool,metavar='S',default=True,help='Supervised or semi-supervised')
    parser.add_argument('--cl_method',type=str,metavar='S',default="AGEM",help='Continual learning method')
    args = parser.parse_args()
    set_seed(args.seed)
    get_gpu(args.gpu)
    print("seed is",args.seed)
    global analyst_labels,labels_ratio,no_of_rand_samples,l_rate,w_decay,batch_minority_allocation,b_m,alpha,lab_samp_in_mem_ratio,bool_gpm,mem_strat,temp_filename,auc_result,seed,bool_closs,mlps,training_cutoff, epochs, ds, beta,cos_dist_ip, mode_value
    epochs = args.n_epochs
    b_m = float(args.b_m)
    labels_ratio=float(args.label_ratio)
    no_of_rand_samples = int(args.nps)
    batch_minority_alloc = float(args.bma)
    alpha = float(args.alpha)
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
    ppt = 25
    ds = args.ds
    beta = args.beta
    type_selector = args.uncertainity
    family_info=args.family_info
    is_supervised=args.is_supervised
    method = args.cl_method
    print(args.uncertainity)
    print("{:<20}  {:<20}".format('Argument','Value'))
    print("*"*80)
    for arg in vars(args):
        print("{:<20}  {:<20}".format(arg, getattr(args, arg)))
    print("*"*80)    
    auc_result= {}
    temp_filename = str(args.filename)
    start_execution(args.ds,l_rate,w_decay)
    label_results={}
    with open(temp_filename+"labels", 'w') as fp:
        label_results[str(args.seed)] =[selector_analyst_attack,selector_analyst_benign]
        print(label_results)
        json.dump(label_results, fp)
    print("*"*80)

