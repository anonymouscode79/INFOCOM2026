from turtle import st
import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torch.optim.lr_scheduler import StepLR
from tabulate import tabulate
import torch.nn.functional as F

import pandas as pd
from math import ceil,floor
from tqdm import tqdm
import os

from utils.customdataloader import load_dataset,Tempdataset,compute_total_minority_testsamples,get_inputshape
from utils.buffermemory import cbrsmemory_update,retrieve_replaysamples
from utils.metrics import compute_results
from utils.utils import log,create_directories,trigger_logging,set_seed,get_gpu,load_model,EarlyStopping,GradientRejection
from utils.config.configurations import cfg
from utils.metadata import initialize_metadata


import time
import random
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
from collections import Counter
import json



from torchmetrics import Accuracy
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

import warnings
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.nn.init import kaiming_uniform_
from torch.nn import Module,Linear,ReLU,Sigmoid
warnings.filterwarnings("ignore")

from androzoo_data_set_info import androzoo
from api_graph_data_set_info import api_graph
from bodmas_data_set_info import bodmas
from ember_data_set_info import ember
def get_dataset_info_local(dataset):
    if dataset == "api_graph":
        return api_graph
    elif dataset=="androzoo":
        return androzoo
    elif dataset=="bodmas":
        return bodmas
    elif dataset=="ember":
        return ember


scaler = MinMaxScaler()

memory_population_time=0
global_priority_list=dict()
local_priority_list=dict()
local_count = Counter()
classes_so_far = set()
full = set()
local_store = {}
global_count, local_count, replay_count,replay_individual_count = Counter(), Counter(),Counter(),Counter()
input_shape,task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,learning_rate = None,None,None,None,None,None,None,None,None
replay_size,memory_size,minority_allocation,epochs,batch_size,device,pattern_per_exp,is_lazy_training,task_num = None,None,None,None,None,None,None,None,None
memory_X, memory_y, memory_y_name = None,None,None
model,opt,loss_fn,train_acc_metric = None,None,None,None
test_x,test_y = None,None
image_resolution = None
nc = 0
no_tasks = 0




class dataset(Dataset):

    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
 
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.length


def load_metadata(dataset_name,lr,w_d):
    # set_seed(125)
    # get_gpu()
    global task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,learning_rate,input_shape,image_resolution
    global replay_size,memory_size,minority_allocation,epochs,batch_size,device,pattern_per_exp,is_lazy_training
    # set_seed(125)
    # get_gpu()
    ds = get_dataset_info_local(dataset_name) 
    label = ds.label
    cfg.avalanche_dir = False
    set_cl_strategy_name(0)
    no_tasks = ds.no_tasks
    metadata_dict = initialize_metadata(label)
    temp_dict = metadata_dict[no_tasks]
    task_order = temp_dict['task_order']
    class_ids = temp_dict['class_ids']
    minorityclass_ids = temp_dict['minorityclass_ids']
    pth = temp_dict['path']
    tasks_list = temp_dict['tasks_list']
    task2_list = temp_dict['task2_list']
    replay_size = ds.replay_size
    memory_size = ds.mem_size
    minority_allocation = ds.minority_allocation
    epochs = 100 #ds.n_epochs
    batch_size = ds.batch_size
    device = cfg.device
    learning_rate = lr # ds.learning_rate
    no_tasks = ds.no_tasks
    image_resolution = ds.image_resolution
    pattern_per_exp = ds.pattern_per_exp
    is_lazy_training = ds.is_lazy_training
    input_shape = get_inputshape(pth,class_ids)
    compute_total_minority_testsamples(pth=pth,dataset_label=label,minorityclass_ids=minorityclass_ids,no_tasks=no_tasks)
    load_model_metadata(w_d)
    create_directories(label)
    trigger_logging(label=label)

def load_model_metadata(w_decay):
    log("loading model parameter")
    global model,opt,loss_fn,train_acc_metric,input_shape
    model = APIGRAPH_FC(inputsize=input_shape)
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=w_decay)
    loss_fn = torch.nn.BCELoss()
    train_acc_metric = Accuracy().to(device)

def set_cl_strategy_name(strategy_id):
    if strategy_id == 0:
        cfg.clstrategy = "CBRS"            
    elif strategy_id == 1:
        cfg.clstrategy = "ECBRS"
     
          

def initialize_buffermemory(tasks,mem_size):
    global memory_X, memory_y, memory_y_name
    initial_X, initial_y, initial_yname,_ = tasks[0]
    mem = min(mem_size, initial_yname.shape[0])    
    memory_X, memory_y, memory_y_name = initial_X[:mem,:], initial_y[:mem], initial_yname[:mem]

def update_buffermemory_counter(memorysamples):
    global local_count
    for class_ in memorysamples:
        local_count[class_]+=1

def update_exemplars_global_counter(samples):
    global global_count,classes_so_far,nc
    for j in range(len(samples)):
      global_count[samples[j]]+=1# global_count stores "class_name : no. of class_name instances in the stream so far"
      if samples[j] not in classes_so_far:
        classes_so_far.add(samples[j])
        nc += 1  


def update_replay_counter(binarymemorysamples,classwisememorysamples):
    global replay_count,replay_individual_count
    for b_class_,class_ in zip(binarymemorysamples,classwisememorysamples):
        replay_count[b_class_]+=1
        replay_individual_count[class_]+=1


def update_mem_samples_indexdict(memorysamples):
    global local_store
    for idx,class_ in enumerate(memorysamples):
        if class_ in local_store :
            local_store[class_].append(idx)
        else:
            local_store[class_] = [idx]

def get_balanced_testset(X,y):
    X_test,y_test = X,y
    bool_idx_list = list()
    no_of_ones= np.count_nonzero(y_test == 1)
    c=0
    for idx in range(X_test.shape[0]):
        if y_test[idx] == 0 and c <=no_of_ones:
            bool_idx_list.append(True)
            c+=1
        elif y_test[idx] == 0 and c >no_of_ones:
            bool_idx_list.append(False)
        else:
            bool_idx_list.append(True)
    X_test = X_test[bool_idx_list]
    y_test = y_test[bool_idx_list]

    return X_test,y_test




def ecbrs_train_epoch(Xt, yt,replay_Xt, replay_yt,a):
    model.train()
    # global model,opt
    stream_dataset = Tempdataset(Xt, yt)
    stream_train_dataloader = DataLoader(stream_dataset, batch_size=batch_size, shuffle=True)    

    replay_dataset = Tempdataset(replay_Xt, replay_yt)
    replay_train_dataloader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=True) 

    for step, (stream,replay) in enumerate(zip(stream_train_dataloader,replay_train_dataloader)):

      x_stream_train, y_stream_train = stream
      x_replay_train, y_replay_train = replay
      x_stream_train, y_stream_train = x_stream_train.to(device),y_stream_train.to(device)
      x_replay_train, y_replay_train = x_replay_train.to(device) , y_replay_train.to(device)            
      if image_resolution is not None:
          x_stream_train = x_stream_train.reshape(image_resolution)
          x_replay_train = x_replay_train.reshape(image_resolution)

          
    #   y_hat_stream = model(x_stream_train).reshape(y_stream_train.shape)
      y_hat_stream = torch.softmax(model(x_stream_train),dim=1).squeeze()
      loss_stream = loss_fn(y_hat_stream,F.one_hot(y_stream_train.to(dtype=torch.long), 2).float())
    #   loss_stream = loss_fn(y_hat_stream,y_stream_train)
      
    #   y_hat_replay = model(x_replay_train).reshape(y_replay_train.shape)
      y_hat_replay = torch.softmax(model(x_replay_train),dim=1).squeeze()
      loss_replay = loss_fn(y_hat_replay,F.one_hot(y_replay_train.to(dtype=torch.long), 2).float())
    #   loss_replay = loss_fn(y_hat_replay,y_replay_train)

      loss = a * loss_stream + (1-a) * loss_replay
            

      opt.zero_grad()
      loss.backward()
      opt.step()  
    # train_acc = train_acc_metric(y_hat_stream,y_stream_train.to(torch.int)).to(device)   
    y_pred,lab_y = [],[]
    y_pred.extend((y_hat_stream[:,1]).detach().cpu().numpy().tolist())
    y_pred.extend((y_hat_replay[:,1]).detach().cpu().numpy().tolist())
    lab_y.extend(y_stream_train.detach().cpu().numpy().tolist())
    lab_y.extend(y_replay_train.detach().cpu().numpy().tolist())

            
    lr_precision, lr_recall, _ = precision_recall_curve(lab_y, y_pred,pos_label=1)
    lr_auc_outlier =  auc(lr_recall, lr_precision)
    lr_precision, lr_recall, _ = precision_recall_curve(lab_y, [1-x for x in y_pred],pos_label=0)
    lr_auc_inliers =  auc(lr_recall, lr_precision)   
    return lr_auc_outlier,lr_auc_inliers,loss 


def train(tasks,X_val,y_val):
    global memory_X, memory_y, memory_y_name,local_count,global_count,local_store,input_shape,memory_size,task_num
    global classes_so_far,full,global_priority_list,local_priority_list,memory_population_time,replay_size
    task_id_temp = 0

    valid_loader = torch.utils.data.DataLoader(dataset(X_val,y_val),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=0)
    

    for X,y,y_classname,_ in tasks:
        
        if not is_lazy_training:
            task_num = task_id_temp
        
        task_id_temp+=1    

        # print("task number:",task_num)
        task_size = X.shape[0]
        no_of_batches = floor(task_size/replay_size)
        check_point_file_name = "checkpoint"+str(os.getpid())+".pt"
        check_point_file_name_norm = "checkpoint"+str(os.getpid())+"grad_norm"+".pt"
        early_stopping = EarlyStopping(patience=3, verbose=True,path=check_point_file_name)
        gradient_rejection = GradientRejection(patience=1, verbose=True,path=check_point_file_name_norm)
        scheduler = StepLR(opt, step_size=1, gamma=0.96)
        for epoch in range(epochs):
            # scheduler.step()
            prog_bar = tqdm(range(no_of_batches))
            for batch_idx in prog_bar:
                curr_batch = batch_idx*replay_size
                Xt, yt, ynamet = X[curr_batch:curr_batch+replay_size,:], y[curr_batch:curr_batch+replay_size], y_classname[curr_batch:curr_batch+replay_size]
                update_exemplars_global_counter(ynamet)
                a=1/nc
                replay_Xt,replay_yt,replay_yname = retrieve_replaysamples(memory_X, memory_y ,memory_y_name,global_priority_list,local_count,replay_size,input_shape,minority_allocation,memory_size)            
                update_replay_counter(binarymemorysamples=replay_yt,classwisememorysamples=replay_yname)   
                replay_Xt, Xt = replay_Xt.astype('float32'), Xt.astype('float32')   
                replay_yt,yt = replay_yt.astype('float32'), yt.astype('float32')
                pr_auc_o,pr_auc_i,loss = ecbrs_train_epoch(Xt=Xt,yt=yt,replay_Xt=replay_Xt,replay_yt=replay_yt,a=a)
                prog_bar.set_description('loss: {:.5f} -  PR-AUC(inliers): {:.2f} - PR_auc(outlier)_curve {:.3f}'.format(
                 loss.item(),  pr_auc_i,pr_auc_o ))
                mem_begin=time.time()
                memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list = cbrsmemory_update(Xt,yt,ynamet,task_num,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list)
                memory_population_time+=time.time()-mem_begin
            
            model.eval() # prep model for evaluation
            val_pred,val_gt = [],[]
            for data, target in valid_loader:
                if image_resolution is not None:
                    data = data.reshape(image_resolution)
                pred = torch.softmax(model(data.to(device)),dim=1)[:,1].reshape(target.shape)
                y_pred = pred.detach().cpu().numpy().tolist()
                val_pred.extend(y_pred)
                val_gt.extend(target.detach().cpu().numpy().tolist())
            lr_precision, lr_recall, _ = precision_recall_curve(val_gt, val_pred,pos_label=1)
            lr_auc =  auc(lr_recall, lr_precision)

            epoch_len = len(str(epochs))
            if no_of_batches !=0:
                print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                        f'train_loss: {loss:.5f} ' +
                        f'PR-AUC (O): {lr_auc:.5f}')
            
                print(print_msg)
            early_stopping(lr_auc, model)
            if early_stopping.counter < 1:
                scheduler.step()
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(check_point_file_name))
    if os.path.exists(check_point_file_name):
        os.remove(check_point_file_name)
    if os.path.exists(check_point_file_name_norm):
        os.remove(check_point_file_name_norm) 
        # for curr_batch in range(0, task_size, replay_size):
        #     print("till here", curr_batch+replay_size)
        #     Xt, yt, ynamet = X[curr_batch:curr_batch+replay_size,:], y[curr_batch:curr_batch+replay_size], y_classname[curr_batch:curr_batch+replay_size]
        #     print("Buffer memory",local_count)
        #     update_exemplars_global_counter(ynamet)        
        #     total_count=sum(global_count.values())
        #     a=1/nc    
        #     replay_Xt,replay_yt,replay_yname = retrieve_replaysamples(memory_X, memory_y ,memory_y_name,global_priority_list,local_count,replay_size,input_shape,minority_allocation,memory_size)            
        #     update_replay_counter(binarymemorysamples=replay_yt,classwisememorysamples=replay_yname)   
        #     replay_Xt, Xt = replay_Xt.astype('float32'), Xt.astype('float32')   
        #     replay_yt,yt = replay_yt.astype('float32'), yt.astype('float32')

        #     # for epoch in range(epochs):
        #     train_acc,loss = ecbrs_train_epoch(Xt=Xt,yt=yt,replay_Xt=replay_Xt,replay_yt=replay_yt,a=a)        
            
        
        #     print("Training acc over epoch: %.4f" % (float(train_acc),))
        #     print("loss over epoch: %.4f" % (float(loss),))
        

        #     mem_begin=time.time()
        #     memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list = cbrsmemory_update(Xt,yt,ynamet,task_num,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list)
        #     memory_population_time+=time.time()-mem_begin

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
def testing(seen_data=False,all_tasks=False):

    global training_cutoff

    dataset_loadtime=0
    
    task_CI_pnt = []
    test_CI_pnt =[]
    prauc_in_pnt = []
    prauc_out_pnt = []

    if seen_data:
        if all_tasks:
            testing_tasks = task_order
        else:
            testing_tasks = task_order[:training_cutoff]
        start_id = 0
    else:
        testing_tasks  = task_order[training_cutoff:]
        start_id = training_cutoff

    for task_id,task in enumerate(testing_tasks, start = start_id):
        
        task_class_ids = []
        task_minorityclass_ids = []
        
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        start = time.time()
        # input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=0,bool_encode_anomaly=1,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=True)
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=False,bool_encode_anomaly=False,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=False)
        
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
        
        val_pred, val_actual = [],[]
        for data, target in valid_loader: 
            with torch.no_grad():
                pred = torch.softmax(model(data.to(device)), dim=1)[:,1].reshape(target.shape)
            
            y_pred = pred.detach().cpu().numpy().tolist()

            val_pred.extend(y_pred)
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
        # precision, recall, thresholds = precision_recall_curve(val_actual, en_val_pred)
        # en_auc_precision_recall_1 = auc(recall, precision)

        precision, recall, thresholds = precision_recall_curve(val_actual, [1-val for val in val_pred], pos_label=0.)
        auc_precision_recall_0 = auc(recall, precision)
        # precision, recall, thresholds = precision_recall_curve(val_actual, [1-val for val in en_val_pred], pos_label=0.)
        # en_auc_precision_recall_0 = auc(recall, precision)

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
        # en_prauc_in_pnt.append(en_auc_precision_recall_0)
        # en_prauc_out_pnt.append(en_auc_precision_recall_1)
        
        # print(f'prauc inliers: {auc_precision_recall_in}')        
        # print(f'prauc outliers: {auc_precision_recall_out}')                     
        # print('')
    
    N = len(testing_tasks) #number of test tasks

    if N<2:
        print('not printing AUT values since it requires atleast 2 test tasks')
        return [prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N]
    
    prauc_in_aut  = 0
    prauc_out_aut = 0
    for i in range(N-1):
        prauc_in_aut+= (prauc_in_pnt[i]+prauc_in_pnt[i+1])/(2)
        prauc_out_aut+=(prauc_out_pnt[i]+prauc_out_pnt[i+1])/(2)
    prauc_in_aut  = prauc_in_aut/(N-1)
    prauc_out_aut = prauc_out_aut/(N-1)
    
    print(f'AUT(prauc inliers,{N}) := {prauc_in_aut}')
    print(f'AUT(prauc outliers,{N}) := {prauc_out_aut}')

    print('\npnt table:')
    pnt_table = [
        # ['task_CI']+ task_CI_pnt, 
        # ['test_CI'] + test_CI_pnt,
        ['prauc Benign traffic'] + prauc_in_pnt, 
        ['prauc Attack traffic'] + prauc_out_pnt
    ]
    print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
    
    print(f'dataset loading time: {dataset_loadtime}s\n')
    return [prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N]


def taskwise_lazytrain():
    global test_x,test_y,task_num
    val_X,val_y = [],[]
    # random.shuffle(task_order)
    for task_id,task in enumerate(task_order):
        task_class_ids = []
        task_minorityclass_ids = []
        print(task_id)
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("loading task:",task_id)     
        input_shape,tasks,X_test,y_test,X_val,y_val = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=True,bool_encode_anomaly=False,label=label,bool_create_tasks_avalanche=False)
        test_x.extend([X_test])
        test_y.extend([y_test])
        val_X.extend([X_val])
        val_y.extend([y_val])
        print("Training task:",task_id)
        task_num = task_id
        if task_id == int(0):
            initialize_buffermemory(tasks=tasks,mem_size=memory_size)
            update_buffermemory_counter(memorysamples=memory_y_name)
            update_mem_samples_indexdict(memorysamples=memory_y_name)
        # train(tasks,X_val,y_val)
        train(tasks,np.concatenate( val_X,axis=0),np.concatenate(val_y,axis=0))
    
    test_set_results = []
    with open(temp_filename, 'w') as fp:
        test_set_results.extend([testing(seen_data=True),testing(seen_data=False),testing(seen_data=True,all_tasks=True)])
        auc_result[str(args.seed)] = test_set_results
        json.dump(auc_result, fp)
# def evaluate_on_testset():
#     global X_test,y_test
#     # X_test,y_test = get_balanced_testset(X=X_test,y=y_test)
#     X_test = torch.from_numpy(X_test.astype(float)).to(device)
#     model.eval()
#     yhat = model(X_test.float()).detach().cpu().numpy()
#     compute_results(y_test,yhat)
#     print("test sample counters are",Counter(y_test))
#     print("Replayed samples are:",replay_individual_count)
#     print("Replayed samples are:",replay_count)

def evaluate_on_testset():
    global X_test,y_test
    
    yhat = None
    model.eval()
    print("computing the results")
    offset = 250000
    for idx in range(0,X_test.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float)).to(device)
        
        if image_resolution is not None:
          X_test1 = X_test1.reshape(image_resolution)
        temp = model(X_test1.float()).detach().cpu().numpy()
        if idx1==0:
            yhat = temp
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
    return compute_results(y_test,yhat)    



def start_execution(dsname,lr,w_decay):
    global input_shape,tasks,X_test,y_test,test_x,test_y
    start_time=time.time()
    load_metadata(dsname,lr,w_decay)
    # load_model_metadata()
    print(model)
    if is_lazy_training:
        test_x,test_y = [],[]
        taskwise_lazytrain()
        X_test,y_test = np.concatenate( test_x, axis=0 ),np.concatenate( test_y, axis=0 )
        

    else:
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,class_ids,minorityclass_ids,tasks_list,task2_list,task_order,bool_encode_benign=True,bool_encode_anomaly=True,label=label,bool_create_tasks_avalanche=False)
        initialize_buffermemory(tasks=tasks,mem_size=memory_size)
        print('Total no.of tasks', len(tasks))
        update_buffermemory_counter(memorysamples=memory_y_name)
        update_mem_samples_indexdict(memorysamples=memory_y_name)
        train(tasks=tasks)
    print("Total execution time is--- %s seconds ---" % (time.time() - start_time))
    print("Total memory population time is--- %s seconds ---" % (memory_population_time))





# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
#     parser.add_argument('--gpu', type=int, default=0, metavar='S',help='random seed (default: 0)')
#     args = parser.parse_args()
#     set_seed(args.seed)
#     get_gpu(args.gpu)
#     print("seed is",args.seed)

#     start_execution()
#     print("seed is",args.seed)
#     evaluate_on_testset()

if __name__ == "__main__":
    s_time = time.time()
    import argparse
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--ds', type=str, default="androzoo", metavar='S',help='dataset name')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)')
    parser.add_argument('--filename', type=str,default="temp", metavar='S',help='json file name')
    parser.add_argument('--lr', type=float, default=0.01, metavar='S',help='learning rate(default: 0.001)')
    parser.add_argument('--wd', type=float, default=1e-6, metavar='S',help='weight decay (default: 0.001)')
    parser.add_argument('--training_cutoff', type=int, default=4, metavar='S',help='train the model for first n tasks and test for time decay on the rest')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch_size')
    

    args = parser.parse_args()
    set_seed(args.seed)
    get_gpu(args.gpu)
    print("seed is",args.seed)
    global lr,w_decay,temp_filename,auc_result,seed,training_cutoff
    lr = float(args.lr)
    w_decay=float(args.wd)
    training_cutoff = args.training_cutoff
    seed = args.seed
    ppt = 50
    print("{:<20}  {:<20}".format('Argument','Value'))
    print("*"*80)
    for arg in vars(args):
        print("{:<20}  {:<20}".format(arg, getattr(args, arg)))
    print("*"*80)    
    auc_result= {}
    temp_filename = str(args.filename)    
    start_execution(args.ds,lr,w_decay)
    print("seed is",args.seed)
    
    tot_time = time.time()-s_time   
    # with open(temp_filename, 'w') as fp:
    #     test_set_results = evaluate_on_testset()
    #     test_set_results.extend([tot_time,memory_population_time])
    #     auc_result[str(args.seed)] = test_set_results
    #     # print("dict",auc_result)
    #     json.dump(auc_result, fp)
    
    # print("*"*80)
    
    
        
