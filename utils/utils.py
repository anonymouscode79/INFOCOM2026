
from pickle import TRUE
import torch
import numpy as np
from torch.nn.utils import prune
from torch.nn.functional import cosine_similarity
from torchmetrics.functional import pairwise_cosine_similarity
import matplotlib.pyplot as plt
from numpy import dot
from torch.utils.data import TensorDataset
from numpy.linalg import norm
# from utils.otdd.ot_distance import compute_ot_distance,compute_otdd_tabular_datasets
import torchvision
from sklearn import svm
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN
from sklearn.mixture import GaussianMixture

import random
import logging
# from imp import reload
# reload(logging)
import os
import time
import pprint
import math

from utils.config.configurations import cfg
from utils.classifiers import *





# import numpy as np
# import torch


def clustering_class_imbalance(data):
    print("computing for DBScan")
    start_time = time.time()
    # Step 1: Apply DB Scan
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(data)
    print("total time",time.time()-start_time)

    print("computing for K-means")
    start_time = time.time()
    # Step 1: Apply K-Means
    kmeans = KMeans(n_clusters=2, random_state=42,n_init=100)
    kmeans_labels = kmeans.fit_predict(data)
    print("total time",time.time()-start_time)

    print("computing for GMM")
    start_time = time.time()
    # Step 2: Apply Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=2, random_state=42,covariance_type="diag")
    gmm_labels = gmm.fit_predict(data)
    print("total time",time.time()-start_time)
    
    # print("computing for Hierarchical clustering")
    # start_time = time.time()
    # # Step 3: Apply Hierarchical Clustering
    # hierarchical = AgglomerativeClustering(n_clusters=2)
    # hierarchical_labels = hierarchical.fit_predict(data)
    # print("total time",time.time()-start_time)

    # Step 4: Function to calculate imbalance ratio
    def calculate_imbalance_ratio(labels):
        class_counts = np.bincount(labels)
        if len(class_counts) < 2:
            return float('inf')  # If only one class is found
        majority_class = max(class_counts[0],class_counts[1])
        minority_class = min(class_counts[0],class_counts[1])
        imbalance_ratio = minority_class / (majority_class+minority_class) # if class_counts[1] > 0 else float('inf')
        return imbalance_ratio

    # Step 5: Calculate imbalance ratios
    dbscan_ratio = calculate_imbalance_ratio(dbscan_labels)
    print("DBScan CIR",dbscan_ratio)
    kmeans_ratio = calculate_imbalance_ratio(kmeans_labels)
    print("kmeans CIR",kmeans_ratio)
    gmm_ratio = calculate_imbalance_ratio(gmm_labels)
    print("GMM CIR",gmm_ratio)
    # hierarchical_ratio = calculate_imbalance_ratio(hierarchical_labels)
    # print("HC CIR",hierarchical_ratio)

    # Step 6: Calculate mean of the imbalance ratios
    # mean_imbalance_ratio = np.mean([kmeans_ratio, gmm_ratio])
    mean_imbalance_ratio = np.mean([kmeans_ratio, gmm_ratio, dbscan_ratio])

    return mean_imbalance_ratio

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0.01, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score1 = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss_min1 = np.Inf
        self.val_loss_min2 = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    #     score1 = val_loss[0]
    #     score2 = val_loss[1]

    #     if self.best_score1 is None:
    #         self.best_score1 = score1
    #         self.best_score2 = score2
    #         self.save_checkpoint(val_loss, model)
    #     elif score1 < self.best_score1 + self.delta and score2 > self.best_score2 + self.delta:
    #         self.counter += 1
    #         self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
    #         if self.counter >= self.patience:
    #             self.early_stop = True
    #     else:
    #         self.best_score1 = score1
    #         self.best_score2 = score2
    #         self.save_checkpoint(val_loss, model)
    #         self.counter = 0    

    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         self.trace_func(f'Validation PR-AUC (inliers) increased ({self.val_loss_min1:.6f} --> {val_loss[0]:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), self.path)
    #     self.val_loss_min1 = val_loss[0]
    #     self.val_loss_min2 = val_loss[1]


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation PR-AUC (inliers) increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class GradientRejection:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=1e-6, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self,  model):

        score = self.compute_gradientnorm(model)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
    
    def compute_gradientnorm(self,model):
        grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
        norm = torch.cat(grads).norm().detach().cpu().numpy().item()

        return norm
    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     self.trace_func(f'Validation PR-AUC (inliers) increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        # self.val_loss_min = val_loss        

def create_directories(label):
    output_root_dir = cfg.root_outputdir
    cl_strategy = cfg.clstrategy
    param_weights_dir_MIR = cfg.param_weights_dir_MIR
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    if not os.path.exists(param_weights_dir_MIR):
        os.makedirs(param_weights_dir_MIR)    
    timestamp = time.strftime("%d_%b_%H_%M_%S")  
    
    output_dir = output_root_dir +'/'+label+'/'+str(cl_strategy)+'/'+timestamp 
    if cfg.avalanche_dir:
        output_dir = output_root_dir +'/'+label+'/'+'avalanche'+'/'+str(cl_strategy)+'/'+timestamp
    cfg.outputdir = output_dir
    cfg.timestamp = timestamp 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/models')
        os.makedirs(output_dir + '/encoded_models')
        os.makedirs(output_dir + '/pickles')
        os.makedirs(output_dir + '/logs')
        os.makedirs(output_dir + '/plots')
        os.makedirs(output_dir + '/weights')
        os.makedirs(output_dir + '/metrics')

        # os.makedirs(output_dir + '')


def log(message, print_to_console=True, log_level=logging.DEBUG):
    if log_level == logging.INFO:
        logging.info(message)
    elif log_level == logging.DEBUG:
        logging.debug(message)
    elif log_level == logging.WARNING:
        logging.warning(message)
    elif log_level == logging.ERROR:
        logging.error(message)
    elif log_level == logging.CRITICAL:
        logging.critical(message)
    else:
        logging.debug(message)

    if print_to_console:
        print(message)             




def trigger_logging(label):
    output_root_dir = cfg.root_outputdir
    log_dir = output_root_dir+'/'+label+'/'+cfg.timestamp+'/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("time stamp is:",cfg.timestamp)    

    logging.basicConfig(filename=log_dir + '/'+cfg.timestamp + '.log', level=logging.DEBUG,force=True,
                        format='%(levelname)s:\t%(message)s')

    # log(pprint.pformat(cfg))    



def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n




def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    # torch.use_deterministic_algorithms(True)
    # torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def get_gpu(id):
    # gpu_list = cfg.gpu_ids.split(',')
    # print("gpu list",gpu_list)
    # gpus = [int(iter) for iter in gpu_list]
    cfg.device = torch.device('cuda:' + str(id)) 


model_path={'byol_imagenet':'./clear10/pretrain_weights/features/byol_imagenet/state_dict.pth.tar',
			'imagenet':	'./clear10/pretrain_weights/features/imagenet/state_dict.pth.tar',
			'moco_b0':'./clear10/pretrain_weights/features/moco_b0/state_dict.pth.tar',
			'moco_imagenet':'./clear10/pretrain_weights/features/moco_imagenet/state_dict.pth.tar'
            }   

def compute_cosine_sim(X,device):
    avg_cos_sim_vec = [0] * X.shape[0]
    threshold = 1000
    for idx in range(0,X.shape[0]):
        if X.shape[0] < threshold:
            sim = cosine_similarity(torch.from_numpy(X),torch.from_numpy(X[idx,:])).detach().cpu().numpy() 
        else:
            indicies = np.random.choice(X.shape[0],size = threshold,replace=False) 
            sim = cosine_similarity(torch.from_numpy(X[indicies,:]),torch.from_numpy(X[idx,:])).detach().cpu().numpy()    
        avg_cos_sim_vec[idx] = 1-np.average(sim)
        # print("Cosine si is",avg_cos_sim_vec[idx])

    return avg_cos_sim_vec    




def obtain_grad_vector(model,numpy_array):

    temp_list = []
    for param in model.parameters():
        temp_list.append(param.grad.view(-1))
    grads = torch.cat(temp_list).cpu().numpy().reshape(1,-1)
    
    if numpy_array is None:
        numpy_array = grads
    else:
        numpy_array = np.concatenate((numpy_array,grads), axis=0)    
 
    return numpy_array


def plot_cosine_sim(array1,array2,dir):
    cos_array = []
    # cos_array = cosine_similarity(torch.from_numpy(array1),torch.from_numpy(array2)).detach().cpu().numpy()
    for idx in range(0,array1.shape[0]):
        cos_sim = dot(array1[idx,:], array2[idx,:])/((norm(array1[idx,:])*norm(array2[idx,:])))
        cos_array.append(cos_sim)
        # print(cos_sim)
    os.makedirs(dir,exist_ok=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis('on')
    # matplotlib.rcParams.update({'font.size': 18})
    plt.legend(prop={'size': 18})
    plt.title("Cosine Similarity bw gradients") 
    plt.xlabel("Batch num")
    # plt.yticks(np.arange(-0.002, 1, 0.02))
    plt.ylabel("Cosine similarity")
    plt.figure(figsize=(10,6))
    
    plt.plot(range(0,len(cos_array)), cos_array, color ="red")
    plt.savefig(dir+'/'+'cosine_sim_grads.pdf')
    # plt.show()


def check_grad_exist(model):

    for param in model.parameters():
        if param.grad is None:
            return False 


    return True

def dataset_from_numpy(X, Y, classes = None):
    targets =  torch.LongTensor(list(Y))
    ds = TensorDataset(torch.from_numpy(X).type(torch.FloatTensor),targets)
    ds.targets =  targets
    ds.classes = classes if classes is not None else [i for i in range(len(np.unique(Y)))]
    return ds








def load_model(label,inputsize,softmax=False,drop_out=0.2):
    model = None
    if label == 'androzoo':
        model = APIGRAPH_FC(inputsize=inputsize, softmax=softmax,dropout=drop_out)
    elif label == 'api_graph':
        model = APIGRAPH_FC(inputsize=inputsize, softmax=softmax)
    elif label == 'bodmas':
        model = APIGRAPH_FC(inputsize=inputsize, softmax=softmax)
    elif label == 'ember':
        model = APIGRAPH_FC(inputsize=inputsize, softmax=softmax)

    return model    


