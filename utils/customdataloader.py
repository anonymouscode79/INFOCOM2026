import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from math import ceil
from collections import Counter

from sys import getsizeof as size
import torch.nn.functional as F

# train_set_x,train_set_y,test_set_x,test_set_y,val_set_x,val_set_y = dict(),dict(),dict(),dict(),dict(),dict() 
train_set_x,train_set_y,train_set_y_bin,test_set_x,test_set_y,test_set_y_bin,val_set_x,val_set_y,val_set_y_bin = None,None,None,None,None,None,None,None,None
samples_per_majority_class = 0


def initialize_dict():
    global train_set_x,train_set_y,train_set_y_bin,test_set_x,test_set_y,test_set_y_bin,val_set_x,val_set_y,val_set_y_bin,total_minority_test_samples
    train_set_x,train_set_y,train_set_y_bin,test_set_x,test_set_y,test_set_y_bin,val_set_x,val_set_y,val_set_y_bin = dict(),dict(),dict(),dict(),dict(),dict() ,dict(),dict(),dict() 
    # total_minority_test_samples = 0

def compute_total_minority_testsamples(pth,dataset_label,minorityclass_ids,no_tasks):
    total_minority_test_samples = 0
    global samples_per_majority_class
    for name in minorityclass_ids:
        name1 = name+'.npy'
        # print("loading "+ str(name)+ ".npy file")
        arr = np.load(pth+name1,allow_pickle=True) # loads the individual data for each class
        X_im,y_im = extract_x_y(dataset_label,name,arr)
        arr_x = X_im.astype('float32')
        class_length = arr_x.shape[0] # no.of instances per class
        #print(class_length)
        train_split = int(0.75*class_length)      
        total_minority_test_samples+=arr_x[train_split:,:].shape[0]
    samples_per_majority_class = ceil(total_minority_test_samples/int(no_tasks))  
    # print("total minority samples:",total_minority_test_samples)
    # print("total majort samples per task:",samples_per_majority_class)   





class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs = sample
        inputs = np.array(inputs)
        #targets = np.array([targets])
        return torch.from_numpy(inputs)#.type(torch.HalfTensor)

def extract_x_y(dataset_label,class_label,arr):
    if len(arr) ==0 :
        return np.array([[]]),np.array([[]])
    X_im= np.array(arr[:,:-1]) #copying the features
    y_im = arr[:,-1] 
    return X_im,y_im    

def get_inputshape(path,class_ids,dataset_label=None):
    for name in class_ids:
        name1 = name+'.npy'
        arr = np.load(path+name1,allow_pickle=True)
        break
    return arr.shape[1]-1   

def load_teset(path,class_ids,label):
    X_test,y_test = None,None
    for idx,name in enumerate(class_ids):
        name1 = name+'.npy'
        arr = np.load(path+name1,allow_pickle=True)
        X_im,y_im = extract_x_y(label,name,arr)
        if idx == 0:
            X_test,y_test = X_im,y_im
        else:
            X_test = np.concatenate((X_test, X_im), axis=0)
            y_test = np.concatenate((y_test, y_im), axis=0)    
        # print("loading "+ str(name)+ ".npy file")
    return X_test,y_test    




def load_dataset(path,class_ids,minorityclass_ids,benignclass_ids,benignclass_ids_str,task_order,bool_encode_benign,bool_encode_anomaly,label,bool_create_tasks_avalanche,shuffle=True,load_whole_train_data=False):
    # arrs,X_im,y_im,X_image = dict(),dict(),dict(),dict()
    initialize_dict()
    global train_set_x,train_set_y,test_set_x,test_set_y
    for name in minorityclass_ids:
        name1 = name+'.npy'
        # print("loading "+ str(name)+ ".npy file")
        arr = np.load(path+name1,allow_pickle=True) # loads the individual data for each class
        arr_family = np.load(path+name+"_labels.npy",allow_pickle=True)
        X_im,y_im = extract_x_y(label,name,arr)
        input_shape=arr.shape[1]-1
        traindata_split_minority(X_im,y_im,arr_family,name,load_whole_train_data=load_whole_train_data)        
    # print("total minority test samples",total_minority_test_samples)
    majorityclass_ids = [class_ for class_ in class_ids if class_ not in minorityclass_ids]
    for name in majorityclass_ids:
        name1 = name+'.npy'
        # print("loading "+ str(name)+ ".npy file")
        arr = np.load(path+name1,allow_pickle=True) # loads the individual data for each class
        X_im,y_im = extract_x_y(label,name,arr)            
        input_shape=arr.shape[1]-1
        traindata_split_majority(X_im,y_im,name,load_whole_train_data=load_whole_train_data) 
    # print("creating tasks")  
    if bool_create_tasks_avalanche:
        multiclass_labels_to_binary(class_labels=class_ids,benignclass_ids=benignclass_ids)
        train_set_x,train_set_y,test_set_x,test_set_y = create_tasks_avalanche(task_order=task_order)
        return input_shape,[train_set_x,train_set_y,test_set_x,test_set_y]
    else:
        tasks = create_tasks(task_order,benignclass_ids_str,bool_encode_benign,bool_encode_anomaly,shuffle)
        test_val_set_binary_label(class_ids,benignclass_ids)
        return input_shape,tasks,test_set_x,test_set_y_bin,val_set_x,val_set_y_bin
def traindata_split_minority(data_x,data_y,arr_family,name,load_whole_train_data=False):
    global train_set_x,train_set_y,test_set_x,test_set_y,val_set_x,val_set_y_bin,total_minority_test_samples
    # total_minority_test_samples = 0
    arr_x = data_x.astype('float32')
    assert not np.isnan(arr_x).any() , "Nan detected in splig_minority"
    arr_y_bin = data_y.astype('float32')
    arr_family = arr_family.astype('float32')
    class_length = arr_x.shape[0] # no.of instances per class
    #print(class_length)
    train_split_rate = 0.85
    if load_whole_train_data:
        train_split_rate = 0.99
    train_split = int(train_split_rate*class_length) 
    test_set_x[name] = arr_x[train_split:,:]
    test_set_y_bin[name] = arr_y_bin[train_split:]
    test_set_y[name]= arr_family[train_split:]
    temp_x = arr_x[:train_split,:]
    temp_y_bin = arr_y_bin[:train_split]
    temp_y_family  = arr_family[:train_split]
    train_split = int(0.882352941*temp_x.shape[0])
    val_set_x[name] = temp_x[train_split:,:]
    val_set_y_bin[name] = temp_y_bin[train_split:]
    val_set_y[name]=temp_y_family[train_split:]
    train_set_x[name] = temp_x[:train_split,:]
    train_set_y_bin[name] = temp_y_bin[:train_split]
    train_set_y[name] = temp_y_family[:train_split]
    # total_length = train_set_x[name].shape[0] + test_set_x[name].shape[0] + val_set_x[name].shape[0]
    # print(name,class_length,train_set_x[name].shape[0],test_set_x[name].shape[0] , val_set_x[name].shape[0],total_length)


def traindata_split_majority(data_x,data_y,name,load_whole_train_data=False):
    global train_set_x,train_set_y,test_set_x,test_set_y,val_set_x,val_set_y,total_minority_test_samples
    data_x = data_x.astype('float32')
    assert not np.isnan(data_x).any() , "Nan detected in split_majority"
    data_y = data_y.astype('float32')
    arr_x = data_x
    arr_y_bin = data_y
    arr_family = np.array([0]*arr_x.shape[0],dtype='float32')
    class_length = arr_x.shape[0] # no.of instances per class
    # sample_ratio = (class_length-samples_per_majority_class)/class_length
    # print("sample ratio",sample_ratio)
    train_split_rate = 0.75 
    if load_whole_train_data:
        train_split_rate = 0.99
    train_split = int(train_split_rate*class_length)
    test_set_x[name] = arr_x[train_split:,:]
    test_set_y_bin[name] = arr_y_bin[train_split:]
    test_set_y[name]= arr_family[train_split:]
    temp_x = arr_x[:train_split,:]
    temp_y_bin = arr_y_bin[:train_split]
    temp_y_family  = arr_family[:train_split]
    train_split = int(0.95*temp_x.shape[0])
    val_set_x[name] = temp_x[train_split:,:]
    val_set_y_bin[name] = temp_y_bin[train_split:]
    val_set_y[name]=temp_y_family[train_split:]
    train_set_x[name] = temp_x[:train_split,:]
    train_set_y_bin[name] = temp_y_bin[:train_split]
    train_set_y[name] = temp_y_family[:train_split]
    # total_length = train_set_x[name].shape[0] + test_set_x[name].shape[0] + val_set_x[name].shape[0]
    # print(name,class_length,train_set_x[name].shape[0],test_set_x[name].shape[0] , val_set_x[name].shape[0],total_length)



def get_balanced_testset(X,y):
    # from numpy.random import RandomState
    # prng = RandomState(1234567890)
    X_test,y_test = X,y
    np.random.seed(42)    
    shuffler = np.random.permutation(len(y_test))
    print(shuffler)
    X_test = X_test[shuffler]
    y_test = y_test[shuffler]
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
def multiclass_labels_to_binary(class_labels,benignclass_ids):
    for class_label in class_labels:
        # print("class label",class_label)
        train_labels_shape = train_set_y[class_label].shape
        test_labels_shape = test_set_y[class_label].shape
        val_labels_shape = val_set_y[class_label].shape

        if int(class_label) in benignclass_ids:
            # print("obtained label",0)
            train_set_y[class_label] = np.zeros(train_labels_shape,dtype=float)
            test_set_y[class_label] = np.zeros(test_labels_shape,dtype=float)
            val_set_y[class_label] = np.zeros(val_labels_shape,dtype=float)
        else:
            # print("obtained label",1)
            train_set_y[class_label] = np.ones(train_labels_shape,dtype=float)
            test_set_y[class_label] = np.ones(test_labels_shape,dtype=float)
            val_set_y[class_label] = np.ones(val_labels_shape,dtype=float)

        train_set_y[class_label] = train_set_y[class_label].reshape(train_set_y[class_label].shape[0],1)    
        test_set_y[class_label] = test_set_y[class_label].reshape(test_set_y[class_label].shape[0],1)    
        val_set_y[class_label] = val_set_y[class_label].reshape(val_set_y[class_label].shape[0],1)    

         
    # print((train_set_y[class_label].shape)) 
def test_val_set_binary_label(class_ids,benignclass_ids):
    global test_set_x,test_set_y_bin,val_set_x,val_set_y_bin
    for i,name in enumerate(class_ids):
        X_temp, y_temp = test_set_x[name],test_set_y_bin[name]
        X_temp1, y_temp1 = val_set_x[name],val_set_y_bin[name]
        if i==0:
            test_x, test_y_bin = X_temp, y_temp #ensures initialisation of X_test and y_test
            val_x, val_y_bin = X_temp1, y_temp1
        else:
            test_x = np.concatenate((test_x,X_temp), axis=0)
            test_y_bin = np.concatenate((test_y_bin,y_temp), axis=0)
            val_x = np.concatenate((val_x,X_temp1), axis=0)
            val_y_bin = np.concatenate((val_y_bin,y_temp1), axis=0)
    test_y_bin = multiclass_to_twoclass(test_y_bin,benignclass_ids)   
    val_y_bin = multiclass_to_twoclass(val_y_bin,benignclass_ids)  
    test_set_x,test_set_y_bin,val_set_x,val_set_y_bin = test_x,test_y_bin,val_x,val_y_bin 
    # return train_set_x,train_set_y,test_x,test_y,val_x,val_y
def multiclass_to_twoclass(labels,benignclass_ids):
    labels = labels.ravel()
    for i in range(labels.shape[0]):
        # For 2-Class setting use the below 4 lines
        if labels[i] in benignclass_ids:
            labels[i] = 0
        else:
            labels[i] = 1 
    return labels.astype(float)  

def create_tasks(task_order,benignclass_ids,bool_encode_benign,bool_encode_anomaly,shuffle=True):
    tasks=[]
    for task in task_order:
        # print("creating the task",task)
        for i,class_ in enumerate(task):
            X_class, y_classname = train_set_x[class_], train_set_y_bin[class_]
            class_size = X_class.shape[0]
            if str(class_) in benignclass_ids:
                class_encoding = [0]
                y_family= train_set_y[class_]
                if bool_encode_benign:
                    y_classname = np.array([199]*class_size)
            else:
                class_encoding = [1]
                y_family= train_set_y[class_]
                if bool_encode_anomaly:
                    y_classname = np.array([299]*class_size)
            y_class = [class_encoding]*class_size 
            y_class = np.array(y_class)  
            if i==0:
                X_task, y_task, yname_task,y_family_task = X_class, y_class, y_classname,y_family
            else:
                X_task = np.concatenate((X_task, X_class), axis=0)
                y_task = np.concatenate((y_task, y_class), axis=0)
                yname_task = np.concatenate((yname_task, y_classname), axis=0)
                y_family_task = np.concatenate((y_family_task,y_family),axis=0)
        if shuffle:
            shuffler = np.random.permutation(len(y_task))
            X_task = X_task[shuffler]
            y_task = y_task[shuffler]
            y_family_task = y_family_task[shuffler]
            yname_task = yname_task[shuffler] 
        y_task = y_task.ravel()
        y_family_task = y_family_task.ravel()
        tasks.append((X_task, y_task, yname_task,y_family_task))
    return tasks

def normalize_torchtabulardata(p_tensor):
    mu = torch.mean(p_tensor,dim=1,keepdim=True)
    sd = torch.std(p_tensor,dim=1,keepdim=True)
    normalized_res = (p_tensor - mu)/sd

    return normalized_res



def create_tasks_avalanche(task_order,shuffle=True):
    tasks=[]
    final_train_data_x = []
    final_train_data_y = []
    final_test_data_x = []
    final_test_data_y = []
    exp_train_first_structure = []
    exp_val_first_structure = []
    exp_test_first_structure = []
    

    # s1,s2,s3,s4 =0,0,0,0
    for task in task_order:
        temp_train_data_x = torch.Tensor([])
        temp_train_data_y = torch.Tensor([])
        temp_test_data_x = torch.Tensor([])
        temp_test_data_y = torch.Tensor([])
        temp_val_data_x = torch.Tensor([])
        temp_val_data_y = torch.Tensor([])
        print("creating the task",task)
        for i,class_ in enumerate(task):
            # print(class_,type(class_))
            temp_train_data_x = torch.cat([temp_train_data_x, numpyarray_to_tensorarray(train_set_x[class_])])
            temp_train_data_y = torch.cat([temp_train_data_y, (numpyarray_to_tensorarray(train_set_y[class_]))])
            temp_test_data_x = torch.cat([temp_test_data_x, numpyarray_to_tensorarray(test_set_x[class_])])
            temp_test_data_y = torch.cat([temp_test_data_y, (numpyarray_to_tensorarray(test_set_y[class_]))])
            temp_val_data_x = torch.cat([temp_val_data_x, numpyarray_to_tensorarray(val_set_x[class_])])
            temp_val_data_y = torch.cat([temp_val_data_y, (numpyarray_to_tensorarray(val_set_y[class_]))])
            assert not np.isnan(train_set_x[class_]).any(), "NaN detected in train data!"
            assert not np.isnan(test_set_x[class_]).any(), "NaN detected in test data!"
            assert not np.isnan(val_set_x[class_]).any(), "NaN detected in val data!"
            print("validation set has:",np.unique(val_set_y[class_]))
        if shuffle:
            shuffler = torch.randperm(temp_train_data_x.size()[0])
            temp_train_data_x = temp_train_data_x[shuffler]
            temp_train_data_y = temp_train_data_y[shuffler]


        
        # temp_test_data_x = torch.nan_to_num(temp_test_data_x, nan=0.0)
        # temp_test_data_y = torch.nan_to_num(temp_test_data_y, nan=0.0)
        # temp_train_data_x = torch.nan_to_num(temp_train_data_x, nan=0.0)
        # temp_train_data_y = torch.nan_to_num(temp_train_data_y, nan=0.0)
        
        # temp_test_data_y = torch.squeeze(temp_test_data_y)
        # temp_train_data_y = torch.squeeze(temp_train_data_y)     
        # print(temp_test_data_y.shape,temp_train_data_y.shape)
        # print(((temp_train_data_y.float()).view(torch.numel(temp_train_data_y))))
            
        # temp_test_data_y = F.one_hot(temp_test_data_y.to(dtype=torch.long), 2).float().squeeze()
        # temp_train_data_y = F.one_hot(temp_train_data_y.to(dtype=torch.long), 2).float().squeeze()
        assert not torch.isnan(temp_train_data_x).any(), "NaN detected in train data!"
        assert not torch.isnan(temp_test_data_x).any(), "NaN detected in test data!"
        assert not torch.isnan(temp_val_data_x).any(), "NaN detected in val data!"

        temp_train_data_y =temp_train_data_y.long().squeeze()
        temp_test_data_y =temp_test_data_y.long().squeeze()
        temp_val_data_y = temp_val_data_y.long().squeeze()
        exp_train_first_structure.append([temp_train_data_x, temp_train_data_y.long()])
        exp_val_first_structure.append([temp_val_data_x, temp_val_data_y.long()])
        # exp_test_first_structure.append([temp_test_data_x,temp_test_data_y.float()])
    #     final_train_data_x.append(temp_train_data_x)
    #     final_train_data_y.append(temp_train_data_y)
        # print(np.unique(temp_val_data_y) )
        final_test_data_x.append(temp_test_data_x)
        final_test_data_y.append(temp_test_data_y)
    #     s1+=size(temp_train_data_x.storage())
    #     s2+=size(temp_train_data_y.storage())
    #     s3+=size(temp_test_data_x.storage())
    #     s4+=size(temp_test_data_y.storage())
    # exp_test_first_structure.append(exp_train_first_structure[0])#passing empty test_tensors to tensors_benchmark may create test set with whole train_tensors, so to avoid we are passing last task test tensors
    # F.one_hot(y_stream_train.to(dtype=torch.long), 2).float()
    
    final_train_data_y = [x.float() for x in final_train_data_y]
    final_test_data_y = [x.float() for x in final_test_data_y] 

    # final_train_data_y = [F.one_hot(x.to(dtype=torch.long), 2).float() for x in final_train_data_y]
    # final_test_data_y = [x.float() for x in final_test_data_y] 
    # print(s1/(1024**3))
    # print(s2/(1024**3))
    # print(s3/(1024**3))
    # print(s4/(1024**3))       

    # return final_train_data_x, final_train_data_y, final_test_data_x, final_test_data_y
    return exp_train_first_structure,exp_val_first_structure,final_test_data_x, final_test_data_y


def numpyarray_to_tensorarray(array):    
    # tensor_array = array
    transform = ToTensor()
    # print(array.shape)
    # for idx in range(0,array.shape[0]):
    #     tensor_array[idx] = transform(array[idx,:])
    return transform(array) 


def list_to_tensor(p_list):
    transform = ToTensor()
    for idx in range(len(p_list)):
        temp = transform(p_list[idx])
        p_list[idx] = temp
            
    return p_list

def avalanche_tensor_to_tensor(test_data_x,test_data_y):
    for tst_set_idx in range(len(test_data_x)):
        tst_X_temp = list_to_tensor(test_data_x[tst_set_idx])
        tst_y_temp = list_to_tensor(test_data_y[tst_set_idx])
        if tst_set_idx == 0:
            tst_X = np.array(tst_X_temp)
            tst_y = np.array(tst_y_temp)
        else:
            tst_X = np.append(tst_X, np.array(tst_X_temp), axis=0)
            tst_y = np.append(tst_y, np.array(tst_y_temp), axis=0)  
    return tst_X,tst_y          



class Tempdataset(Dataset):
    def __init__(self, train_x,train_y, transform=None,target_transform=None):
        self.train_x = train_x
        self.train_y = train_y
        self.transform = ToTensor()
        


    def __getitem__(self, index):
        x = self.train_x[index]
        y = self.train_y[index]
        
        if self.transform is None:
            x = self.transform(x)

        # if self.target_transform is not None:
        #     y = self.target_transform(y)

        return x,y

    def __len__(self):
        return len(self.train_x)  

    def size(self):
        return self.train_x.shape[0]      


   


