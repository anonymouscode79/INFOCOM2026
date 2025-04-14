import numpy as np
import random
from collections import Counter
import math
from functorch import make_functional_with_buffers, vmap, grad
import torch
from torch import linalg as LA

from math import floor,ceil
import itertools



def retrieve_replaysamples(memory_X, memory_y ,memory_y_name,global_priority_list,local_count,replay_size,input_shape,minority_allocation,memory_size,local_store=False,ecbrs_taskaware = False,pred_diff=None):
    numbers = None
    
    if local_store:
        temp_replay_size = replay_size
        indicies = []
        tmp_local_store = {}
        for idx,class_ in enumerate(memory_y_name):
            if class_ in tmp_local_store :
                tmp_local_store[class_].append(idx)
            else:
                tmp_local_store[class_] = [idx]
        
        weight_dict = {}
        unique_classes = np.unique(memory_y_name)
        # print(unique_classes)
        weight_dict = {}
        # print(global_priority_list)
        for class_idx,_ in enumerate(unique_classes):
            _class = unique_classes[class_idx]
            if _class in list(global_priority_list.keys()) and global_priority_list[_class] == max(global_priority_list.values()) and local_count[_class] == (minority_allocation*memory_size):
                weight_dict[_class] = 1
            else:
                weight_dict[_class] = 1/local_count[_class]
        # print(weight_dict)
        # exit() 
              
        

        total = sum(weight_dict.values())
        weight_dict = {key: value / total for key, value in weight_dict.items()}     
        # print(weight_dict) 

        while temp_replay_size >0 and weight_dict:
            itemMaxValue=max(weight_dict, key=weight_dict.get)
            # print("max values is:",itemMaxValue)
            listOfKeys = list()
            # Iterate over all the items in dictionary to find keys with max value
            for key, value in weight_dict.items():
                if value == weight_dict[itemMaxValue]:
                    listOfKeys.append(key)
                    # print("key:",key,len(tmp_local_store[key]))
            # print(listOfKeys)    
            # print(temp_replay_size)    
            if len(tmp_local_store[listOfKeys[0]])>0:
                for key in listOfKeys:
                    
                    if temp_replay_size > 0 and len(tmp_local_store[key])>0:
                        indicies.extend([tmp_local_store[key].pop(0)])
                        temp_replay_size = temp_replay_size-1
            else:
                for key in listOfKeys:
                    del weight_dict[key]
                total = sum(weight_dict.values())
                weight_dict = {key: value / total for key, value in weight_dict.items()} 
        #     print(weight_dict)  
        #     print("temp replay size:",temp_replay_size)    

        # print("temp replay size:",temp_replay_size)
        numbers = indicies  
        # 
        # print("replaysamples counter:",Counter(numbers))       
    else:
        weights,probas = [],[]
        for x in range(memory_y_name.shape[0]):
            # weights.append(1/local_count[memory_y_name[x]])
            if memory_y_name[x] in list(global_priority_list.keys()) and global_priority_list[memory_y_name[x]] == max(global_priority_list.values()) and local_count[memory_y_name[x]] == (minority_allocation*memory_size):
                weights.append(1)
            else:
                weights.append(1/local_count[memory_y_name[x]])

        #appending predict differences to weight values
        
        if pred_diff is not None:
            weights = [x+y for x,y in zip(weights, pred_diff)]       

        total_weight = sum(weights)
        probas = [weight/total_weight for weight in weights] #translates class_weigths to their probabilities
        numbers = np.random.choice(range(0,memory_y_name.shape[0]), min(replay_size,memory_y_name.shape[0]), replace=False, p=probas)
    
    
    if ecbrs_taskaware:
        replay_Xt, replay_yt, replay_yname = np.zeros((min(replay_size,memory_size),input_shape)), np.zeros(min(replay_size,memory_size)), []
    else:
        # numbers = np.random.choice(range(0,memory_y_name.shape[0]), min(replay_size,memory_y_name.shape[0]), replace=False, p=probas) 
        replay_Xt, replay_yt, replay_yname = np.zeros((replay_size,input_shape)), np.zeros(replay_size), [] # Xt.shape[1] = 7, Xt.shape[2] = 10, Xt.shape[3] = 1

    for ind,rand in enumerate(numbers): #loading the samples from CBRS Memory which will replay later
      replay_Xt[ind], replay_yt[ind] = memory_X[rand], memory_y[rand] 
      replay_yname.append(memory_y_name[rand])  

    
    
    return replay_Xt,replay_yt,replay_yname  


def retrieve_MIR_replaysamples(memory_X, memory_y ,memory_y_name,local_count,replay_size,input_shape):
    weights,probas = [],[]

    for x in range(memory_y_name.shape[0]):
      weights.append(1/local_count[memory_y_name[x]])

    total_weight = sum(weights)
    probas = [weight/total_weight for weight in weights] #translates class_weigths to their probabilities

    numbers = np.random.choice(range(0,memory_y_name.shape[0]), min(replay_size,memory_y_name.shape[0]), replace=False, p=probas) 

    replay_Xt, replay_yt, replay_yname = np.zeros((replay_size,input_shape)), np.zeros(replay_size), [] # Xt.shape[1] = 7, Xt.shape[2] = 10, Xt.shape[3] = 1

    for ind,rand in enumerate(numbers): #loading the samples from CBRS Memory which will replay later
      replay_Xt[ind], replay_yt[ind] = memory_X[rand], memory_y[rand] 
      replay_yname.append(memory_y_name[rand])  

    return replay_Xt,replay_yt,replay_yname  



def cbrsmemory_update(Xt,yt,ynamet,task_id,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list):
    if(task_id == 0 and memory_y_name.shape[0]==memory_size and curr_batch+replay_size>memory_y_name.shape[0]) or (task_id != 0):
        for j in range(len(ynamet)):
            #if memory noot filled
            if (memory_y_name.shape[0]<memory_size):
                Xtreshape = Xt[j].reshape(1,input_shape)#.reshape(1,7,10,1)
                ytnew = np.empty(1, dtype=object)
                ytnew[0] = yt[j]
                ynametnew = np.empty(1, dtype=object)
                ynametnew[0] = ynamet[j]
                memory_X = np.concatenate((memory_X,Xtreshape), axis=0)
                memory_y = np.concatenate((memory_y,ytnew), axis=0)
                memory_y_name = np.concatenate((memory_y_name,ynametnew), axis=0)
                local_count[ynamet[j]]+=1

                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(memory_y_name.shape[0])
                else:
                    local_store[ynamet[j]] = [memory_y_name.shape[0]]

            elif ynamet[j] not in full:
                largest = set()     
                for class_ in local_count.keys():
                    if local_count[class_] == max(local_count.values()):          
                        largest.add(class_)
                        full.add(class_)

                largest_members = []

                for k in local_store.keys():
                    if k in largest:
                        largest_members.extend(local_store[k])
              
          # ------------------------------------------------------------------------------------------------# 

        # We randomly pick one of the "largest" class members (which we stored in largest_members)
                rand = random.randint(0,len(largest_members)-1)
                rand_largest = largest_members[rand]

        # Update local_count, which stores the counts of each class present in the memory buffer
                local_count[memory_y_name[rand_largest]]-=1
                local_count[ynamet[j]]+=1

        # Update the local store  ,, at rand_largest position, we are removing  'memory_y_name[rand_largest]' class instance and adding 'ynamet[j]' class instance
                local_store[memory_y_name[rand_largest]].remove(rand_largest)
                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(rand_largest)
                else:
                    local_store[ynamet[j]] = [rand_largest]
          #local_store[ynamet[j]].append(rand_largest)

        # Replace the "largest" class member we picked earlier with the current instance
                memory_X[rand_largest] = Xt[j]
                memory_y[rand_largest] = yt[j]
                memory_y_name[rand_largest] = ynamet[j]


            else:
                threshold = local_count[ynamet[j]] / global_count[ynamet[j]]
                u = random.uniform(0, 1)
          
                if u <= threshold:
                    rand_same_class = local_store[ynamet[j]][0]

          # We randomly pick one of the same class members (which we stored in same_class)
            #rand = random.randint(0,len(same_class)-1)
            #rand_same_class = same_class[rand]

          # NOTE: We don't need to update local_count since we are replacing a member from the same class
          # as the current instance, so overall count for that class stays same

          # Replace the chosen member with current instance
                    memory_X[rand_same_class] = Xt[j]
                    memory_y[rand_same_class] = yt[j]
                    memory_y_name[rand_same_class] = ynamet[j] 

    return memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list                   





def memory_update(Xt,yt,ynamet,task_id,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list):
    if(task_id == 0 and memory_y_name.shape[0]==memory_size and curr_batch+replay_size>memory_y_name.shape[0]) or (task_id != 0):
        for j in range(len(ynamet)):
            #if memory not filled
            if (memory_y_name.shape[0]<memory_size):
                Xtreshape = Xt[j].reshape(1,input_shape)#.reshape(1,7,10,1)
                ytnew = np.empty(1, dtype=object)
                ytnew[0] = yt[j]
                ynametnew = np.empty(1, dtype=object)
                ynametnew[0] = ynamet[j]
                memory_X = np.concatenate((memory_X,Xtreshape), axis=0)
                memory_y = np.concatenate((memory_y,ytnew), axis=0)
                memory_y_name = np.concatenate((memory_y_name,ynametnew), axis=0)
                local_count[ynamet[j]]+=1

                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(memory_y_name.shape[0])
                else:
                    local_store[ynamet[j]] = [memory_y_name.shape[0]]

            elif ynamet[j] not in full:
                # print("ynametj",ynamet[j])
                for class_ in classes_so_far:
                    if class_ in local_count.keys():
                        #print(class_)
                        global_priority_list[class_]=global_count[class_]/sum(global_count.values())
                        local_priority_list[class_]=local_count[class_]/sum(local_count.values())

                largest = set()
                class_to_add=ynamet[j]
                list_keys = list(local_count.keys())
                list_keys = [int(x) for x in list_keys]
                global_max_class = [max(global_priority_list, key=global_priority_list.get)]
                global_max_class = [int(x) for x in global_max_class]
                list_keys =  [x for x in list_keys if x not in  global_max_class] 
                class_to_add =  list_keys[0] if len(list_keys)>0 else ynamet[j] # fix when no data shuffling and loading initial samples for custom LSTM train
                for class_ in list_keys:
                    if local_count[class_] >= local_count[class_to_add]:# and global_priority_list[class_] != max(global_priority_list.values()):
                        class_to_add = class_        
                
          
         
                temp = min(global_priority_list.values())
                for idx,value in local_count.items():
                    if (global_priority_list[idx] >= temp and global_priority_list[idx] == max(global_priority_list.values()) and local_count[idx] > int(minority_allocation*memory_size)):# or round(global_priority_list[idx],offset) >= round(temp,offset):
                        class_to_add = idx
                        temp =global_priority_list[idx]  

                if global_priority_list[class_to_add] != max(global_priority_list.values()):
                    keys_list = list(local_count.keys())
                    keys_list = [int(x) for x in keys_list]
                    global_max_class = [max(global_priority_list, key=global_priority_list.get)]
                    #print("global max class is",global_max_class)
                    global_max_class = [int(x) for x in global_max_class]
                    keys_list = [x for x in keys_list if x not in  global_max_class]
                    curr_max = local_count[keys_list[0]]
                    for class_ in keys_list:
                        if local_count[class_] >curr_max:# max(local_count.values()) and global_priority_list[class_] != max(global_priority_list.values()):
                            class_to_add = class_
                            curr_max = local_count[class_]              

                largest.add(class_to_add)
                full.add(class_to_add)

                largest_members = []

                for k in local_store.keys():
                    if k in largest:
                        largest_members.extend(local_store[k])
              
          # ------------------------------------------------------------------------------------------------# 

        # We randomly pick one of the "largest" class members (which we stored in largest_members)
                rand = random.randint(0,len(largest_members)-1)
                rand_largest = largest_members[rand]

        # Update local_count, which stores the counts of each class present in the memory buffer
                local_count[memory_y_name[rand_largest]]-=1
                local_count[ynamet[j]]+=1

        # Update the local store  ,, at rand_largest position, we are removing  'memory_y_name[rand_largest]' class instance and adding 'ynamet[j]' class instance
                local_store[memory_y_name[rand_largest]].remove(rand_largest)
                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(rand_largest)
                else:
                    local_store[ynamet[j]] = [rand_largest]
          #local_store[ynamet[j]].append(rand_largest)

        # Replace the "largest" class member we picked earlier with the current instance
                memory_X[rand_largest] = Xt[j]
                memory_y[rand_largest] = yt[j]
                memory_y_name[rand_largest] = ynamet[j]


            else:
                threshold = local_count[ynamet[j]] / global_count[ynamet[j]]
                u = random.uniform(0, 1)
          
                if u <= threshold:
                    rand_same_class = local_store[ynamet[j]][0]

          # We randomly pick one of the same class members (which we stored in same_class)
                    # rand_same_class = random.randint(0,len(local_store[ynamet[j]])-1)
                    # rand_same_class = same_class[rand]

          # NOTE: We don't need to update local_count since we are replacing a member from the same class
          # as the current instance, so overall count for that class stays same

          # Replace the chosen member with current instance
                    memory_X[rand_same_class] = Xt[j]
                    memory_y[rand_same_class] = yt[j]
                    memory_y_name[rand_same_class] = ynamet[j] 

    return memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list                   






def memory_update2(Xt,yt,ynamet,task_id,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list,model,loss_fn,device,data_reshape=None):
    model.eval()
    if(task_id == 0 and memory_y_name.shape[0]==memory_size and curr_batch+replay_size>memory_y_name.shape[0]) or (task_id != 0):
        for j in range(len(ynamet)):
            #if memory not filled
            if (memory_y_name.shape[0]<memory_size):
                Xtreshape = Xt[j].reshape(1,input_shape)#.reshape(1,7,10,1)
                ytnew = np.empty(1, dtype=object)
                ytnew[0] = yt[j]
                ynametnew = np.empty(1, dtype=object)
                ynametnew[0] = ynamet[j]
                memory_X = np.concatenate((memory_X,Xtreshape), axis=0)
                memory_y = np.concatenate((memory_y,ytnew), axis=0)
                memory_y_name = np.concatenate((memory_y_name,ynametnew), axis=0)
                local_count[ynamet[j]]+=1

                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(memory_y_name.shape[0])
                else:
                    local_store[ynamet[j]] = [memory_y_name.shape[0]]

            elif ynamet[j] not in full:
                # print("ynametj",ynamet[j])
                for class_ in classes_so_far:
                    if class_ in local_count.keys():
                        #print(class_)
                        global_priority_list[class_]=global_count[class_]/sum(global_count.values())
                        local_priority_list[class_]=local_count[class_]/sum(local_count.values())

                largest = set()
                class_to_add=ynamet[j]
                list_keys = list(local_count.keys())
                list_keys = [int(x) for x in list_keys]
                global_max_class = [max(global_priority_list, key=global_priority_list.get)]
                global_max_class = [int(x) for x in global_max_class]
                list_keys =  [x for x in list_keys if x not in  global_max_class] 
                class_to_add =  list_keys[0] if len(list_keys)>0 else ynamet[j] # fix when no data shuffling and loading initial samples for custom LSTM train
                for class_ in list_keys:
                    if local_count[class_] >= local_count[class_to_add]:# and global_priority_list[class_] != max(global_priority_list.values()):
                        class_to_add = class_        
                
          
         
                temp = min(global_priority_list.values())
                for idx,value in local_count.items():
                    if (global_priority_list[idx] >= temp and global_priority_list[idx] == max(global_priority_list.values()) and local_count[idx] > int(minority_allocation*memory_size)):# or round(global_priority_list[idx],offset) >= round(temp,offset):
                        class_to_add = idx
                        temp =global_priority_list[idx]  

                if global_priority_list[class_to_add] != max(global_priority_list.values()):
                    keys_list = list(local_count.keys())
                    keys_list = [int(x) for x in keys_list]
                    global_max_class = [max(global_priority_list, key=global_priority_list.get)]
                    #print("global max class is",global_max_class)
                    global_max_class = [int(x) for x in global_max_class]
                    keys_list = [x for x in keys_list if x not in  global_max_class]
                    curr_max = local_count[keys_list[0]]
                    for class_ in keys_list:
                        if local_count[class_] >curr_max:# max(local_count.values()) and global_priority_list[class_] != max(global_priority_list.values()):
                            class_to_add = class_
                            curr_max = local_count[class_]              

                largest.add(class_to_add)
                full.add(class_to_add)

                largest_members = []

                for k in local_store.keys():
                    if k in largest:
                        largest_members.extend(local_store[k])
              
          # ------------------------------------------------------------------------------------------------# 

        # We randomly pick one of the "largest" class members (which we stored in largest_members)
                
                def compute_loss_stateless_model (params, buffers, sample, target):
                    # if data_reshape is not None:
                    batch = sample.unsqueeze(0)
                    targets = target.unsqueeze(0)

                    predictions = fmodel(params, buffers, batch) 
                    loss = loss_fn(predictions, targets)
                    return loss
                    
                fmodel, params, buffers = make_functional_with_buffers(model)
                ft_compute_grad = grad(compute_loss_stateless_model)
                ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
                grad_list = []
                
                for lm_idx in largest_members:
                    # if data_reshape is not None:
                    data = torch.from_numpy(memory_X[lm_idx]).reshape(data_reshape).to(device)
                    # else:
                    #     data = torch.from_numpy(memory_X[lm_idx]).to(device)

                    target = torch.as_tensor(memory_y[lm_idx]).reshape(1,1).float().to(device)
                    ft_per_sample_grad = ft_compute_sample_grad(params, buffers, data, target)
                    grad2_list =[]
                    sum_grad=0
                    for tch_len in range(len(ft_per_sample_grad)):
                        temp_grad = torch.flatten(ft_per_sample_grad[tch_len])                        
                        sum_grad = sum_grad+(LA.norm(temp_grad).item())                       

                       
                    grad_list.extend([sum_grad])

                # print(len(grad_list))
                # print(len(largest_members))
                # print(grad_list.index(min(grad_list)))
                # print(np.amin(grad_list))
                # rand = random.randint(0,len(largest_members)-1)
                rand = grad_list.index(min(grad_list))
                # rand = random.randint(0,len(largest_members)-1)
                rand_largest = largest_members[rand]

        # Update local_count, which stores the counts of each class present in the memory buffer
                local_count[memory_y_name[rand_largest]]-=1
                local_count[ynamet[j]]+=1

        # Update the local store  ,, at rand_largest position, we are removing  'memory_y_name[rand_largest]' class instance and adding 'ynamet[j]' class instance
                local_store[memory_y_name[rand_largest]].remove(rand_largest)
                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(rand_largest)
                else:
                    local_store[ynamet[j]] = [rand_largest]
          #local_store[ynamet[j]].append(rand_largest)

        # Replace the "largest" class member we picked earlier with the current instance
                memory_X[rand_largest] = Xt[j]
                memory_y[rand_largest] = yt[j]
                memory_y_name[rand_largest] = ynamet[j]


            else:
                threshold = local_count[ynamet[j]] / global_count[ynamet[j]]
                u = random.uniform(0, 1)
          
                if u <= threshold:
                    rand_same_class = local_store[ynamet[j]][0]

          # We randomly pick one of the same class members (which we stored in same_class)
            #rand = random.randint(0,len(same_class)-1)
            #rand_same_class = same_class[rand]

          # NOTE: We don't need to update local_count since we are replacing a member from the same class
          # as the current instance, so overall count for that class stays same

          # Replace the chosen member with current instance
                    memory_X[rand_same_class] = Xt[j]
                    memory_y[rand_same_class] = yt[j]
                    memory_y_name[rand_same_class] = ynamet[j] 

    model.train()
    return memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list                   

def memory_update3(Xt,yt,ynamet,task_id,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list,model,loss_fn,device):
    model.eval()
    if(task_id == 0 and memory_y_name.shape[0]==memory_size and curr_batch+replay_size>memory_y_name.shape[0]) or (task_id != 0):
        for j in range(len(ynamet)):
            #if memory not filled
            if (memory_y_name.shape[0]<memory_size):
                Xtreshape = Xt[j].reshape(1,input_shape)#.reshape(1,7,10,1)
                ytnew = np.empty(1, dtype=object)
                ytnew[0] = yt[j]
                ynametnew = np.empty(1, dtype=object)
                ynametnew[0] = ynamet[j]
                memory_X = np.concatenate((memory_X,Xtreshape), axis=0)
                memory_y = np.concatenate((memory_y,ytnew), axis=0)
                memory_y_name = np.concatenate((memory_y_name,ynametnew), axis=0)
                local_count[ynamet[j]]+=1

                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(memory_y_name.shape[0])
                else:
                    local_store[ynamet[j]] = [memory_y_name.shape[0]]

            elif ynamet[j] not in full:
                # print("ynametj",ynamet[j])
                for class_ in classes_so_far:
                    if class_ in local_count.keys():
                        #print(class_)
                        global_priority_list[class_]=global_count[class_]/sum(global_count.values())
                        local_priority_list[class_]=local_count[class_]/sum(local_count.values())

                largest = set()
                class_to_add=ynamet[j]
                list_keys = list(local_count.keys())
                list_keys = [int(x) for x in list_keys]
                global_max_class = [min(global_priority_list, key=global_priority_list.get)]
                global_max_class = [int(x) for x in global_max_class]
                list_keys =  [x for x in list_keys if x not in  global_max_class] 
                class_to_add =  list_keys[0] if len(list_keys)>0 else ynamet[j] # fix when no data shuffling and loading initial samples for custom LSTM train
                for class_ in list_keys:
                    if local_count[class_] <= local_count[class_to_add]:# and global_priority_list[class_] != max(global_priority_list.values()):
                        class_to_add = class_        
                
          
         
                temp = max(global_priority_list.values())
                for idx,value in local_count.items():
                    if (global_priority_list[idx] <= temp and global_priority_list[idx] == min(global_priority_list.values()) and local_count[idx] > int(minority_allocation*memory_size)):# or round(global_priority_list[idx],offset) >= round(temp,offset):
                        class_to_add = idx
                        temp =global_priority_list[idx]  

                if global_priority_list[class_to_add] != min(global_priority_list.values()):
                    keys_list = list(local_count.keys())
                    keys_list = [int(x) for x in keys_list]
                    global_max_class = [min(global_priority_list, key=global_priority_list.get)]
                    #print("global max class is",global_max_class)
                    global_max_class = [int(x) for x in global_max_class]
                    keys_list = [x for x in keys_list if x not in  global_max_class]
                    curr_max = local_count[keys_list[0]]
                    for class_ in keys_list:
                        if local_count[class_] <curr_max:# max(local_count.values()) and global_priority_list[class_] != max(global_priority_list.values()):
                            class_to_add = class_
                            curr_max = local_count[class_]              

                largest.add(class_to_add)
                full.add(class_to_add)

                largest_members = []

                for k in local_store.keys():
                    if k in largest:
                        largest_members.extend(local_store[k])
              
          # ------------------------------------------------------------------------------------------------# 

        # We randomly pick one of the "largest" class members (which we stored in largest_members)
                def compute_loss_stateless_model (params, buffers, sample, target):
                    batch = sample.unsqueeze(0)
                    targets = target.unsqueeze(0)

                    predictions = fmodel(params, buffers, batch) 
                    loss = loss_fn(predictions, targets)
                    return loss
                    
                fmodel, params, buffers = make_functional_with_buffers(model)
                ft_compute_grad = grad(compute_loss_stateless_model)
                ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
                grad_list = []
                
                for lm_idx in largest_members:
                    data = torch.from_numpy(memory_X[lm_idx]).reshape(1,3,32,32).to(device)
                    target = torch.as_tensor(memory_y[lm_idx]).reshape(1,1).float().to(device)
                    ft_per_sample_grad = ft_compute_sample_grad(params, buffers, data, target)
                    grad2_list =[]
                    sum_grad=0
                    for tch_len in range(len(ft_per_sample_grad)):
                        temp_grad = torch.flatten(ft_per_sample_grad[tch_len])                        
                        sum_grad = sum_grad+(LA.norm(temp_grad).item())                       

                       
                    grad_list.extend([sum_grad])

                # print(len(grad_list))
                # print(len(largest_members))
                # print(grad_list.index(min(grad_list)))
                # print(np.amin(grad_list))
                # rand = random.randint(0,len(largest_members)-1)
                rand = grad_list.index(min(grad_list))
                rand_largest = largest_members[rand]

        # Update local_count, which stores the counts of each class present in the memory buffer
                local_count[memory_y_name[rand_largest]]-=1
                local_count[ynamet[j]]+=1

        # Update the local store  ,, at rand_largest position, we are removing  'memory_y_name[rand_largest]' class instance and adding 'ynamet[j]' class instance
                local_store[memory_y_name[rand_largest]].remove(rand_largest)
                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(rand_largest)
                else:
                    local_store[ynamet[j]] = [rand_largest]
          #local_store[ynamet[j]].append(rand_largest)

        # Replace the "largest" class member we picked earlier with the current instance
                memory_X[rand_largest] = Xt[j]
                memory_y[rand_largest] = yt[j]
                memory_y_name[rand_largest] = ynamet[j]


            else:
                threshold = local_count[ynamet[j]] / global_count[ynamet[j]]
                u = random.uniform(0, 1)
          
                if u <= threshold:
                    rand_same_class = local_store[ynamet[j]][0]

          # We randomly pick one of the same class members (which we stored in same_class)
            #rand = random.randint(0,len(same_class)-1)
            #rand_same_class = same_class[rand]

          # NOTE: We don't need to update local_count since we are replacing a member from the same class
          # as the current instance, so overall count for that class stays same

          # Replace the chosen member with current instance
                    memory_X[rand_same_class] = Xt[j]
                    memory_y[rand_same_class] = yt[j]
                    memory_y_name[rand_same_class] = ynamet[j] 
    model.train()                

    return memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list                   



def random_memory_update(Xt,yt,ynamet,task_id,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list):

    indices_len = math.ceil(0.30*memory_X.shape[0]) if Xt.shape[0] > math.ceil(0.30*memory_X.shape[0]) else math.ceil(0.30*Xt.shape[0])
    memory_indices = np.random.choice(range(0,memory_X.shape[0]),size=indices_len,replace=False).tolist()
    xt_indices = np.random.choice(range(0,(Xt.shape[0])),size=indices_len,replace=False).tolist()
    for i,j in zip(memory_indices,xt_indices):
        memory_X[i] = Xt[j]
        memory_y[i] = yt[j]
        memory_y_name[i] = ynamet[j]


                

    return memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list                  

def memory_update_equal_allocation(Xt,yt,ynamet,memory_size,memory_X,memory_y, memory_y_name,minority_class_ids,majority_class_memory_share=0.15,random_sample_selection=False,temp_model=None,image_resolution=None,device=None):
    temp_memory_X, temp_memory_y, temp_memory_y_name = None,None,None
    if memory_X is None:
        class_in_memory = np.unique(ynamet)
        minority_class = [int(loop_var) for loop_var in minority_class_ids]    
        minorityclass_in_memory = [loop_var for loop_var in minority_class if loop_var in class_in_memory]
        majorityclass_in_memory = [loop_var for loop_var in class_in_memory if loop_var not in minority_class]
        alloc_per_majority_class = floor(memory_size*majority_class_memory_share)
        alloc_per_majority_class = floor(alloc_per_majority_class/len(majorityclass_in_memory))
        for class_idx in majorityclass_in_memory:
            indices = (np.where(ynamet == int(class_idx))[0]).tolist()
            alloc_count = alloc_per_majority_class
            if len(indices) < alloc_count:
                alloc_count = len(indices)
            indices = indices[0:alloc_count]
            temp_memory_X = Xt[indices,:]
            temp_memory_y = yt[indices]
            temp_memory_y_name = ynamet[indices]

        residual_memory_size = memory_size-alloc_per_majority_class
        alloc_per_class = floor(residual_memory_size/len(minorityclass_in_memory))   
        for class_idx in minorityclass_in_memory:
            indices = np.where(ynamet == int(class_idx))[0]
            indices = indices[0:alloc_per_class]
            if temp_memory_X is None:
                temp_memory_X = Xt[indices,:]
                temp_memory_y = yt[indices]
                temp_memory_y_name = ynamet[indices]
            else:
                temp_memory_X = np.concatenate((temp_memory_X,Xt[indices,:]), axis=0)   
                temp_memory_y = np.concatenate((temp_memory_y,yt[indices]), axis=0)  
                temp_memory_y_name = np.concatenate((temp_memory_y_name,ynamet[indices]), axis=0)  
        return temp_memory_X,temp_memory_y,temp_memory_y_name
    else:

        return memory_update_equal_allocation3(Xt,yt,ynamet,memory_size,memory_X, memory_y, memory_y_name,minority_class_ids,majority_class_memory_share=majority_class_memory_share,random_sample_selection=random_sample_selection,temp_model=temp_model,image_resolution=image_resolution,device=device)      



def memory_update_equal_allocation3(Xt,yt,ynamet,memory_size,memory_X, memory_y, memory_y_name,minority_class_ids,majority_class_memory_share=0.15,random_sample_selection=False,temp_model=None,image_resolution=None,device=None):
    temp_memory_X, temp_memory_y, temp_memory_y_name = None,None,None
    indices_dict_mem, indices_dict_task = {},{}
    class_in_memory = np.unique(memory_y_name)
    class_in_task = np.unique(ynamet)
       
    minority_class = [int(loop_var) for loop_var in minority_class_ids]
    
    minorityclass_in_memory = [loop_var for loop_var in minority_class if loop_var in class_in_memory]
    minorityclass_in_task = [loop_var for loop_var in minority_class if loop_var in class_in_task]
    temp_total_minority_classess = minorityclass_in_memory + minorityclass_in_task
    majorityclass_in_memory = [loop_var for loop_var in class_in_memory if loop_var not in temp_total_minority_classess]
    # majorityclass_in_memory = [loop_var for loop_var in class_in_memory if loop_var not in minority_class]
    # print("majority class in memory",majorityclass_in_memory)
    
    
    # print("minority class in task",minorityclass_in_task)
    majorityclass_in_task = [loop_var for loop_var in class_in_task if loop_var not in temp_total_minority_classess]
    # majorityclass_in_task = [loop_var for loop_var in class_in_task if loop_var not in minority_class]
    # print("majority class in task",majorityclass_in_task)
    total_minority_classes = len(minorityclass_in_memory) + len(minorityclass_in_task)
    alloc_per_majority_class = floor(memory_size*majority_class_memory_share)
    # print("majority class memory",alloc_per_majority_class)
    whole_majority_classes = majorityclass_in_memory + majorityclass_in_task
    alloc_per_majority_class = floor(alloc_per_majority_class/whole_majority_classes)
    for class_idx in majorityclass_in_memory:
        indices = (np.where(memory_y_name == int(class_idx))[0]).tolist()
        # print("tem_x shape:0",len(indices))
        # print("indices in memory",indices)
        alloc_count = alloc_per_majority_class
        if len(indices) < alloc_count:
            alloc_count = len(indices)
        indices = indices[0:alloc_count]
        if temp_memory_X is None:
            temp_memory_X = memory_X[indices,:]
            temp_memory_y = memory_y[indices]
            temp_memory_y_name = memory_y_name[indices]
            if class_idx not in indices_dict_mem.keys():
                indices_dict_mem[class_idx] = indices
            else:
                indices_dict_mem[class_idx].extend(indices)
            # print("tem_x shape:1",temp_memory_X.shape[0])        

    

    for class_idx in majorityclass_in_task:
        indices = (np.where(ynamet == int(class_idx))[0]).tolist()
        # print("indices in task",indices)  
        alloc_count = alloc_per_majority_class
        if temp_memory_X is None:
            if len(indices) < alloc_count:
                alloc_count = len(indices)
            indices = indices[0:alloc_count]
            temp_memory_X = Xt[indices,:]
            temp_memory_y = yt[indices]
            temp_memory_y_name = ynamet[indices]  
            if class_idx not in indices_dict_task.keys():
                indices_dict_task[class_idx] = indices
            else:
                indices_dict_task[class_idx].extend(indices)  
            # print("tem_x shape:2",temp_memory_X.shape[0])    

        else:
            mem_maj_class_indices = list(range(temp_memory_X.shape[0]))
            # for maj_class in (whole_majority_classes):
            #     mem_maj_class_indices.extend(list(np.where(temp_memory_y_name == int(maj_class))[0]))
                
               
            
            if  len(mem_maj_class_indices)< alloc_per_majority_class:
                residual_allocation =  alloc_per_majority_class - len(mem_maj_class_indices)
                if len(indices) < residual_allocation:
                    residual_allocation = len(indices)
                indices = indices[0:residual_allocation]    
                temp_memory_X = np.concatenate((temp_memory_X,Xt[indices,:]), axis=0)   
                temp_memory_y = np.concatenate((temp_memory_y,yt[indices]), axis=0)  
                temp_memory_y_name = np.concatenate((temp_memory_y_name,ynamet[indices]), axis=0)  
                if class_idx not in indices_dict_task.keys():
                    indices_dict_task[class_idx] = indices
                else:
                    indices_dict_task[class_idx].extend(indices)  
                # print("tem_x shape:3",temp_memory_X.shape[0])    
            
            else:

                initial_selection = floor(0.1*len(mem_maj_class_indices))
                indices = random.sample(indices,min(initial_selection,len(indices)))
                mem_indices = random.sample(mem_maj_class_indices,min(initial_selection,len(indices)))
                temp_memory_X[mem_indices,:] = Xt[indices,:]
                temp_memory_y[mem_indices] = yt[indices]
                temp_memory_y_name[mem_indices] = ynamet[indices]
                # print("tem_x shape:4",temp_memory_X.shape[0])
                
                # print(len(mem_indices))
                # new_list = [xyz for xyz in mem_indices if xyz in mem_maj_class_indices]
                # print(len(new_list))
                # # exit()
                # for mem_index in mem_indices:
                #     indices_dict_mem[temp_memory_y_name[mem_index]].remove(mem_index)

                if class_idx not in indices_dict_task.keys():
                            indices_dict_task[class_idx] = indices
                else:
                            indices_dict_task[class_idx].extend(indices)
                # indices = indices[0:initial_selection]
                # mem_indices
                # for sample_idx in indices:
                #     u = random.uniform(0, 1)
                #     if u >=0.5:
                #         # exit()
                #         if random_sample_selection:
                #             rand = random.randint(0,temp_memory_X.shape[0]-1)
                #             temp_memory_X[rand] = Xt[sample_idx]
                #             temp_memory_y[rand] = yt[sample_idx]
                #             temp_memory_y_name[rand] = ynamet[sample_idx]
                #             # if class_idx not in indices_dict_task.keys():
                #             #     indices_dict_task[class_idx] = [sample_idx]
                #             # else:
                #             #     indices_dict_task[class_idx].extend([sample_idx])

                #         else:
                #             # exit()
                #             temp_model.eval()
                #             if image_resolution is not None:
                #                 X_temp = torch.from_numpy(temp_memory_X).reshape(image_resolution).to(device)
                #             else:
                #                 X_temp = torch.from_numpy(temp_memory_X).to(device)    
                #             yhat = temp_model(X_temp).detach().cpu().numpy()     
                #             samples_loss = abs(yhat-temp_memory_y.reshape(yhat.shape)).tolist()
                #             minpos = samples_loss.index(min(samples_loss))
                #             temp_memory_X[minpos] = Xt[sample_idx]
                #             temp_memory_y[minpos] = yt[sample_idx]
                #             temp_memory_y_name[minpos] = ynamet[sample_idx]
                #         if class_idx not in indices_dict_task.keys():
                #             indices_dict_task[class_idx] = [sample_idx]
                #         else:
                #             indices_dict_task[class_idx].extend([sample_idx])

                #         temp_model.train()    





   

    residual_memory_size = memory_size-alloc_per_majority_class
    alloc_per_class = floor(residual_memory_size/total_minority_classes)
    # print("minory class allocation",alloc_per_class)
    # temp_memory_X,temp_memory_y,temp_memory_y_name = None,None,None
    for class_idx in minorityclass_in_memory:
        indices = np.where(memory_y_name == int(class_idx))[0]
        indices = indices[0:alloc_per_class]
        if temp_memory_X is None:
            temp_memory_X = memory_X[indices,:]
            temp_memory_y = memory_y[indices]
            temp_memory_y_name = memory_y_name[indices]
        else:
            temp_memory_X = np.concatenate((temp_memory_X,memory_X[indices,:]), axis=0)   
            temp_memory_y = np.concatenate((temp_memory_y,memory_y[indices]), axis=0)  
            temp_memory_y_name = np.concatenate((temp_memory_y_name,memory_y_name[indices]), axis=0) 
        if class_idx not in indices_dict_mem.keys():
            indices_dict_mem[class_idx] = indices
        else:
            indices_dict_mem[class_idx].extend(indices)    

    for class_idx in minorityclass_in_task:
        indices = np.where(ynamet == int(class_idx))[0]
        indices = indices[0:alloc_per_class]
        if temp_memory_X is None:
            temp_memory_X = Xt[indices,:]
            temp_memory_y = yt[indices]
            temp_memory_y_name = ynamet[indices]
        else:
            temp_memory_X = np.concatenate((temp_memory_X,Xt[indices,:]), axis=0)   
            temp_memory_y = np.concatenate((temp_memory_y,yt[indices]), axis=0)  
            temp_memory_y_name = np.concatenate((temp_memory_y_name,ynamet[indices]), axis=0)   

        if class_idx not in indices_dict_task.keys():
            indices_dict_task[class_idx] = indices
        else:
            indices_dict_task[class_idx].extend(indices)

    mem_residual_indices, task_residual_indices = [],[]
    for dict_key in indices_dict_task.keys():
        task_residual_indices.extend(indices_dict_task[dict_key])  
    for dict_key in indices_dict_mem.keys():
        mem_residual_indices.extend(indices_dict_mem[dict_key])    

    task_residual_indices = list(set(range(0,Xt.shape[0]))-set(task_residual_indices))# [idx_num for idx_num in list(range(0,Xt.shape[0])) if idx_num not in task_residual_indices]    
    mem_residual_indices = list(set(range(0,memory_X.shape[0]))-set(mem_residual_indices))#[idx_num for idx_num in list(range(0,memory_X.shape[0])) if idx_num not in mem_residual_indices]

    residual_X,residual_y,residual_yname = memory_X[mem_residual_indices,:],memory_y[mem_residual_indices],memory_y_name[mem_residual_indices]
    residual_X,residual_y,residual_yname = np.concatenate((residual_X,Xt[task_residual_indices]), axis=0),np.concatenate((residual_y,yt[task_residual_indices]), axis=0),np.concatenate((residual_yname,ynamet[task_residual_indices]), axis=0) 
    whole_minority_classes = minorityclass_in_memory + minorityclass_in_task
    residual_class = [int(x) for x in np.unique(residual_yname) if x in whole_minority_classes]
    # residual_class = [int(x) for x in np.unique(residual_yname)]
    # residual_class = [1]
    # residual_class = [int(x) for x in np.unique(residual_y)]
    if len(residual_class)>0 and residual_X.shape[0]>0:
        residual_mem_size = memory_size - temp_memory_X.shape[0] 
        residual_select_samples_per_class = floor(residual_mem_size/len(residual_class))

    
        for idx,class_idx in enumerate(residual_class):
            indices = list(np.where(residual_yname == int(class_idx))[0])
            select_length = min(len(indices),residual_select_samples_per_class)
            indices = indices[0:select_length]
            temp_memory_X = np.concatenate((temp_memory_X,residual_X[indices,:]), axis=0)   
            temp_memory_y = np.concatenate((temp_memory_y,residual_y[indices]), axis=0)  
            temp_memory_y_name = np.concatenate((temp_memory_y_name,residual_yname[indices]), axis=0) 

    #      residual_class_indicies_list.insert(idx,indices)


    # for concat_list in itertools.zip_longest(*residual_class_indicies_list):
    #     residual_class_indicies.extend(list(concat_list))

    # residual_mem_size = memory_size - temp_memory_X.shape[0] 

    # residual_class_indicies = [x_var for x_var in residual_class_indicies if x_var is not None]  
    # if residual_mem_size > 0:
    #     if len(residual_class_indicies) > residual_mem_size:
    #         residual_class_indicies = residual_class_indicies[0:residual_mem_size]

        
    #     temp_memory_X = np.concatenate((temp_memory_X,residual_X[residual_class_indicies,:]), axis=0)   
    #     temp_memory_y = np.concatenate((temp_memory_y,residual_y[residual_class_indicies]), axis=0)  
    #     temp_memory_y_name = np.concatenate((temp_memory_y_name,residual_yname[residual_class_indicies]), axis=0) 
        

  
    return temp_memory_X,temp_memory_y,temp_memory_y_name       




def memory_update_equal_allocation2(Xt,yt,ynamet,memory_size,memory_X, memory_y, memory_y_name,minority_class_ids,majority_class_memory_share=0.15,random_sample_selection=False,temp_model=None,image_resolution=None,device=None):
    temp_memory_X, temp_memory_y, temp_memory_y_name = None,None,None
    
    class_in_memory = np.unique(memory_y_name)
    
    class_in_task = np.unique(ynamet)
    
    total_classes = len(class_in_memory) + len(class_in_task)
    alloc_per_majority_class = floor(memory_size/total_classes)
    # print("majority class memory",alloc_per_majority_class)
    for class_idx in class_in_memory:
        indices = np.where(memory_y_name == int(class_idx))[0]
        # print("indices in memory",indices)
        alloc_count = alloc_per_majority_class
        if len(indices) < alloc_count:
            alloc_count = len(indices)
        indices = indices[0:alloc_count]
        if temp_memory_X is None:
            temp_memory_X = memory_X[indices,:]
            temp_memory_y = memory_y[indices]
            temp_memory_y_name = memory_y_name[indices]
        else:
            temp_memory_X = np.concatenate((temp_memory_X,memory_X[indices,:]), axis=0)   
            temp_memory_y = np.concatenate((temp_memory_y,memory_y[indices]), axis=0)  
            temp_memory_y_name = np.concatenate((temp_memory_y_name,memory_y_name[indices]), axis=0) 

    for class_idx in class_in_task:
        indices = np.where(ynamet == int(class_idx))[0]
        # print("indices in memory",indices)
        alloc_count = alloc_per_majority_class
        if len(indices) < alloc_count:
            alloc_count = len(indices)
        indices = indices[0:alloc_count]
        if temp_memory_X is None:
            temp_memory_X = Xt[indices,:]
            temp_memory_y = yt[indices]
            temp_memory_y_name = ynamet[indices]
        else:
            temp_memory_X = np.concatenate((temp_memory_X,Xt[indices,:]), axis=0)   
            temp_memory_y = np.concatenate((temp_memory_y,yt[indices]), axis=0)  
            temp_memory_y_name = np.concatenate((temp_memory_y_name,ynamet[indices]), axis=0)         


    

    
    return temp_memory_X,temp_memory_y,temp_memory_y_name         

    
    
    