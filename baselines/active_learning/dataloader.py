import numpy as np
def load_tasks(tasks,dataset_name,folder = "data_processed/baseline"):
    X_train,y_train,y_train_bin = [],[],[]
    for i in tasks:
        task_path = f"folder/{dataset_name}/{i}.npz"
        task_data = np.load(task_path)
        X = task_data['X']
        y_binary = task_data['y_binary']
        y = task_data['y']
        X_train.append(X)
        y_train.append(y)
        y_train_bin.append(y_binary)
    return X_train,y_train,y_train_bin
