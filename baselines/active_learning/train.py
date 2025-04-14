import os
from sklearn.metrics import auc, precision_recall_curve
import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from baselines.active_learning.utils.utils import save_model, to_categorical
from baselines.active_learning.losses import HiDistanceXentLoss, TripletMSELoss
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix

from utils.utils import EarlyStopping
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def pseudo_loss( encoder, X_train, y_train, \
                X_test, test_offset, total_epochs,batch_size,sample_reduce = "mean"):
    # contruct the dataset loader
    X_tensor = torch.from_numpy(np.vstack((X_train, X_test))).float()
    # y has a mix of real labels and predicted pseudo labels.
    # y = np.concatenate((y_train, y_test_pred), axis=0)
    # y has binary labels
    # logging.debug(f'y_train_binary, {y_train_binary}')
    # logging.debug(f'y_test_pred, {y_test_pred}')
    
    # logging.debug(f'y, {y}')
    # logging.debug(f'y.shape, {y.shape}')
    # y_tensor is used for computing similarity matrix => supcon loss
    y_tensor = torch.from_numpy(y_train)
    device = (torch.device('cuda')
            if torch.cuda.is_available()
            else torch.device('cpu'))
    encoder = encoder.to(device)
    # DEBUG buggy version
    # y_bin_pred = torch.zeros(X_test.shape[0], 2)
    # y_bin_train_cat = torch.from_numpy(to_categorical(y_train_binary)).float()
    # y_bin_cat_tensor = torch.cat((y_bin_train_cat, y_bin_pred), dim = 0)

    split_tensor = torch.zeros(X_tensor.shape[0]).int()
    split_tensor[test_offset:] = 1
    index_tensor = torch.from_numpy(np.arange(y_tensor.shape[0]))

    all_data = TensorDataset(X_tensor, y_tensor, index_tensor, split_tensor)
    data_loader = DataLoader(dataset=all_data, batch_size=bsize)
    bsize = batch_size
    sample_num = y_tensor.shape[0]
    sum_loss = np.zeros([sample_num])
    cur_sample_loss = np.zeros([sample_num])
    for epoch in range(1, total_epochs + 1):
        # pseudo_loss goes through one epoch, loss for all samples
        sample_loss = pseudo_loss_one_epoch( encoder, data_loader, sample_num, epoch)
        if sample_reduce == 'mean':
            sum_loss += sample_loss
            # average the loss per sample, including both train and test
            cur_sample_loss = sum_loss / epoch
            # only print test sample cur_sample_loss
            print('epoch {}, b {},  (sorted avg loss)[:50] {}'.format(epoch, bsize, sorted(cur_sample_loss[test_offset:], reverse=True)[:50]))
        else:
            # args.sample_reduce == 'max':
            cur_sample_loss = np.maximum(cur_sample_loss, sample_loss)
            # only print test sample cur_sample_loss
            print('epoch {}, b {},  (sorted max loss)[:50] {}'.format(epoch, bsize,  sorted(cur_sample_loss[test_offset:], reverse=True)[:50]))
    return cur_sample_loss

def pseudo_loss_one_epoch(encoder, data_loader, sample_num,xent_lambda  = 100, sample_reduce="mean"):
    """
    measure one epoch of pseudo loss for train + test samples.
    default data points number in an epoch: length_before_new_iter=100000
    """
    
    select_count = torch.zeros(sample_num, dtype=torch.float64)
    total_loss = torch.zeros(sample_num, dtype=torch.float64)
    # if args.sample_reduce == 'mean'
    sample_avg_loss = torch.zeros(sample_num, dtype=torch.float64)
    # if args.sample_reduce == 'max'
    sample_max_loss = torch.zeros(sample_num, dtype=torch.float64)

    idx = 0
    # average the loss for each index in batch_indices
    device = (torch.device('cuda')
                if torch.cuda.is_available()
                else torch.device('cpu'))
    
    for idx, (x_batch, y_batch, y_bin_batch, batch_indices, split_tensor) in enumerate(data_loader):
        # data_time.update(time.time() - end)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_bin_batch = y_bin_batch.to(device)
        _, features, y_pred = encoder(x_batch)
        HiDistanceXent = HiDistanceXentLoss().to(device)
        loss, _, _ = HiDistanceXent(xent_lambda, 
                                    y_pred, y_bin_batch,
                                    features, labels=y_batch,
                                    split = split_tensor)
        loss = loss.to('cpu').detach()
        
        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        select_count[batch_indices] = torch.add(select_count[batch_indices], 1)
        non_select_count = sample_num - torch.count_nonzero(select_count).item()
        # # update the loss values for batch_indices
        if sample_reduce == 'mean':
            total_loss[batch_indices] = torch.add(total_loss[batch_indices], loss)
        if sample_reduce == 'max':
            sample_max_loss[batch_indices] = torch.maximum(sample_max_loss[batch_indices], loss)
            
    # sample average loss
    if sample_reduce == 'mean':
        sample_avg_loss = torch.div(total_loss, select_count)
        return sample_avg_loss.numpy()
    else:
        # args.sample_reduce == 'max':
        return sample_max_loss.numpy()

def train_encoder(encoder, X_train, y_train, y_train_binary,
                  optimizer, total_epochs, model_path, valid_loader,
                  batch_size,
                  device,
                  type_selector,
                  lambda_=100,
                  margin=1,
                  weight=None, upsample=None, adjust=False, warm=False,
                  save_best_loss=False,
                  family_info=False,
                  early_stopping_patience=3):
    # construct the dataset loader
    # y_train is multi-class, y_train_binary is binary class
    print(f"Using device: {device}")
    
    # Debug: Print the shape of the input tensors
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, y_train_binary shape: {y_train_binary.shape}")

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).type(torch.int64)
    y_train_binary_cat_tensor = torch.from_numpy(to_categorical(y_train_binary)).float()
    train_data = TensorDataset(X_train_tensor, y_train_tensor, y_train_binary_cat_tensor)
    bsize = batch_size
    train_loader = DataLoader(dataset=train_data, batch_size=bsize, shuffle=True)
    print(f"Train loader initialized with batch size {batch_size}")
    check_point_path  = model_path+str(os.getpid())+type_selector+".pt"
    best_loss = np.inf
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=check_point_path)
    
    for epoch in range(1, total_epochs + 1):
        # Train for one epoch
        loss, encoder = train_encoder_one_epoch(encoder, train_loader, optimizer, epoch,device=device, lambda_=lambda_,margin=margin,family_info=family_info,type_selector=type_selector,)
        
        if epoch >= total_epochs - 10 and save_best_loss:
            if loss < best_loss:
                best_loss = loss
                print(f"Saving the best loss {loss} model from epoch {epoch}...")
                save_model(encoder, optimizer, epoch, model_path)
        
        encoder.eval()  # prep encoder for evaluation
        if type_selector == "pseudo-loss":
            lr_auc = evaluate_model(
                encoder=encoder,
                valid_loader=valid_loader,
                device=device,
                loss=loss
            )["PR-AUC (Attack)"]
            
            # Check early stopping
            early_stopping(lr_auc, encoder)
            if early_stopping.early_stop:
                print("Early stopping triggered. Ending training...")
                break
        elif type_selector == "cade":
            print(loss)
            early_stopping(-loss,encoder)
            if early_stopping.early_stop:
                print("Early stopping triggered. Ending training...")
                break
    encoder.load_state_dict(torch.load(check_point_path))
    return
def train_encoder_one_epoch(encoder, train_loader, optimizer, epoch,device,type_selector, lambda_=100, margin=1,family_info=False,):
    """ Train one epoch for the model """
    encoder = encoder.to(device)
    losses = AverageMeter()
    supcon_losses = AverageMeter()
    xent_losses = AverageMeter()

    HiDistanceXent = HiDistanceXentLoss().to(device)

    # Wrap train_loader with tqdm for progress bar
    with tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch") as t:
        for idx, (x_batch, y_batch, y_bin_batch) in enumerate(t):
            encoder.train()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_bin_batch = y_bin_batch.to(device)

            # Debug: Print batch details
            bsz = y_bin_batch.shape[0]
            if type_selector == 'cade':
                features, decoded = encoder(x_batch)

                TripletMSE = TripletMSELoss().cuda()
                loss, supcon_loss, mse_loss = TripletMSE(lambda_, \
                                    x_batch, decoded, features, labels=y_batch,margin=margin \
                                    )
                
                # update metric
                losses.update(loss.item(), bsz)
                supcon_losses.update(supcon_loss.item(), bsz)

                # SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()   
                t.set_postfix({
                    "loss": loss.item(),
                    "supcon_loss": supcon_loss.item(),
                    "mse_loss": mse_loss.item(),
                })

            elif type_selector=="pseudo-loss":
                _, cur_f, y_pred = encoder(x_batch)
                features = cur_f
                # Compute loss
                loss, supcon_loss, xent_loss = HiDistanceXent(
                    lambda_,
                    y_pred,
                    y_bin_batch,
                    features,
                    labels=y_batch,
                    margin=margin,
                    family_info=family_info,
                    device=device
                )

                # Update metrics
                losses.update(loss.item(), bsz)
                supcon_losses.update(supcon_loss.item(), bsz)
                xent_losses.update(xent_loss.item(), bsz)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update tqdm bar with metrics
                t.set_postfix({
                    "loss": loss.item(),
                    "supcon_loss": supcon_loss.item(),
                    "xent_loss": xent_loss.item(),
                })
    return losses.avg, encoder
def train_classifier( classifier, X_train, y_train,
                    optimizer, total_epochs,
                batch_size,model_path,
                valid_loader,
                device,
                    save_best_loss = False,
                    save_snapshot = False,family_info=False):
    # contruct the dataset loader
    # y_train is multi-class, y_train_binary is binary class
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    bsize = batch_size
    train_loader = DataLoader(dataset=train_data, batch_size=bsize)
    best_loss = np.inf
    check_point_path  = model_path+str(os.getpid())+"classifier"+".pt"
    early_stopping = EarlyStopping(patience=7, verbose=True, path=check_point_path)
    
    for epoch in range(1, total_epochs + 1):
        # train one epoch
        loss = train_classifier_one_epoch(classifier, train_loader,
                                        optimizer, epoch,device=device)
        # for name, param in classifier.state_dict().items():
        #     print(f"{name}:\n{param}\n")
        classifier.eval()
        lr_auc = evaluate_model(
                encoder=classifier,
                valid_loader=valid_loader,
                device=device,
                loss=loss
            )["PR-AUC (Attack)"]
            
            # Check early stopping
        early_stopping(lr_auc, classifier)
        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training...")
            break
    classifier.load_state_dict(torch.load(check_point_path))

def train_classifier_one_epoch( classifier, train_loader, optimizer, epoch,device, multi = False):
    """ Train one epoch for the model """
    losses = AverageMeter()

    classifier = classifier.to(device)
    
    idx = 0
    with tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch") as t:
        for idx, (x_batch, y_batch, ) in enumerate(t):
            classifier.train()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # DEBUG
            # print(Counter(y_batch.cpu().detach().numpy()))
            # print(y_cat_batch.cpu().detach().numpy())

            bsz = y_batch.shape[0]
            y_multi_pred = classifier.predict_proba(x_batch)
            loss = torch.nn.functional.cross_entropy(y_multi_pred, y_batch)
            # update metric
            losses.update(loss.item(), bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print info, print every mlp_display_interval batches.
            t.set_postfix({
                "loss": loss.item(),
            })
    return losses.avg
def evaluate_model(encoder, valid_loader, device,  loss):
    """
    Evaluate the model on the validation data.
    
    Args:
        encoder (nn.Module): The trained model.
        valid_loader (DataLoader): Validation data loader.
        device (torch.device): Device for computation.
        epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.
        loss (float): Training loss for the current epoch.

    Returns:
        dict: A dictionary with evaluation metrics.
    """
    val_pred = []
    val_gt = []
    
    # Collect predictions and ground truths
    for data, target in valid_loader:
        encoder.eval()
        # print(f"Shape of Data is {data.shape} and {encoder.predict_proba(data.to(device))}")
        pred = encoder.predict_proba(data.to(device))
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)  # Add a batch dimension if it's missing

        # Extract the second column (probability for class 1)
        pred = pred[:, 1]
        # print(f"Shape of the pred is {pred.shape} : and target is {target.shape}")
        pred = pred.reshape(target.shape)
        y_pred = pred.detach().cpu().numpy().tolist()
        val_pred.extend(y_pred)
        val_gt.extend(target.detach().cpu().numpy().tolist())
    
    # Precision-Recall AUC
    lr_precision, lr_recall, _ = precision_recall_curve(val_gt, [x for x in val_pred], pos_label=1)
    lr_auc_minority = auc(lr_recall, lr_precision)
    lr_auc = lr_auc_minority

    # F1 Score
    val_pred_binary = [1 if p >= 0.5 else 0 for p in val_pred]  # Threshold at 0.5
    f1_attack = f1_score(val_gt, val_pred_binary, pos_label=1)
    f1_benign = f1_score(val_gt, val_pred_binary, pos_label=0)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(val_gt, val_pred_binary,labels=[0,1]).ravel()
    # FNR = FN / (FN + TP), FPR = FP / (FP + TN)
    fnr_attack = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr_benign = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_benign = fp / (fp + tp) if (fp + tp) > 0 else 0
    fpr_attack = fn / (fn + tn) if (fn + tn) > 0 else 0

    # Print evaluation metrics
    print_msg = (
        f'PR-AUC (Attack): {lr_auc:.5f} '
        f'F1 (Attack): {f1_attack:.5f}, F1 (Benign): {f1_benign:.5f} '
        f'FNR (Attack): {fnr_attack:.5f}, FNR (Benign): {fnr_benign:.5f} '
        f'FPR (Attack): {fpr_attack:.5f}, FPR (Benign): {fpr_benign:.5f}'
    )
    print(print_msg)
    metrics = {
        "PR-AUC (Attack)": lr_auc_minority,
        "F1 (Attack)": f1_attack,
        "F1 (Benign)": f1_benign,
        "FNR (Attack)": fnr_attack,
        "FNR (Benign)": fnr_benign,
        "FPR (Attack)": fpr_attack,
        "FPR (Benign)": fpr_benign,
    }
    return metrics

def evaluate_model_cl(encoder, valid_loader, device):
    """
    Evaluate the model on the validation data.

    Args:
        encoder (nn.Module): The trained model.
        valid_loader (DataLoader): Validation data loader.
        device (torch.device): Device for computation.

    Returns:
        dict: A dictionary with evaluation metrics.
    """
    val_pred = []
    val_gt = []
    encoder.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for data, target in valid_loader:
            data = data.to(device)
            assert not torch.isnan(data).any(), "NaN detected in input data!"
            # Forward pass
            output = encoder(data)
            # Apply softmax or sigmoid depending on output shape
            if output.shape[1] > 1:
                pred = torch.nn.functional.softmax(output, dim=1)[:, 1]  # Class 1 probability
            else:
                pred = torch.sigmoid(output).squeeze()  # Binary sigmoid output
            # Ensure shape consistency
            pred = pred.reshape(target.shape)
            # Convert to numpy
            y_pred = pred.detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_gt.extend(target.detach().cpu().numpy().tolist())

    # PrecisionRecall AUC
    lr_precision, lr_recall, _ = precision_recall_curve(val_gt, val_pred, pos_label=1)
    lr_auc_minority = auc(lr_recall, lr_precision)
    # F1 Score
    val_pred_binary = [1 if p >= 0.5 else 0 for p in val_pred]  # Threshold at 0.5
    f1_attack = f1_score(val_gt, val_pred_binary, pos_label=1)
    f1_benign = f1_score(val_gt, val_pred_binary, pos_label=0)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(val_gt, val_pred_binary, labels=[0,1]).ravel()
    fnr_attack = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr_benign = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_benign = fp / (fp + tp) if (fp + tp) > 0 else 0
    fpr_attack = fn / (fn + tn) if (fn + tn) > 0 else 0

    # Print evaluation metrics
    print_msg = (
        f'PRAUC (Attack): {lr_auc_minority:.5f} '
        f'F1 (Attack): {f1_attack:.5f}, F1 (Benign): {f1_benign:.5f} '
        f'FNR (Attack): {fnr_attack:.5f}, FNR (Benign): {fnr_benign:.5f} '
        f'FPR (Attack): {fpr_attack:.5f}, FPR (Benign): {fpr_benign:.5f}'
    )
    print(print_msg)
    metrics = {
        "PRAUC (Attack)": lr_auc_minority,
        "F1 (Attack)": f1_attack,
        "F1 (Benign)": f1_benign,
        "FNR (Attack)": fnr_attack,
        "FNR (Benign)": fnr_benign,
        "FPR (Attack)": fpr_attack,
        "FPR (Benign)": fpr_benign,
    }
    return metrics
   