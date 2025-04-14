import math
import torch
import numpy as np
# def adjust_learning_rate(learning_rate, optimizer, epoch, warm = False):
#     lr = learning_rate
#     # use the same learning rate scheduler in active learning and initial training
#     steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
#     if steps > 0:
#             lr = lr * (args.lr_decay_rate ** steps)
#     else:
#         raise Exception('scheduler {args.scheduler} not supported yet.}')
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
        
#     return lr
def save_model(model, optimizer, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
class AverageMeter(object):
    """Computes and stores the average and current value"""
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
def to_categorical(y, num_classes=2):
    y = np.array(y, dtype='int').ravel()
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))  # Always create an array with 2 classes
    categorical[np.arange(n), y] = 1
    return categorical