import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
from torchvision.models import resnet18
from collections import OrderedDict
from torch.nn.init import kaiming_uniform_
from torch.nn import Module,Linear,ReLU,Sigmoid
from torch.nn.init import xavier_uniform_
from torch.nn import Parameter
import torch.nn.utils as utils
import math
from utils.config.configurations import cfg


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)

#Adding Identity actiovation function
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    

class APIGRAPH_FC(nn.Module):
    def __init__(self,inputsize, softmax=False,dropout=0.2):
        super(APIGRAPH_FC, self).__init__()
        self.softmax=softmax
        self.act=OrderedDict()
        drop_out_ratio=dropout
        self.hidden1 = Linear(inputsize,100,bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(drop_out_ratio)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 250,bias=False)
        self.bn2 = nn.BatchNorm1d(250)
        self.dropout2 = nn.Dropout(drop_out_ratio)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        self.hidden3 = Linear(250, 500,bias=False)
        self.bn3 = nn.BatchNorm1d(500)
        self.dropout3 = nn.Dropout(drop_out_ratio)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden3.weight)
        self.act3 = ReLU()
        self.hidden4 = Linear(500, 150,bias=False)
        self.bn4 = nn.BatchNorm1d(150)
        self.dropout4 = nn.Dropout(drop_out_ratio)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden4.weight)
        self.act4 = ReLU()
        self.hidden5 = Linear(150, 50,bias=False)
        self.bn5 = nn.BatchNorm1d(50)
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

        return self.forward_classifier(encoded_X)

class EMBER_FC(nn.Module):
    def __init__(self,inputsize, softmax=False):
        sizes = [inputsize,100,250,500,150,50,1]
        super(APIGRAPH_FC, self).__init__()
        layers = []
        self.softmax=softmax
        
        
        self.act=OrderedDict()
        self.hidden1 = Linear(inputsize,100,bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 250,bias=False)
        self.bn2 = nn.BatchNorm1d(250)
        self.dropout2 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        self.hidden3 = Linear(250, 500,bias=False)
        self.bn3 = nn.BatchNorm1d(500)
        self.dropout3 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden3.weight)
        self.act3 = ReLU()
        self.hidden4 = Linear(500, 150,bias=False)
        self.bn4 = nn.BatchNorm1d(150)
        self.dropout4 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden4.weight)
        self.act4 = ReLU()
        self.hidden5 = Linear(150, 50,bias=False)
        self.bn5 = nn.BatchNorm1d(50)
        self.dropout5 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = ReLU()
        self.hidden6 = Linear(50,2,bias=False)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act6 = ReLU()

    def forward_encoder(self, X):
        X = X.float()
        X = self.hidden1(X)
        X = self.bn1(X)
        X = self.act1(self.dropout1(X))

        X = self.hidden2(X)
        X = self.bn2(X)
        X = self.act2(self.dropout2(X))

        X = self.hidden3(X)
        X = self.bn3(X)
        X = self.act3(self.dropout3(X))

        X = self.hidden4(X)
        X = self.bn4(X)
        X = self.act4(self.dropout4(X))

        X = self.hidden5(X)
        X = self.bn5(X)
        X = self.act5(self.dropout5(X))

        return X  # Encoded output

    def forward_classifier(self, encoded_X):
        X = self.hidden6(encoded_X)
        X = self.act6(X)
        if self.softmax:
            X = torch.softmax(X, dim=1).squeeze()
        return X

    def forward(self, X):
        encoded_X = self.forward_encoder(X)
        return self.forward_classifier(encoded_X)
class ANDORZOO_FC(nn.Module):
    def __init__(self,inputsize, softmax=False):
        sizes = [inputsize,100,250,500,150,50,1]
        super(APIGRAPH_FC, self).__init__()
        layers = []
        self.softmax=softmax
        
        
        self.act=OrderedDict()
        self.hidden1 = Linear(inputsize,100,bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 250,bias=False)
        self.bn2 = nn.BatchNorm1d(250)
        self.dropout2 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        self.hidden3 = Linear(250, 500,bias=False)
        self.bn3 = nn.BatchNorm1d(500)
        self.dropout3 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden3.weight)
        self.act3 = ReLU()
        self.hidden4 = Linear(500, 150,bias=False)
        self.bn4 = nn.BatchNorm1d(150)
        self.dropout4 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden4.weight)
        self.act4 = ReLU()
        self.hidden5 = Linear(150, 50,bias=False)
        self.bn5 = nn.BatchNorm1d(50)
        self.dropout5 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = ReLU()
        self.hidden6 = Linear(50,2,bias=False)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act6 = ReLU()

    def forward_encoder(self, X):
        X = X.float()
        X = self.hidden1(X)
        X = self.bn1(X)
        X = self.act1(self.dropout1(X))

        X = self.hidden2(X)
        X = self.bn2(X)
        X = self.act2(self.dropout2(X))

        X = self.hidden3(X)
        X = self.bn3(X)
        X = self.act3(self.dropout3(X))

        X = self.hidden4(X)
        X = self.bn4(X)
        X = self.act4(self.dropout4(X))

        X = self.hidden5(X)
        X = self.bn5(X)
        X = self.act5(self.dropout5(X))

        return X  # Encoded output

    def forward_classifier(self, encoded_X):
        X = self.hidden6(encoded_X)
        # X = self.act6(X)
        if self.softmax:
            X = torch.softmax(X, dim=1).squeeze()
        return X

    def forward(self, X):
        encoded_X = self.forward_encoder(X)
        return self.forward_classifier(encoded_X)
class BODMAS_FC(nn.Module):
    def __init__(self,inputsize, softmax=False):
        sizes = [inputsize,100,250,500,150,50,1]
        super(BODMAS_FC, self).__init__()
        layers = []
        self.softmax=softmax

        self.act=OrderedDict()
        self.hidden1 = Linear(inputsize,100,bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 250,bias=False)
        self.bn2 = nn.BatchNorm1d(250)
        self.dropout2 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        self.hidden3 = Linear(250, 500,bias=False)
        self.bn3 = nn.BatchNorm1d(500)
        self.dropout3 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden3.weight)
        self.act3 = ReLU()
        self.hidden4 = Linear(500, 150,bias=False)
        self.bn4 = nn.BatchNorm1d(150)
        self.dropout4 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden4.weight)
        self.act4 = ReLU()
        self.hidden5 = Linear(150, 50,bias=False)
        self.bn5 = nn.BatchNorm1d(50)
        self.dropout5 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = ReLU()
        self.hidden6 = Linear(50,2,bias=False)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act6 = ReLU()

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
        X = self.hidden6(X)
        X = self.act6(X)
        
        return X  # Encoded output

    def forward_classifier(self, encoded_X):
        X = self.hidden6(encoded_X)
        X = self.act6(X)
        if self.softmax:
            X = torch.softmax(X, dim=1).squeeze()
        return X

    def forward(self, X):
        encoded_X = self.forward_encoder(X)
        return self.forward_classifier(encoded_X)

