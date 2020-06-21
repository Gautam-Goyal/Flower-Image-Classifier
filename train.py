##
# author=Gautam
#Created=15 June 2020
##

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import train_save_functions

ap = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', default = 'MASTER_CHECKPOINT.pth', type=str, help='set the checkpoint path')
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=14)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=5024)
ap.add_argument('--dropout', type=float, dest="dropout", action="store", default=0.2)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
learningr = pa.learning_rate
architect = pa.arch
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs
drop=pa.dropout

print('Loading_data........')
train_loaders,valid_loaders,test_loaders,train_datasets=train_save_functions.Load_Data(where)

print('Setting up parameters.......')
model,criterion,device,hidden_layer= train_save_functions.setup_para(architect,hidden_layer1,learningr,power)

print('*************************')
print('*************************')
print('TRAINING NETWORK......')
model,optimizer=train_save_functions.train_network(model,criterion,device,train_loaders,valid_loaders,epochs,learningr)

print('##########################')
print("Saving checkpoint......\n")
train_save_functions.save_checkpoint(train_datasets,model,optimizer,path,architect,epochs,learningr)


print("Hoorayyy....The Model is trained") # Coffee timeee