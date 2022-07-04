import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.modules import ModuleList
from torch.nn.modules.normalization import LayerNorm

import pickle
import pandas as pd
import datetime 
import matplotlib.pyplot as plt
from IPython.display import display
import os
import argparse
import random
import tqdm
import time
import numpy as np
from time import time











def load_checkpoint(filepath, train = False):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    
    if train :
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.train()
    else :
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



#prepare forecasting data
def gen_RUL_sequence(id_df, seq_length, seq_cols, type_data = 'Input', ind_start = 0):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    if type_data == 'Input' :
        for start, stop in zip(range(0+ind_start, num_elements-seq_length+1), range(seq_length+ind_start, num_elements+1)):
            yield data_array[start:stop, :]
    else :
        for start, stop in zip(range(0+ind_start, num_elements-seq_length+1), range(seq_length+ind_start, num_elements+1)):
            yield data_array[stop-1, :]



def seq_preprocess(data, sequence_length, seq_cols_in, seq_cols_out, type_set = 'Train', num_type = 'float') :
    

	
    #generate sequences and convert to numpy array

    if type_set == 'Test' :
        dbX = [data[data['ID']==id][seq_cols_in].values[-sequence_length:] for id in data['ID'].unique()]
        dbX = np.asarray(dbX)
        dbY = [data[data['ID']==id][seq_cols_out].values[-1] for id in data['ID'].unique()]
        dbY = np.asarray(dbY)

    else :	
        seq_gen = (list(gen_RUL_sequence(data[data['ID']==id], sequence_length, seq_cols_in, type_data= 'Input')) for id in data['ID'].unique())
        dbX = np.concatenate(list(seq_gen))
        seq_gen = (list(gen_RUL_sequence(data[data['ID']==id], sequence_length, seq_cols_out,  type_data= 'Output')) for id in data['ID'].unique())
        dbY = np.concatenate(list(seq_gen)).reshape(-1,)

    print(dbX.shape)
    print(dbY.shape)
    


    

    
    print('Preparing datasets')
    if num_type =='float' :
        torch_type = torch.float
        Y = torch.tensor(dbY, dtype=torch.float)#.to(device)       
    elif num_type =='long' :
        torch_type = torch.long
        Y = torch.tensor(dbY, dtype=torch.long)#.to(device)
    
    X = torch.tensor(dbX, dtype=torch.float)#.to(device)
    
    print('Preparing datasets')
    
    X_torch = torch.tensor(X, dtype=torch.float)
    y_torch = torch.tensor(Y, dtype=torch_type)



    return X_torch, y_torch
