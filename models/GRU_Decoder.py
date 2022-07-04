import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.normalization import LayerNorm
#from torchviz import make_dot
from torch.autograd import Variable
from torch.nn.modules import ModuleList
import copy


import numpy as np
import os
from tqdm import tqdm_notebook, trange
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class GRU_Decoder(nn.Module):
    def __init__(self,  input_dim, emb_dim, hidden_dim, output_dim, n_layers, drop_prob, mean_val, std_val, criterion, init_bias, length_seq):
        super(GRU_Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.length_seq = length_seq
        
        
        self.encoder = nn.GRU(emb_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob).to(device)
        self.emb = nn.Linear(input_dim, emb_dim).to(device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)
        
        self.criterion = criterion.to(device)
        self.factor = -1
        self.act = nn.SiLU()
        self.mean = mean_val.to(device)
        self.std = std_val.to(device)
        self.ft = False
        self.drop = nn.Dropout(0)
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.init_bias = init_bias
        self.init_weights()
    
    def init_weights(self):
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)) and (module.bias is not None) :#and ():
            nn.init.xavier_normal_(module.weight.data)
            if module.weight.shape[0] == self.output_dim : 
                module.bias.data = module.bias.data + self.init_bias


        elif isinstance(module, nn.GRU)  :
            for layer_p in module._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.xavier_normal_(module.__getattr__(p))
                    
                         
    def transform_minmax(self, input) :
        return (input-self.mean)/self.std
    
    def invtransform_minmax(self, input) :
        return input*self.std+self.mean
    

    def forward(self, input, y = None):

        input = self.transform_minmax(input).to(device)       
        x = self.emb(input)
        x = self.norm(x)
        x = self.drop(x)
        
        memory, cn = self.encoder(x) 
        inp = x + self.norm1(memory)
        
        out = self.factor*self.act(inp)

        if self.ft == False :	
            out = self.fc(out[:,-self.length_seq:,:])
        else :
            out = self.fc(out,cn[-1:])
        
       
       
        if y != None :
            y = self.transform_minmax(y)
            loss = self.criterion(out,y)
            return out, loss, memory, cn, inp
        else : 
            return out, memory, cn,inp
        
    def train_model(self, loader, optimizer) :
        loss = 0   
        for i, data in enumerate(loader):   
            X_train_batch, y_train_batch = data[0].cuda(),data[1].cuda().float()  #torch.cuda.device_count()
            optimizer.zero_grad()
            loss = self.forward(X_train_batch, y_train_batch)[1]#[0].reshape(-1)       
            loss.backward()
            optimizer.step()
        
    def eval_mape(self, loader) :
        metric_mape = 0
        with torch.no_grad() :
            for i, data in enumerate(loader):   
                x, y = data[0].to(device),data[1].to(device).float()
                y_out = self.invtransform_minmax(self.forward(x)[0])
                metric_mape += torch.mean(torch.abs((y_out-y)/y)).item()
        return metric_mape/(i+1)
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))





