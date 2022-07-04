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


class VGRU_ED(nn.Module):
    def __init__(self,  input_dim, emb_dim, hidden_dim, output_dim, n_layers, drop_prob, mean_val, std_val, criterion, init_bias, length_seq, weight_kl = 5e-4):
        super(VGRU_ED, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.length_seq = length_seq
        self.ft = False
        self.w = weight_kl
        
        self.encoder = nn.GRU(emb_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob).to(device)
        self.fc_enc_mu = nn.Sequential(nn.Linear(64,64),
              nn.Dropout(0.1),
              nn.LayerNorm(64, elementwise_affine=False),                      
              nn.GELU(),
              nn.Linear(64,64)).to(device)
        
        self.fc_enc_logvar = nn.Sequential(nn.Linear(64,64),
              nn.Dropout(0.1),
              nn.LayerNorm(64, elementwise_affine=False),                      
              nn.GELU(),
              nn.Linear(64,64)).to(device)
        
        
        self.decoder = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob).to(device)
        self.emb = nn.Linear(input_dim, emb_dim).to(device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)
        
        self.criterion = criterion.to(device)
        self.factor = -1
        self.act = nn.SiLU()
        self.mean = mean_val.to(device)
        self.std = std_val.to(device)

        self.drop = nn.Dropout(0)
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
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
    
    def reparameterize(self, mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
        
    def forward(self, input, y = None):

        input = self.transform_minmax(input).to(device)       
        
        #embed
        x = self.emb(input)
        x = self.norm(x)
        #x = self.drop(x)
        
        #encode
        memory, context = self.encoder(x) 
        memory = self.act(x + self.norm1(memory)) #z
        
        # Split the result embedding into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_enc_mu(memory)
        log_var = self.fc_enc_logvar(memory)
        
        #compute the latent embedding
        z = self.reparameterize(mu, log_var)
        
        
        if self.ft == False :
          if self.train :
            #decode
            out, cn = self.decoder(z,context)
            out = self.act(z + self.norm2(out))
            
          else :
            #decode
            out, cn = self.decoder(mu,context)
            out = self.act(mu + self.norm2(out))
          
          #linear layer
          out = self.fc(self.factor*out)
        
        else :
          out = self.fc(mu, context[-1:]) #keeps only the mean, not the latent embedding
        
       
        if y != None :
            y = self.transform_minmax(y)
            reconstruction_error = self.criterion(out,y)
            kl_divergence = (-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp()))
            loss = (reconstruction_error + self.w*kl_divergence).sum()
            return out, loss, memory, mu, z
        else : 
            return out, memory, mu, z
        
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





