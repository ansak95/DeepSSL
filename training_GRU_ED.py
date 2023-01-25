# import deepspeed
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
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

import torch.nn.functional as F
from torch.nn.modules import ModuleList
from torch.nn.modules.normalization import LayerNorm
from torch.cuda.amp import autocast
from GRU_ED import GRU_ED






def training_args():
    parser=argparse.ArgumentParser(description='GRU')
    
    parser.add_argument('--timestep', default=1, type=int,
                        help='Pred timestep')	
    parser.add_argument('--num_gpus', default=1, type=int,
                        help='nb_gpus')
    parser.add_argument('--nlayers', default=4, type=int,
                        help='Number of Layers (default: 2)')
    parser.add_argument('-b', '--batch_size', default=4096, type=int,
                        help='mini-batch size (default: 4096)')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--hidden_size', default=64, type=int,
                        help='Nb_neurons (default: 64)')
    
    parser.add_argument('--device', default=0, type=int,
                        help='which device')
    
    parser.add_argument('--maxlen', default=30, type=int,
                        help='Windows length (default : 30)')
    parser.add_argument('--timestep_pred', default=1, type=int,
                        help='Pred sequence length (default : 1)')
    
    parser.add_argument('--ratio', default=1, type=float,
                        help='Ratio sequence (default: 1)')
    parser.add_argument('--drop', default=0.1, type=float,
                        help='Dropout (default: 0.1)')
    
    
    #     parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()
    return args

# constants
args = training_args()
print(args)
# cmd_args = add_argument()
nb_gauges = 3

device = torch.device('cuda')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bs = args.batch_size
epochs = args.epochs
maxlen = args.maxlen



import os
fd_data = os.path.split(os.getcwd())[0] ##+ '/Data_'+ str(codebook_size) +'Clusters' 

df = pd.read_pickle(fd_data + '/raw_data_train_complete').reset_index().iloc[:,1:]
data_train = df[(df.ID<=95) & (df.cycle != -1)]
data_val = df[(df.ID>9995) & (df.cycle != -1)].reset_index()


# instantiate model
torch.manual_seed(7)
torch.cuda.manual_seed(7)

seq_cols =  ['gauge'+ str(i+1) for i in range(3)]
sequence_length = 30
timesteps_pred = args.timestep_pred


def gen_sequence_autoregressive(id_df, seq_length, seq_cols,timesteps_pred,h, ratio = 1):
    
    ind_start = 0
    data_array = id_df[seq_cols].values
    th = int(ratio*data_array.shape[0])
    data_array = data_array[:th]
    num_elements = data_array.shape[0]
    
    for start, stop in zip(range(0+ind_start, num_elements-seq_length+1-timesteps_pred), range(seq_length+ind_start, num_elements+1-timesteps_pred)):
        yield data_array[start+h:stop+h, :]#,data_array[start:stop, :])
      

def autoregressive_preprocess(data, sequence_length, seq_cols, timestep_pred, type_set = 'float', ratio = 1) :
    
    seq_gen = (list(gen_sequence_autoregressive(data[data['ID']==id], sequence_length, seq_cols, timesteps_pred=timestep_pred, h = 0, ratio = ratio)) 
                   for id in data['ID'].unique() if len(data[data['ID']==id]) >= sequence_length)
    # generate sequences and convert to numpy array
    dbX = np.concatenate(list(seq_gen))#[:,:,:1]
    
    seq_gen = (list(gen_sequence_autoregressive(data[data['ID']==id], sequence_length, seq_cols, timesteps_pred=timestep_pred, h = 0, ratio = ratio)) 
                   for id in data['ID'].unique() if len(data[data['ID']==id]) >= sequence_length)
    # generate sequences and convert to numpy array
    dbY = np.concatenate(list(seq_gen))#[:,:,:1]
    #dbY = dbY[:,-timestep_pred:,:]
    
    print(dbX.shape)
    print(dbY.shape)
    
    
    
    print('Preparing datasets')
    if type_set =='float' :
        X = torch.tensor(dbX, dtype=torch.float)#.to(device)
        Y = torch.tensor(dbY, dtype=torch.float)#.to(device)
    elif type_set =='long' :
        X = torch.tensor(dbX, dtype=torch.long)#.to(device)
        Y = torch.tensor(dbY, dtype=torch.long)#.to(device)

    return TensorDataset(X, Y), X, Y#, dbY.mean(0), dbY.std(0)

from torch.utils.data import TensorDataset, DataLoader

def create_loaders(data, bs=512, jobs=0):
    data = DataLoader(data, bs, shuffle=True, num_workers=jobs, pin_memory = True)
    return data

for rt in [60, 70, 80, 90] :

    
    timesteps_pred = 0
    train_dl, X_train, y_train = autoregressive_preprocess(data_train, sequence_length, seq_cols, timesteps_pred, type_set = 'float', ratio = rt/100) 
    val_dl,  X_val,y_val  = autoregressive_preprocess(data_val, sequence_length, seq_cols, timesteps_pred, type_set = 'float', ratio = rt/100) 
    
    X_trn_full = torch.cat([X_train,X_val],0)
    y_trn_full = torch.cat([y_train,y_val],0)
    full_train_dl = TensorDataset(X_trn_full, y_trn_full)
    
    tmp = X_trn_full[:,-1,:]#.values
    #trn_min = tmp.min(axis=0).reshape(1,-1)#[0]
    #trn_max = tmp.max(axis=0).reshape(1,-1)#[0]
    trn_mean = tmp.mean(axis=0).reshape(1,-1)#[0]
    trn_std = tmp.std(axis=0).reshape(1,-1)#[0]
    print(trn_mean)
    print(trn_std)
    
    bs = args.batch_size
    trn_dl = create_loaders(train_dl, bs, jobs=1)
    val_dl = create_loaders(val_dl, 4096, jobs=1)
    
    
    
    def update_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr
    
    import time
       
        
    hidden_size = args.hidden_size
    nlayers = args.nlayers
    embedding_size = args.hidden_size
    dropout = args.drop
    
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
    
    criterion = nn.MSELoss()
    trn_mean = torch.tensor(trn_mean).float()#.to(device).float()
    trn_std = torch.tensor(trn_std).float()
    bias = torch.tensor([torch.mean((y_train[:,:,k]-trn_mean[0,k])/trn_std[0,k]) for k in range(3)]).to(device)
    model = GRU_ED(input_dim=3,emb_dim = hidden_size, hidden_dim=hidden_size, output_dim=3, n_layers=nlayers, drop_prob=dropout, mean_val = trn_mean, std_val = trn_std, criterion = criterion, init_bias = bias, length_seq = args.timestep_pred)
    nb_params = model.number_of_parameters()
    print(nb_params)
    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)#, betas=(0.9, 0.95), eps=1e-08)
    # criterion = nn.MS#nn.CrossEntropyLoss(weight = class_weights).to(device)
      
        
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)        
    model.to(device)
    
    
    
    #create folder
    dir_path = f"Decoder_t1_{rt}_100"#folder_models + '/'
    os.makedirs(dir_path)
    dir_path = dir_path + '/'
    
    #save the model architecture
    f = open(dir_path+"model_parameters.txt", "a")
    f.write(str(model.state_dict))
    f.close()
    
    # #save the log 
    f = open(dir_path+"log_loss.txt", "a")
    # f.write(str(model.state_dict))
    f.close()
    
    PATH = "model.pth"
    
    #save the args
    f = open(dir_path+"args.txt", "w+")
    f.write(str(args))
    f.close()
    
    #which optimizer
    f = open(dir_path+"optim.txt", "w+")
    f.write(str(optimizer))
    f.close()                    
    
    t0 = time.time()
    
    
    all_trn_mape_track = []
    all_val_mape_track = []
    
    trn_mape_track = []
    val_mape_track = []
    
    
    j = 0  #indicator used to load the model
    k = 0
    best_mape = 10000
    step = 0
    epoch_stop = np.zeros(3)
    #save some useful informations
    infos_model = f'Number of training aircraft components : {len(np.unique(data_train.ID))} \\ Number of validation aircraft components : {len(np.unique(data_val.ID))} \\  Sequence length : {maxlen} \
    \\  Number of training samples : {X_train.shape[0]} \\ Number of validation samples : {X_val.shape[0]} \\ Number of epochs : {k+1} \
    \\ Optimizer learning rate : {lr} \\  Running time in minutes : {(time.time()-t0)/60} \\ Nb model parameters : {model.number_of_parameters()}' 
    f = open(dir_path+"training_readme.txt", "w+")
    f.write(infos_model)
    f.close()
    
    
    
    for l_r in [lr, 1e-3, 1e-4] :
        
        
        if j != 0 :
            f = open(dir_path+"log_loss.txt", "a")
            f.write("Load the model...")
            f.write("\n")
            f.close()
            
            model = load_checkpoint(dir_path+PATH, train = True)
            
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)        
            model.to(device)
            
         
            
            checkpoint = torch.load(dir_path+PATH)
            optimizer = optim.Adam(model.parameters(), lr=l_r)#, betas=(0.9, 0.95), eps=1e-08)
            optimizer.load_state_dict(checkpoint['optimizer_dic'])
            update_lr(optimizer, l_r)
            best_mape = checkpoint['best_mape']   
            trn_mape_track = checkpoint['mape']
            val_mape_track = checkpoint['val_mape']
            epoch_stop = checkpoint['epoch_stop']
    
        j = j+1
        
        # TRAINING    
        f = open(dir_path+"log_loss.txt", "a")
        f.write("Begin training." + "\n")
        f.write('Learning rate adjusted to {:0.7f}'.format(optimizer.param_groups[0]['lr']))
        f.write("\n")
        f.close()
         
        
        
        
        for epoch in range(args.epochs):
            model.train()
        #         patience = patience-1
            t1 = time.time()
            loss = 0
            trn_mape = 0
    
    
            for i, data in enumerate(trn_dl):   
         
            
                X_train_batch, y_train_batch = data[0].to(device),data[1].to(device).float()
                optimizer.zero_grad()
                loss = model(X_train_batch, y_train_batch)[1]
                loss.backward()
                optimizer.step()
    
    
    
            # Eval phase  
            model.eval()
            with torch.no_grad() :
                train_mape = model.eval_mape(trn_dl)#torch.mean(torch.abs((pred_train-y_train.to(device))/y_train.to(device)))#.item()
                trn_mape_track.append(train_mape)
                all_trn_mape_track.append(train_mape)
                
                
                val_mape = model.eval_mape(val_dl)#torch.mean(torch.abs((pred_train-y_train.to(device))/y_train.to(device)))#.item()
                val_mape_track.append(val_mape)
                all_val_mape_track.append(val_mape)
    
    
            f = open(dir_path+"log_loss.txt", "a")
            f.write(f'Epoch {epoch+1}/{args.epochs} in {time.time()-t1}s, mape : {train_mape:2.2%}, val mape : {val_mape:2.2%}')
            f.write("\n")
            f.close()
            
            
            if val_mape < best_mape :
                #trials = 0
                best_mape = val_mape#.item()
                epoch_stop[j:] = k
                
                f = open(dir_path+"log_loss.txt", "a")
                f.write(f'Epoch {epoch+1} best model saved with mape: {val_mape:2.2%}')
                f.write("Save the model...")
                f.write("\n")
                f.close()
        
                checkpoint = {'model': model, 
                              'mape': trn_mape_track, 'val_mape' : val_mape_track, 'all_mape' : all_trn_mape_track , 'all_val_mape' : all_val_mape_track,
                      'state_dict': model.state_dict(), 'best_mape' : best_mape, 'epoch_stop' : epoch_stop,
                      'optimizer_dic' : optimizer.state_dict()}
                torch.save(checkpoint, dir_path+PATH)
                    
            k = k+1        
    
    
    
    for l_r in [1e-5] :
    
        f = open(dir_path+"log_loss.txt", "a")
        f.write("Load the model...")
        f.write("\n")
        f.close()
        
        model = load_checkpoint(dir_path+PATH, train = True)
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)        
        model.to(device)
        
     
        
        checkpoint = torch.load(dir_path+PATH)
        optimizer = optim.Adam(model.parameters(), lr=l_r)#, betas=(0.9, 0.95), eps=1e-08)
        optimizer.load_state_dict(checkpoint['optimizer_dic'])
        update_lr(optimizer, l_r)
        best_mape = 10000#checkpoint['best_mape']   
        trn_mape_track = checkpoint['mape']
        val_mape_track = checkpoint['val_mape']
        epoch_stop = checkpoint['epoch_stop']
    
        j = j+1
        
        # TRAINING    
        f = open(dir_path+"log_loss.txt", "a")
        f.write("Begin training (full set)." + "\n")
        f.write('Learning rate adjusted to {:0.7f}'.format(optimizer.param_groups[0]['lr']))
        f.write("\n")
        f.close()
         
        print(f'Creating data loaders with batch size: {bs}')
        trn_dl = create_loaders(full_train_dl, bs, jobs=1)#4*args.num_gpus)
        #trn_dl_eval = create_loaders(full_train_dl, 4096*4, jobs=1)#4*args.num_gpus)
        torch.cuda.empty_cache()  
            
            
        for epoch in range(args.epochs):
            model.train()
            t1 = time.time()
            loss = 0
            trn_mape = 0
    
    
            for i, data in enumerate(trn_dl):  
                X_train_batch, y_train_batch = data[0].to(device),data[1].to(device).float()
                optimizer.zero_grad()
                loss = model(X_train_batch, y_train_batch)[1]
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad() :
                train_mape = model.eval_mape(trn_dl)#torch.mean(torch.abs((pred_train-y_train.to(device))/y_train.to(device)))#.item()
            trn_mape_track.append(train_mape)
            all_trn_mape_track.append(train_mape)
            
            
            f = open(dir_path+"log_loss.txt", "a")
            f.write(f'Epoch {epoch+1}/{args.epochs} in {time.time()-t1}s, mape : {train_mape:2.2%}')
            f.write("\n")
            f.close()
            
            if train_mape < best_mape :
                #trials = 0
                best_mape = train_mape#.item()
                epoch_stop[j:] = k
                
                f = open(dir_path+"log_loss.txt", "a")
                f.write(f'Epoch {epoch+1} best model saved with mape: {train_mape:2.2%}')
                f.write("Save the model...")
                f.write("\n")
                f.close()
        
                checkpoint = {'model': model, 
                              'mape': trn_mape_track, 'val_mape' : val_mape_track, 'all_mape' : all_trn_mape_track , 'all_val_mape' : all_val_mape_track,
                      'state_dict': model.state_dict(), 'best_mape' : best_mape, 'epoch_stop' : epoch_stop,
                      'optimizer_dic' : optimizer.state_dict()}
                torch.save(checkpoint, dir_path+PATH)
            
            
    
    
    #save some useful informations
    infos_model = f'Number of training aircraft components : {len(np.unique(data_train.ID))} \\ Number of validation aircraft components : {len(np.unique(data_val.ID))} \\  Sequence length : {maxlen} \
    \\  Number of training samples : {X_train.shape[0]} \\ Number of validation samples : {X_val.shape[0]} \\  Number of epochs : {k+1} \
    \\ Optimizer learning rate : {lr} \\  Running time in minutes : {(time.time()-t0)/60} \\ Nb model parameters : {model.number_of_parameters()}' 
    f = open(dir_path+"training_readme.txt", "w+")
    f.write(infos_model)
    f.close()            
    
    #del model and empty cache
    del(model)
    torch.cuda.empty_cache()       
