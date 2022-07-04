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

from utils import utils_ft

import warnings
warnings.filterwarnings("ignore")



# !pip3 install pandas==0.25.1
def training_args():
    parser=argparse.ArgumentParser(description='fine_tune')

    parser.add_argument('--path', default='', type=str,
                        help='model path')
    parser.add_argument('--folder', default='', type=str,
                        help='folder path')
#     parser.add_argument('--folder_data', default='', type=str,
#                         help='folder data path')
    
    parser.add_argument('--reset', default='no', type=str,
                        help='Reset weights ?')
    
    parser.add_argument('--freeze', default='freeze', type=str,
                        help='Freeze weights')
    parser.add_argument('--finetune', default=False, type=bool,
                        help='Finetune')
    parser.add_argument('--cv', default=5, type=int,
                        help='k fold')

    parser.add_argument('--num_gpus', default=1, type=int,
                        help='nb_gpus')
    parser.add_argument('--nb_samples', default=10, type=int,
                        help='Number of samples (default: 10)')
    parser.add_argument('-b', '--batch_size', default=4096, type=int,
                        help='mini-batch size (default: 4096)')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='number of total epochs (default: 30)')
    
    parser.add_argument('--device', default=0, type=int,
                        help='which device')
    
    parser.add_argument('--maxlen', default=30, type=int,
                        help='Windows length (default : 30)')

    parser.add_argument('--nb_gauges', default=3, type=int,
                        help='Number of gauges (default : 3)')	
    parser.add_argument('--thinning', default=500, type=int,
                        help='Thinning (default : 500)')	

    
    parser.add_argument('--lr', default=1e-6, type=float,
                        help='Learning rate')


    parser.add_argument('--drop', default=0.1, type=float,
                        help='Dropout (default: 0.1)')
    

    args=parser.parse_args()
    return args

def create_loaders(data, bs=512, jobs=0):
    data = DataLoader(data, bs, shuffle=True, num_workers=jobs, pin_memory = False)
    return data


class GRU_Layer(nn.Module):
    def __init__(self,  input_dim, hidden_dim, n_layers, drop_prob):
        super(GRU_Layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob).to(device)
        
    def forward(self, x,hidden):
        out, cn = self.gru(x, hidden)    
        return nn.SiLU()(out)

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
        
        
if __name__ == "__main__":



    args = training_args()
    print(args)

    #set variables
    nb_gauges = args.nb_gauges
    device = torch.device('cuda')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs = args.batch_size
    epochs = args.epochs
    maxlen = args.maxlen
    nb_samples = args.nb_samples
    thinning = args.thinning
    cv = args.cv
    patience = 500

    # instantiate model
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    #set data folder dir and load data
    os.chdir("/home/anassakrim/FolderThesis/ProjectSSL/")
    fd_data = 'data/'
#     fd_km = fd + args.folder_data
    data_train_raw = pd.read_pickle(fd_data + '/data_train_ft').reset_index().iloc[:,1:]
    data_test = pd.read_pickle(fd_data + '/data_test').reset_index().iloc[:,1:]  
    data_train = data_train_raw[data_train_raw.ID <= nb_samples]

    #prepare data
    seq_cols_in =  ['gauge'+ str(i+1) for i in range(3)]
    seq_cols_out =  ['RUL']
    sequence_length = maxlen

    #prepare variables for model prediction
    tmp = data_train[seq_cols_in].values
    trn_mean = tmp.mean(axis=0).reshape(1,-1)#[0]
    trn_std = tmp.std(axis=0).reshape(1,-1)#[0]
    trn_mean = torch.tensor(trn_mean).float()#.to(device).float()
    trn_std = torch.tensor(trn_std).float()
    bias = np.log(data_train[seq_cols_out].values+500).mean() #for fine_tuning

    #prepare test set
    X_test , y_test  = utils_ft.seq_preprocess(data_test , sequence_length, seq_cols_in, seq_cols_out, type_set = 'Test' )
    y_test = y_test.to(device).reshape(-1)
    X_test = X_test.to(device)

#     criterion = nn.MSELoss().to(device)
    # from GRU_Decoder import GRU_Decoder


    #create folder
    fd = args.folder
    dt = f"{datetime.datetime.now():%Y%h%d_%Hh%M}" #datetime
    path_model = fd + "/L2_PT_Test_k_fold_" + dt + '_' + f"Finetune{nb_samples}" 
    os.makedirs(path_model)
    dir_path = path_model + "/"
#     import time
    
    
    #set dir in order to load pre trained model
    mdls = args.path 
    model_path = mdls + 'model.pth'


    it = 0
    lr = args.lr
    #in order to get saved data from used model and optimizer
    model = utils_ft.load_checkpoint(model_path,train=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-6)


    #save the model architecture
    f = open(dir_path+"model_parameters.txt", "a")
    f.write(str(model.state_dict))
    f.close()

    #save the log 
    f = open(dir_path+"log_loss.txt", "a")
    # f.write(str(model.state_dict))
    f.close()


    #save the args
    f = open(dir_path+"args.txt", "w+")
    f.write(str(args))
    f.close()

    #which optimizer
    f = open(dir_path+"optim.txt", "w+")
    f.write(str(optimizer))
    f.close()                    

    t0 = time.time()

    scores_val = []
    scores_test = []
    for i in range(0,nb_samples,nb_samples//cv) :

        patience = 500
        best_mape = 10000
        saved_mape_test = 10000
        model = utils_ft.load_checkpoint(model_path,train=True)

        
        if args.reset == 'reset' :
            print('Reset weights...')
            for m in model.children() :
                if len([p for p in m.parameters()])!= 0 : #only sub layers that have trainable parameters 
                    m.reset_parameters()
    
            model.std = trn_std.to(device)
            model.mean = trn_mean.to(device)
            num_ftrs = model.fc.in_features
            model.fc = mySequential(GRU_Layer(num_ftrs, num_ftrs, 1, 0.1), nn.LayerNorm(num_ftrs, elementwise_affine=False), nn.Linear(num_ftrs,1)).to(device)
            model.length_seq = 30
            model.ft = True
            nn.init.xavier_normal_(model.fc[-1].weight.data)
            model.fc[-1].bias.data = model.fc[-1].bias.data +bias
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-6)

        else : 

            if args.freeze == 'freeze' :
                print('Freeze pre-trained layers')	
                ft_extract = True
                utils_ft.set_parameter_requires_grad(model, ft_extract)
                num_ftrs = model.fc.in_features
                model.fc = mySequential(GRU_Layer(num_ftrs, num_ftrs, 1, 0.1), nn.LayerNorm(num_ftrs, elementwise_affine=False), nn.Linear(num_ftrs,1)).to(device)
                model.length_seq = 30
                model.ft = True
                nn.init.xavier_normal_(model.fc[-1].weight.data)
                model.fc[-1].bias.data = model.fc[-1].bias.data +bias
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-6)


            else :	
                print('Unfreeze pre-trained layers')
                ft_extract = False
#                 utils_ft.set_parameter_requires_grad(model, ft_extract)
                num_ftrs = model.fc.in_features
                model.fc = mySequential(GRU_Layer(num_ftrs, num_ftrs, 1, 0.1), nn.LayerNorm(num_ftrs, elementwise_affine=False), nn.Linear(num_ftrs,1)).to(device)
                model.length_seq = 30
                model.ft = True
                nn.init.xavier_normal_(model.fc[-1].weight.data)#*10
                model.fc[-1].bias.data = model.fc[-1].bias.data +bias
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-6)
    #             optimizer = optim.Adam([{'params': model.parameters()}, {'params': filter(lambda p : p == model.fc.parameters(), model.parameters()), 'lr': lr}], lr=lr/10, weight_decay = 1e-6)


        criterion = nn.MSELoss().to(device)
        it = it+1
        PATH = f"Fold{it}_model.pth"
        print('----------------------------------')
        print(f'Fold {it}')
        print('----------------------------------')
        list_train = np.arange(1,nb_samples+1)
        list_val = list_train[i:i+nb_samples//cv]
        idx = np.arange(i,i+nb_samples//cv)
        list_train = np.delete(list_train,idx)

        data_train = data_train_raw[data_train_raw.ID.isin(list_train)]
        data_val = data_train_raw[data_train_raw.ID.isin(list_val)]


        X_train, y_train = utils_ft.seq_preprocess(data_train, sequence_length, seq_cols_in, seq_cols_out, type_set = 'Train')
        X_val, y_val = utils_ft.seq_preprocess(data_val, sequence_length, seq_cols_in, seq_cols_out, type_set = 'Val')


        train_dl = TensorDataset(X_train, torch.log(y_train+500))
        val_dl   = TensorDataset(X_val, torch.log(y_val+500))

        print(f'Creating data loaders with batch size: {bs}')
        trn_dl = create_loaders(train_dl, bs, jobs=4)
        val_dl = create_loaders(val_dl, bs, jobs=4)


        trn_mape_track = []
        val_mape_track = []



        #save some useful informations
        infos_model = f'Number of training aircraft components : {len(np.unique(data_train.ID))} \\  Sequence length : {maxlen} \
        \\  Number of training samples : {X_train.shape[0]}  \\ Number of epochs : {1} \
        \\ Optimizer learning rate : {args.lr} \\  Running time in minutes : {(time.time()-t0)/60} \\ Nb model parameters : {model.number_of_parameters()}' 
        f = open(dir_path+"training_readme.txt", "w+")
        f.write(infos_model)
        f.close()
        y_train = y_train.to(device).reshape(-1)
        y_val = y_val.to(device).reshape(-1)

        X_train= X_train.to(device)
        X_val= X_val.to(device)
        from tqdm import trange
        it_lr = 0
        for l_r in [args.lr/(10**p) for p in range(3)] :
            patience = 500 #reset patience variable
            if it_lr != 0 : #i.e. if 2nd fold or more
                print("Load the model...")
                f = open(dir_path+"log_loss.txt", "a")
                f.write("Load the model...")
                f.write("\n")
                f.close()

                model = utils_ft.load_checkpoint(dir_path+PATH, train = True) 

                if (args.freeze == 'freeze') and (args.reset == 'no') :
                    utils_ft.set_parameter_requires_grad(model, ft_extract) #freeze all weights
                    for param in model.fc.parameters(): #unfreeze weights of final layer
                        param.requires_grad = True

                model.to(device)
                checkpoint = torch.load(dir_path+PATH)

                optimizer = optim.Adam(model.parameters(), lr=l_r, weight_decay = 1e-6)
                optimizer.load_state_dict(checkpoint['optimizer_dic'])
                update_lr(optimizer, l_r)


            it_lr = it_lr + 1



            # TRAINING    
            print('Learning rate adjusted to {:0.7f}'.format(optimizer.param_groups[0]['lr']))
            f = open(dir_path+"log_loss.txt", "a")
            f.write("Begin training." + "\n")
            f.write('Learning rate adjusted to {:0.7f}'.format(optimizer.param_groups[0]['lr']))
            f.write("\n")
            f.close()

            pbar = trange(args.epochs,  unit="epoch")
            for epoch in pbar:
                time.sleep(0.1)
                model.train()
                t1 = time.time()

                loss = 0
                mape_loss = 0



                for i, data in enumerate(trn_dl):   
                    X_train_batch, y_train_batch = data[0].to(device),data[1].to(device).float()
                    optimizer.zero_grad()

                    y_train_pred = model(X_train_batch)[0][:,-1,:].reshape(-1)

                    mse_loss = criterion(y_train_pred, y_train_batch)
                    mse_loss.backward()
                    optimizer.step()


                model.eval()
                with torch.no_grad() :
                    x = X_train
                    pred_train = model(x)[0][:,-1,:].reshape(-1)
                    pred_train = torch.exp(pred_train)-500 #the variable Y was transformed into tr(Y) = log(Y) - 500
                    train_mape = torch.abs((pred_train-y_train)/y_train).detach()
                    train_mape = torch.mean(train_mape.masked_fill(train_mape.isinf(),0))
                    trn_mape_track.append(train_mape.item())

                    x = X_val
                    pred_val = model(x)[0][:,-1,:].reshape(-1)
                    pred_val = torch.exp(pred_val)-500  #the variable Y was transformed into tr(Y) = log(Y) - 500
                    val_mape = torch.abs((pred_val-y_val)/y_val).detach()
                    val_mape = torch.mean(val_mape.masked_fill(val_mape.isinf(),0))
                    val_mape_track.append(val_mape.item())

                    x = X_test
                    pred_test = model(x)[0][:,-1,:].reshape(-1)
                    pred_test = torch.exp(pred_test)-500 #the variable Y was transformed into tr(Y) = log(Y) - 500
                    test_mape = torch.mean(torch.abs((pred_test-y_test)/y_test)).item()

                pbar.set_description(f'Epoch {epoch+1}/{args.epochs}')
                pbar.set_postfix_str(f'Train set mape {train_mape:2.2%}, Val set mape {val_mape:2.2%}, Test set mape {test_mape:2.2%}, Best mape {best_mape:2.2%}, Saved test mape {saved_mape_test:2.2%}, Patience = {patience}')
                
                f = open(dir_path+"log_loss.txt", "a")
                f.write(f'Epoch {epoch+1}/{args.epochs} in {time.time()-t1}s, Train set mape {train_mape:2.2%}, Val set mape {val_mape:2.2%}, Test set mape {test_mape:2.2%}, Best mape {best_mape:2.2%}, Saved test mape {saved_mape_test:2.2%}, Patience = {patience}')
                f.write("\n")
                f.close()



                if epoch%100 == 0 :   
                    torch.cuda.empty_cache() 
                if val_mape < best_mape :
                    epoch_save = epoch
                    patience = 500
                    saved_mape_test = test_mape
                    best_mape = val_mape
                    f = open(dir_path+"log_loss.txt", "a")
                    f.write("Save the model...")
                    f.write("\n")
                    f.close()

                    checkpoint = {'model': model, 'mape': trn_mape_track, 'val_mape': val_mape_track,  
                          'state_dict': model.state_dict(),
                          'optimizer_dic' : optimizer.state_dict(), 'lr' : lr}
                    torch.save(checkpoint, dir_path+PATH)
                else : 
                    patience = patience - 1

                if patience == 0 :
                    pbar.set_postfix_str(f'Ended, Best mape {best_mape:2.2%}, Saved mape test {saved_mape_test:2.2%}')
                    break


        plt.rc('figure',figsize=(22,12))
        plt.rcParams.update({'font.size': 22})
        plt.rc('ytick', labelsize=18)
        plt.rc('xtick', labelsize=18)


        plt.plot(trn_mape_track, label =  f'Training set')
        plt.plot(val_mape_track, label =  f'Val set with best mape = {best_mape.cpu().numpy():2.2%}')

        # plt.ylim(0,20)
        plt.grid()
        plt.legend()
        plt.title(f'{nb_samples} training structures, Fold {it}, Test set with {saved_mape_test:2.2%}')

        plt.xlabel('Epoch')
        plt.ylabel('MAPE (%)')   
        plt.savefig(dir_path+f'Fold{it}.jpg')
        plt.close()

        scores_val.append(best_mape.cpu().numpy())
        scores_test.append(saved_mape_test)

        #del model and empty cache
        del(model)
        torch.cuda.empty_cache()  

    scores_val = np.array(scores_val)
    scores_test = np.array(scores_test)
    #     print(scores)
    lst_mape_val = [f'Best Val MAPE, fold {i+1} = {scores_val[i]:2.2%}' for i in range(cv)]
    lst_mape_test = [f'Best Test MAPE, fold {i+1} = {scores_test[i]:2.2%}' for i in range(cv)]

    mn_val = scores_val.mean()
    std_val = scores_val.std()

    mn_test = scores_test.mean()
    std_test = scores_test.std()

    print(*lst_mape_val, sep = '\n')
    print(f'Average Val MAPE = {mn_val:2.2%}, Std Val MAPE = {std_val:2.2%} \n')

    print(*lst_mape_test, sep = '\n')
    print(f'Average Test MAPE = {mn_test:2.2%}, Std Test MAPE = {std_test:2.2%} \n')


    f = open(dir_path+"log_loss.txt", "a")
    for ks in range(cv) :
        f.write(f'Best Val MAPE, fold {ks+1} = {scores_val[ks]:2.2%} \n')
    f.write(f'Average Val MAPE = {mn_val:2.2%}, Std Val MAPE = {std_val:2.2%} \n')

    for ks in range(cv) :
        f.write(f'Best Test MAPE, fold {ks+1} = {scores_test[ks]:2.2%} \n')
    f.write(f'Average Test MAPE = {mn_test:2.2%}, Std Test MAPE = {std_test:2.2%} \n')

    f.close()






























        
