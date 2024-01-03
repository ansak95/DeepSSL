# Essential imports for PyTorch and data manipulation
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch.nn.modules import ModuleList, normalization

# Additional utilities
import argparse
import datetime 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import time
import tqdm
from utils import utils_ft # Import custom utilities for feature transformation

# Disable warnings (consider reviewing this for better debugging)
import warnings
warnings.filterwarnings("ignore")

# Define a function to parse training arguments
def training_args():
    """
    Parses command line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments with training configurations.
    """
    parser = argparse.ArgumentParser(description='fine_tune')
    # Define arguments
    parser.add_argument('--path', default='', type=str, help='model path')
    parser.add_argument('--folder', default='', type=str, help='folder path')
    parser.add_argument('--reset', default='no', type=str, help='Reset weights?')
    parser.add_argument('--freeze', default='freeze', type=str, help='Freeze weights')
    parser.add_argument('--finetune', default=False, type=bool, help='Finetune')
    parser.add_argument('--cv', default=5, type=int, help='k fold')
    parser.add_argument('--num_gpus', default=1, type=int, help='number of GPUs')
    parser.add_argument('--nb_samples', default=10, type=int, help='Number of samples')
    parser.add_argument('-b', '--batch_size', default=4096, type=int, help='mini-batch size')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='number of total epochs')
    parser.add_argument('--device', default=0, type=int, help='which device')
    parser.add_argument('--maxlen', default=30, type=int, help='Windows length')
    parser.add_argument('--nb_gauges', default=3, type=int, help='Number of gauges')
    parser.add_argument('--thinning', default=500, type=int, help='Thinning')
    parser.add_argument('--lr', default=1e-6, type=float, help='Learning rate')
    parser.add_argument('--drop', default=0.1, type=float, help='Dropout rate')

    # Parse and return arguments
    return parser.parse_args()

# Define function to create data loaders
def create_loaders(data, bs=512, jobs=0):
    """
    Creates a data loader for the given dataset.

    Args:
        data (Dataset): The dataset for which to create the data loader.
        bs (int): Batch size. Default is 512.
        jobs (int): Number of worker processes to use. Default is 0.

    Returns:
        DataLoader: Data loader for the given dataset.
    """
    return DataLoader(data, batch_size=bs, shuffle=True, num_workers=jobs, pin_memory=False)

# Define a custom GRU Layer class
class GRU_Layer(nn.Module):
    """
    Custom GRU Layer class.
    """
    def __init__(self, input_dim, hidden_dim, n_layers, drop_prob):
        super(GRU_Layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)

    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)    
        return F.silu(out), hidden

# Define a custom sequential class
class MySequential(nn.Sequential):
    """
    Custom Sequential class to handle multiple input formats.
    """
    def forward(self, *inputs):
        for module in self._modules.values():
            inputs = module(*inputs) if type(inputs) == tuple else module(inputs)
        return inputs

# Define a function to update the learning rate
def update_lr(optimizer, lr):
    """
    Updates the learning rate for an optimizer.

    Args:
        optimizer (Optimizer): The optimizer to update.
        lr (float): The new learning rate.
    """
    for g in optimizer.param_groups:
        g['lr'] = lr


def configure_model(model, args, trn_std, trn_mean, bias, lr, device):
    """
    Configures the model based on the given arguments.

    Args:
        model (torch.nn.Module): The model to configure.
        args (argparse.Namespace): Command-line arguments with 'reset' and 'freeze' options.
        trn_std (torch.Tensor): Standard deviation for normalization.
        trn_mean (torch.Tensor): Mean for normalization.
        bias (float): Bias value to add to the last layer.
        lr (float): Learning rate for the optimizer.
        device (torch.device): The device to use for tensors.

    Returns:
        torch.optim.Optimizer: Configured optimizer for the model.
    """
    def reset_weights(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    def set_fc_layer(num_features):
        fc_layer = mySequential(
            GRU_Layer(num_features, num_features, 1, 0.1),
            nn.LayerNorm(num_features, elementwise_affine=False),
            nn.Linear(num_features, 1)
        ).to(device)
        nn.init.xavier_normal_(fc_layer[-1].weight.data)
        fc_layer[-1].bias.data += bias
        return fc_layer

    if args.reset == 'reset':
        print('Reset weights...')
        model.apply(reset_weights)
        model.std = trn_std.to(device)
        model.mean = trn_mean.to(device)

    elif args.freeze == 'freeze':
        print('Freeze pre-trained layers')
        utils_ft.set_parameter_requires_grad(model, True)
    else:
        print('Unfreeze pre-trained layers')
        utils_ft.set_parameter_requires_grad(model, False)

    # Set up the fully connected layer
    num_features = model.fc.in_features
    model.fc = set_fc_layer(num_features)
    model.length_seq = 30
    model.ft = True

    # Return the configured optimizer
    return model, optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

def prepare_data_for_fold(data_train_raw, i, cv, nb_samples, sequence_length, seq_cols_in, seq_cols_out, bs, device):
    """
    Prepares training and validation data for a given fold in cross-validation.

    Args:
        data_train_raw (DataFrame): Raw training data.
        i (int): Index of the current fold in cross-validation.
        cv (int): Total number of folds in cross-validation.
        nb_samples (int): Number of samples in the dataset.
        sequence_length (int): Length of the sequence for training.
        seq_cols_in (list): List of column names for input features.
        seq_cols_out (list): List of column names for output labels.
        bs (int): Batch size for data loaders.
        device (torch.device): Device to use for tensors.

    Returns:
        DataLoader: DataLoader for training data.
        DataLoader: DataLoader for validation data.
    """
    print('----------------------------------')
    print(f'Fold {i + 1}')
    print('----------------------------------')

    # Create indices for training and validation data
    list_train = np.arange(1, nb_samples + 1)
    list_val = list_train[i:i + nb_samples // cv]
    list_train = np.delete(list_train, np.arange(i, i + nb_samples // cv))

    # Split data into training and validation sets
    data_train = data_train_raw[data_train_raw.ID.isin(list_train)]
    data_val = data_train_raw[data_train_raw.ID.isin(list_val)]

    # Preprocess the data
    X_train, y_train = utils_ft.seq_preprocess(data_train, sequence_length, seq_cols_in, seq_cols_out, type_set='Train')
    X_val, y_val = utils_ft.seq_preprocess(data_val, sequence_length, seq_cols_in, seq_cols_out, type_set='Val')

    # Create TensorDatasets
    train_dl = TensorDataset(X_train, torch.log(y_train + 500).to(device))
    val_dl = TensorDataset(X_val, torch.log(y_val + 500).to(device))

    # Create DataLoaders
    print(f'Creating data loaders with batch size: {bs}')
    trn_dl = create_loaders(train_dl, bs, jobs=4)
    val_dl = create_loaders(val_dl, bs, jobs=4)

    return trn_dl, val_dl

if __name__ == "__main__":
    # Parse arguments for training
    args = training_args()
    print(args)

    # Set up various training configurations
    nb_gauges = args.nb_gauges
    device = torch.device('cuda') # Consider checking for CUDA availability
    bs = args.batch_size
    epochs = args.epochs
    maxlen = args.maxlen
    nb_samples = args.nb_samples
    thinning = args.thinning
    cv = args.cv
    patience = 500  # Initial patience for early stopping

    # Set manual seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Change directory to data folder and load data
    os.chdir("/home/anassakrim/FolderThesis/ProjectSSL/")
    fd_data = 'data/'
    data_train_raw = pd.read_pickle(fd_data + 'data_train_ft').reset_index().iloc[:, 1:]
    data_test = pd.read_pickle(fd_data + 'data_test').reset_index().iloc[:, 1:]  
    data_train = data_train_raw[data_train_raw.ID <= nb_samples]

    # Data preparation
    seq_cols_in = ['gauge' + str(i + 1) for i in range(nb_gauges)]
    seq_cols_out = ['RUL']
    sequence_length = maxlen

    # Preprocess training data
    tmp = data_train[seq_cols_in].values
    trn_mean = tmp.mean(axis=0).reshape(1, -1)
    trn_std = tmp.std(axis=0).reshape(1, -1)
    trn_mean = torch.tensor(trn_mean).float()
    trn_std = torch.tensor(trn_std).float()
    bias = np.log(data_train[seq_cols_out].values + 500).mean()  # Bias for fine-tuning

    # Preprocess test set
    X_test, y_test = utils_ft.seq_preprocess(data_test, sequence_length, seq_cols_in, seq_cols_out, type_set='Test')
    y_test = y_test.to(device).reshape(-1)
    X_test = X_test.to(device)

    # Prepare the model directory
    fd = args.folder
    dt = f"{datetime.datetime.now():%Y%h%d_%Hh%M}"
    path_model = fd + "/L2_PT_Test_k_fold_" + dt + '_' + f"Finetune{nb_samples}"
    os.makedirs(path_model, exist_ok=True)
    dir_path = path_model + "/"

    # Load pre-trained model
    model_path = args.path + 'model.pth'
    it = 0
    lr = args.lr
    model = utils_ft.load_checkpoint(model_path, train=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # Save model architecture and training configuration
    with open(dir_path + "model_parameters.txt", "a") as f:
        f.write(str(model.state_dict()))

    with open(dir_path + "log_loss.txt", "a") as f:
        pass  # Currently empty, consider logging training progress here

    with open(dir_path + "args.txt", "w+") as f:
        f.write(str(args))

    with open(dir_path + "optim.txt", "w+") as f:
        f.write(str(optimizer))

    # Main training and evaluation loop
    t0 = time.time()

    scores_val = []
    scores_test = []

    for i in range(0, nb_samples, nb_samples // cv):
        patience = 500
        best_mape = float('inf')
        saved_mape_test = float('inf')

        model = utils_ft.load_checkpoint(model_path, train=True)
        model, optimizer = configure_model(model, args, trn_std, trn_mean, bias, lr, device)
        model.to(device)
        criterion = nn.MSELoss().to(device)


        # Prepare DataLoader for both training and validation datasets
        trn_dl, val_dl = prepare_data_for_fold(data_train_raw, it, cv, nb_samples, 
                                               sequence_length, seq_cols_in, seq_cols_out, bs, device)
        trn_mape_track = []
        val_mape_track = []

        # Save model information
        infos_model = (
            f'Number of training aircraft components: {len(np.unique(data_train.ID))} '
            f'\\ Sequence length: {maxlen} '
            f'\\ Number of training samples: {X_train.shape[0]} '
            f'\\ Number of epochs: {args.epochs} '
            f'\\ Optimizer learning rate: {args.lr} '
            f'\\ Running time in minutes: {(time.time() - t0) / 60} '
            f'\\ Number of model parameters: {model.number_of_parameters()}'
        )
        with open(os.path.join(dir_path, "training_readme.txt"), "w+") as f:
            f.write(infos_model)
        
        # Move training and validation data to the specified device
        y_train = y_train.to(device).reshape(-1)
        y_val = y_val.to(device).reshape(-1)
        X_train = X_train.to(device)
        X_val = X_val.to(device)
        
        # Iterate over a range of learning rates
        it_lr = 0
        for l_r in [args.lr / (10 ** p) for p in range(3)]:
            patience = 500  # Reset patience variable for early stopping
            if it_lr != 0:  # For second fold and beyond
                print("Load the model...")
                with open(os.path.join(dir_path, "log_loss.txt"), "a") as f:
                    f.write("Load the model...\n")
        
                model = utils_ft.load_checkpoint(os.path.join(dir_path, PATH), train=True)
                # Freeze or unfreeze model layers based on arguments
                if args.freeze == 'freeze' and args.reset == 'no':
                    utils_ft.set_parameter_requires_grad(model, True)  # Freeze all weights
                    for param in model.fc.parameters():
                        param.requires_grad = True  # Unfreeze weights of the final layer
        
                model.to(device)
                checkpoint = torch.load(os.path.join(dir_path, PATH))
                optimizer = optim.Adam(model.parameters(), lr=l_r, weight_decay=1e-6)
                optimizer.load_state_dict(checkpoint['optimizer_dic'])
                update_lr(optimizer, l_r)
        
            it_lr += 1
        
            print(f'Learning rate adjusted to {optimizer.param_groups[0]["lr"]:.7f}')
            with open(os.path.join(dir_path, "log_loss.txt"), "a") as f:
                f.write(f"Begin training.\nLearning rate adjusted to {optimizer.param_groups[0]['lr']:.7f}\n")
        
            # Initialize progress bar for training epochs
            pbar = trange(args.epochs, unit="epoch")
            for epoch in range(args.epochs):
                # Training step
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

               # Model Evaluation
                model.eval()
                with torch.no_grad():
                    # Evaluate on training data
                    pred_train = model(X_train)[0][:, -1, :].reshape(-1)
                    pred_train = torch.exp(pred_train) - 500  # Inverse transform
                    train_mape = torch.mean(torch.abs((pred_train - y_train) / y_train).masked_fill(torch.isinf(pred_train - y_train), 0))
                
                    # Evaluate on validation data
                    pred_val = model(X_val)[0][:, -1, :].reshape(-1)
                    pred_val = torch.exp(pred_val) - 500
                    val_mape = torch.mean(torch.abs((pred_val - y_val) / y_val).masked_fill(torch.isinf(pred_val - y_val), 0))
                
                    # Evaluate on test data
                    pred_test = model(X_test)[0][:, -1, :].reshape(-1)
                    pred_test = torch.exp(pred_test) - 500
                    test_mape = torch.mean(torch.abs((pred_test - y_test) / y_test))
                
                # Update progress bar and log results
                pbar.set_description(f'Epoch {epoch + 1}/{args.epochs}')
                pbar.set_postfix_str(f'Train MAPE {train_mape:.2%}, Val MAPE {val_mape:.2%}, Test MAPE {test_mape:.2%}, Best MAPE {best_mape:.2%}, Saved Test MAPE {saved_mape_test:.2%}, Patience {patience}')
                with open(os.path.join(dir_path, "log_loss.txt"), "a") as f:
                    f.write(f'Epoch {epoch + 1}/{args.epochs}, Train MAPE {train_mape:.2%}, Val MAPE {val_mape:.2%}, Test MAPE {test_mape:.2%}, Best MAPE {best_mape:.2%}, Saved Test MAPE {saved_mape_test:.2%}, Patience {patience}\n')
                
                # Checkpointing and Early Stopping
                if epoch % 100 == 0:
                    torch.cuda.empty_cache()
                if val_mape < best_mape:
                    best_mape = val_mape
                    saved_mape_test = test_mape
                    patience = 500
                    with open(os.path.join(dir_path, "log_loss.txt"), "a") as f:
                        f.write("Save the model...\n")
                    checkpoint = {'model': model, 'mape': trn_mape_track, 'val_mape': val_mape_track, 'state_dict': model.state_dict(), 'optimizer_dic': optimizer.state_dict(), 'lr': lr}
                    torch.save(checkpoint, os.path.join(dir_path, PATH))
                else:
                    patience -= 1
                if patience == 0:
                    break
                
                # Plotting
                plt.figure(figsize=(22, 12))
                plt.plot(trn_mape_track, label='Training set')
                plt.plot(val_mape_track, label=f'Val set with best MAPE = {best_mape:.2%}')
                plt.grid()
                plt.legend()
                plt.title(f'{nb_samples} training structures, Fold {it}, Test set MAPE {saved_mape_test:.2%}')
                plt.xlabel('Epoch')
                plt.ylabel('MAPE (%)')
                plt.savefig(os.path.join(dir_path, f'Fold{it}.jpg'))
                plt.close()
                
                # Update scores
                scores_val.append(best_mape.cpu().numpy())
                scores_test.append(saved_mape_test)
                
                # Clean up
                del model
                torch.cuda.empty_cache()


    scores_val = np.array(scores_val)
    scores_test = np.array(scores_test)
    mn_val, std_val = scores_val.mean(), scores_val.std()
    mn_test, std_test = scores_test.mean(), scores_test.std()

    with open(dir_path + "log_loss.txt", "a") as f:
        # Logic for logging fold-wise and overall performance

    print("Training completed.")

    
