import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, Sampler
from copy import deepcopy
from tqdm import tqdm
import sys
sys.path.append('../FakeDatasetMaking/')
from pair_nomet_creation2 import KinematicDataset, EpochSampler
import yaml
from yaml_losses import custom_loss_normal, custom_loss_no_mse


yaml_path='/home/ddemler/HNLclassifier/fnn_FeatureRegression/yamls/Aug18_1.yaml'

with open(yaml_path, 'r') as stream:
    yaml_dict=yaml.safe_load(stream)

activation_fn_mapping = {
    'relu': F.relu,
    'sigmoid': F.sigmoid,
    'tanh': F.tanh
}

loss_fn_mapping = {
    'normal': custom_loss_normal,
    'no_mse': custom_loss_no_mse
}



out_feats=['deltaphi', 'deltaeta', 'deltaR', 'mt', 'norm_mt', 'mass', 'pt', 'eta' , 'phi',  'px', 'py', 'pz', 'energy']
tryrel=[ 'mt', 'pt', 'px', 'py', 'pz', 'energy']
customlossindices=[idx for idx, feat in enumerate(out_feats) if feat in tryrel]

hidden_layers = [16,26,32,32,32,48,52,48,32,32,32,26]



train_dataset = KinematicDataset(num_events=1000000, seed=0)
input_dim, output_dim = train_dataset.usefulvariables()
# print(input_dim, output_dim)

# print(input_dim, output_dim)

train_sampler = EpochSampler(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=320, sampler=train_sampler)

val_dataset = KinematicDataset(num_events=500000, seed=10000)
input_dim, output_dim = val_dataset.usefulvariables()

val_sampler = EpochSampler(val_dataset)

val_loader = DataLoader(val_dataset, batch_size=320, sampler=val_sampler)




for model_params in yaml_dict['models']:
    name = model_params['name']
    loss_function_name = model_params['loss_function']
    loss_function=loss_fn_mapping[loss_function_name]
    patience = model_params['patience']
    hidden_layers = model_params['hidden_layers']
    activation_function = activation_fn_mapping[model_params['activation_function']]
    learning_rate = model_params['learning_rate']
    
    # print(name, loss_function, patience, hidden_layers, activation_function, learning_rate)
