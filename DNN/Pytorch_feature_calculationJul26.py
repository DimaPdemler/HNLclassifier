import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from copy import deepcopy

import os
import pickle
from pytorch2python import create_data_loaders, create_test_loader, train_model, test_model
from DD_DNN_models import DNN_bestFeature, DNN_flexible
from torchsummary import summary

"""
This module implements a Neural Architecture Search (NAS) algorithm for fully 
connected Deep Neural Networks (DNN).
It trains models using all input features defined in the 'selectionlonger' list.
The architecture of each model, defined by the depth and width of the network, 
is determined by the variable 'hidden_layer_configs'.

Main steps of the algorithm:

Load training, validation, and test datasets from specified pickle files.
The datasets are expected to be Pandas DataFrame objects.
Create PyTorch data loaders for these datasets.
These loaders are used to provide data to the model during training and testing phases.
Initialize a model with a specific configuration of hidden layers and assign it to the appropriate computing device (GPU if available, else CPU).
Define a Binary Cross-Entropy (BCE) loss function and an Adam optimizer.
Train the model for 1000 epochs, saving the model parameters that yield the best performance on the validation set.
Repeat the process for each configuration defined in 'hidden_layer_configs'.

At the end of the process, a DataFrame named 'model_info_list' is saved as a pickle file. 
This DataFrame contains the following information for each trained model: save_path (the path where the model is saved), 
save_name (the name of the saved model file), model_info (a string representation of the model's architecture), 
input_variables (the list of input variables used for the model), hidden_layers (the configuration of hidden layers in the model), 
and scaler_path (the path of the saved data scaler file used for feature normalization).
"""





# Data loading part
cdpath="/home/ddemler/dmitri_stuff/"

train = pd.read_pickle(cdpath + 'extracted_data/TEST10_v4_train_DD2')
val = pd.read_pickle(cdpath + 'extracted_data/TEST10_v4_val_DD2')
test = pd.read_pickle(cdpath + 'extracted_data/TEST10_v4_test_DD2')

selectionlonger = ['charge_1', 'charge_2', 'charge_3', 'pt_1',
       'pt_2', 'pt_3', 'pt_MET', 'eta_1', 'eta_2', 'eta_3', 'mass_1', 'mass_2',
       'mass_3', 'deltaphi_12', 'deltaphi_13', 'deltaphi_23', 'deltaphi_1MET',
       'deltaphi_2MET', 'deltaphi_3MET', 'deltaphi_1(23)', 'deltaphi_2(13)',
       'deltaphi_3(12)', 'deltaphi_MET(12)', 'deltaphi_MET(13)',
       'deltaphi_MET(23)', 'deltaphi_1(2MET)', 'deltaphi_1(3MET)',
       'deltaphi_2(1MET)', 'deltaphi_2(3MET)', 'deltaphi_3(1MET)',
       'deltaphi_3(2MET)', 'deltaeta_12', 'deltaeta_13', 'deltaeta_23',
       'deltaeta_1(23)', 'deltaeta_2(13)', 'deltaeta_3(12)', 'deltaR_12',
       'deltaR_13', 'deltaR_23', 'deltaR_1(23)', 'deltaR_2(13)',
       'deltaR_3(12)', 'pt_123', 'mt_12', 'mt_13', 'mt_23', 'mt_1MET',
       'mt_2MET', 'mt_3MET', 'mt_1(23)', 'mt_2(13)', 'mt_3(12)', 'mt_MET(12)',
       'mt_MET(13)', 'mt_MET(23)', 'mt_1(2MET)', 'mt_1(3MET)', 'mt_2(1MET)',
       'mt_2(3MET)', 'mt_3(1MET)', 'mt_3(2MET)', 'mass_12', 'mass_13',
       'mass_23', 'mass_123', 'Mt_tot', 'HNL_CM_angle_with_MET_1',
       'HNL_CM_angle_with_MET_2', 'W_CM_angle_to_plane_1',
       'W_CM_angle_to_plane_2', 'W_CM_angle_to_plane_with_MET_1',
       'W_CM_angle_to_plane_with_MET_2', 'HNL_CM_mass_1', 'HNL_CM_mass_2',
       'HNL_CM_mass_with_MET_1', 'HNL_CM_mass_with_MET_2', 'W_CM_angle_12',
       'W_CM_angle_13', 'W_CM_angle_23', 'W_CM_angle_1MET', 'W_CM_angle_2MET',
       'W_CM_angle_3MET', 'n_tauh', 'signal_label',
       'mass_hyp', 'weightNorm']

train_loader, val_loader, scaler = create_data_loaders(train, val, selectionlonger)
test_loader = create_test_loader(test, selectionlonger, scaler)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

features = deepcopy(selectionlonger)
features.remove('signal_label')
features.remove('weightNorm')

model = DNN_bestFeature(features).to(device)
criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters())

save_path = os.path.join(cdpath, 'saved_models/', 'simple_dnn', 'NAS1')
save_name = '/simple1_nelsonnorm3_depth4_1000epochs'

import itertools
from tqdm import tqdm



model_info_list=pd.DataFrame()
# hidden_layer_configs = [[10], [20], [10, 10], [20, 10], [30, 20], [40, 20], [60, 20], [30, 20, 10], [50, 40, 20], [50, 30, 10], [40,20,10], [50,50,40], [40, 10,5], [60,30,20,10], [50,40,30,20]]
hidden_layer_configs = [[20],[30],[40],[50],[10,10],[20,10],[20,10],[30,10],[40,10],[60,20],[50,20],[30,10]]

input_vars = selectionlonger

pbar = tqdm(enumerate(hidden_layer_configs), desc='layer configs', total=len(hidden_layer_configs))
for i, hidden_layer_sizes in pbar:
    desc = 'layer config: ' + str(hidden_layer_sizes)
    

    train_loader, val_loader, scaler = create_data_loaders(train, val, input_vars)
    test_loader = create_test_loader(test, input_vars, scaler)

    model = DNN_flexible(input_vars, hidden_layer_sizes).to(device)
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters())
    
    # summary(model, (len(input_vars),))

    save_path = os.path.join(cdpath, 'saved_models/', 'simple_dnn', 'NAS3')
    save_name = '/DNN_saved'+str(i+1)

    scaler_filename = f'scaler_FeatureSearch.pkl' 
    # scaler_filename = f'scaler_FeatureSearch_{i}.pkl' 
    with open(os.path.join(save_path, scaler_filename), 'wb') as f:
        pickle.dump(scaler, f)

    # Train the model
    model = train_model(train_loader, val_loader, model, optimizer, criterion, device, save_path, save_name, epochs=1000)
    # model = train_model(train_loader, val_loader, model, optimizer, criterion, device, save_path, save_name, epochs=4)

    pbar.set_description(desc)
    # Append info to the list
    new_row = pd.DataFrame({
    'save_path': [save_path], 
    'save_name': [save_name], 
    'model_info': [str(model)], 
    'input_variables': [input_vars], 
    'hidden_layers': [hidden_layer_sizes],
    'scaler_path': [os.path.join(save_path, scaler_filename)]
    })
    

    model_info_list = pd.concat([model_info_list, new_row], ignore_index=True)

model_info_list.to_pickle(save_path + '/model_info_list3.pkl')
