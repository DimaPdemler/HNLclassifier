# %%
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import os
from copy import deepcopy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from itertools import permutations
import torch.nn.functional as F
# from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import vector
from tqdm import tqdm
import yaml


# %%
#load data
base_path = os.path.dirname(os.getcwd())

yaml_name='Aug13run1.yaml'
yaml_file_path= os.path.join(base_path, "fnn_FeatureRegression", 'yamls', yaml_name)

full_folder_path = os.path.join(base_path,"saved_files", "fake_data")
# data_df=pd.read_pickle(os.path.join(full_folder_path,"Aug7_1mil.pkl"))
with open(os.path.join(full_folder_path,"Aug10_5mil.pkl"), 'rb') as f:
# with open(os.path.join(full_folder_path,"Aug7_1mil.pkl"), 'rb') as f:
    clean_data_dict = pickle.load(f)
# print(clean_data_dict.keys())
numevents=len(clean_data_dict['2_phi'])
# print("number of events:",numevents)

data_dict=clean_data_dict
# print(data_dict.keys())

data_dict_np={}
for key in data_dict.keys():
    data_dict_np[key]=np.array(data_dict[key])


# output features: deltaphi, deltaeta, deltaR, m_t,  norm_mt, mass, pt, eta , phi*(maybe dont use),  px, py, pz, energy


# %%
input_data_names_ordered = [
    ['MET_phi', 'pt_MET'], 
    ['1_phi', 'charge_1', 'pt_1', 'eta_1', 'mass_1'], 
    ['2_phi', 'charge_2', 'pt_2', 'eta_2', 'mass_2'], 
    ['3_phi', 'charge_3', 'pt_3', 'eta_3', 'mass_3']
]
input_data_particle_order = ['MET', '1', '2', '3']

pair_order = ["MET_1", "MET_2", "MET_3", "1_2", "1_3", "2_3"]
used_labels2 = [
    ['deltaphi_1MET', 'mt_1MET'], 
    ['deltaphi_2MET', 'mt_2MET'], 
    ['deltaphi_3MET', 'mt_3MET'], 
    ['deltaphi_12', 'deltaeta_12', 'deltaR_12', 'mt_12', 'norm_mt_12'], 
    ['deltaphi_13', 'deltaeta_13', 'deltaR_13', 'mt_13', 'norm_mt_13'], 
    ['deltaphi_23', 'deltaeta_23', 'deltaR_23', 'mt_23', 'norm_mt_23']
]

lepton_input_ordered = input_data_names_ordered[1:]
lepton_output_ordered = used_labels2[3:]

l_input_shape=(numevents,len(lepton_input_ordered), len(lepton_input_ordered[0]))
# print("events, particles, input features: ",l_input_shape)
l_input= np.empty(l_input_shape)

for i in range(len(lepton_input_ordered)):
    for j, feature in enumerate(lepton_input_ordered[i]):
        l_input[:,i,j] = data_dict_np[feature]

l_output_shape=(numevents, len(lepton_output_ordered), len(lepton_output_ordered[0]))
# print("events, particle pairs, output kin. features: ",l_output_shape)
l_output= np.empty(l_output_shape)

for i in range(len(lepton_output_ordered)):
    for j, feature in enumerate(lepton_output_ordered[i]):
        l_output[:,i,j] = data_dict_np[feature]

lepton_pair_order = pair_order[3:]
lepton_particle_order = input_data_particle_order[1:]
# print("lepton pair order: ", lepton_pair_order)
# print("lepton particle order: ", lepton_particle_order)

# %%
def add_extra_features(data):
    p1_pt=data['pt_1']
    p2_pt=data['pt_2']
    p3_pt=data['pt_3']

    p1_phi=data["1_phi"]
    p2_phi=data["2_phi"]
    p3_phi=data["3_phi"]

    p1_eta=data["eta_1"]
    p2_eta=data["eta_2"]
    p3_eta=data["eta_3"]

    p1_mass=data["mass_1"]
    p2_mass=data["mass_2"]
    p3_mass=data["mass_3"]

    particle1=vector.arr({"pt": p1_pt, "phi": p1_phi, "eta": p1_eta, "mass": p1_mass})
    particle2=vector.arr({"pt": p2_pt, "phi": p2_phi, "eta": p2_eta, "mass": p2_mass})
    particle3=vector.arr({"pt": p3_pt, "phi": p3_phi, "eta": p3_eta, "mass": p3_mass})

    p4_mother12=particle1+particle2
    p4_mother23=particle2+particle3
    p4_mother13=particle1+particle3

    pairs=['12','13','23']
    motherpairs=[p4_mother12, p4_mother13, p4_mother23]
    features_toadd=[ 'mass', 'pt', 'eta' , 'phi',  'px', 'py', 'pz', 'energy']
    # features_toadd=[ 'mass', 'pt', 'eta' , 'phi',  'px', 'py', 'pz']

    add_feat_size=(len(data['pt_1']), len(pairs), len(features_toadd))
    add_feat_array= np.empty(add_feat_size)

    for feature in features_toadd:
        for i, pair in enumerate(pairs):
           add_feat_array[:, i, features_toadd.index(feature)] = getattr(motherpairs[i], feature)
    return add_feat_array

    
    # for i, pair in enumerate(pairs):
    #     features_toadd=[ 'mass', 'pt', 'eta' , 'phi',  'px', 'py', 'pz', 'energy']
    #     for feature in features_toadd:
    #         data['mother_' + feature + '_' + pair] = motherpairs[i].feature
    # return data

data_conc=add_extra_features(data_dict_np)
# print(data_conc.shape)
# print(data_new.columns)
l_output2= np.concatenate((l_output, data_conc), axis=2)
# print("l_output new shape: ",l_output2.shape)


# %% [markdown]
# ## normalizing ##

# %%
def normalize_l(data):
    # l_input_reshaped=l_input.reshape(l_input.shape[0], -1)
    # print("input reshaped shape:",l_input_reshaped.shape)
    # scaler = StandardScaler()
    # data_normalized = scaler.fit_transform(l_input_reshaped)

    means = data.mean(axis=(0))
    stds = data.std(axis=(0))
    # print("means shape:",means.shape)

    data_normalized = (data - means) / (stds + 1e-10)
    return data_normalized, means, stds

n_l_input, _, _ = normalize_l(l_input)
# print("normalized input shape:",n_l_input.shape)


# n_l_output, l_output_means, l_output_stds = normalize_l(l_output2)
# print("normalized output shape:",n_l_output.shape)
n_l_output = l_output2

def invert_normalize(data_normalized, means, stds):
    return (data_normalized * stds) + means




# %% [markdown]
# ## Converting to tensor and adding pairs ##

# %%
linput_tensor = torch.tensor(n_l_input, dtype=torch.float32)
llabel_tensor = torch.tensor(n_l_output, dtype=torch.float32)

lpairs_data=[]
lpairs_labels=[]

# lepton_pair_order = ['1_2', '1_3', '2_3']
lepton_pair_mapping={(0,1): lepton_pair_order.index('1_2'), (0,2): lepton_pair_order.index('1_3'), (1,2): lepton_pair_order.index('2_3')}

for key, value in lepton_pair_mapping.items():
    concatonated_data=torch.cat((linput_tensor[:,key[0],:], linput_tensor[:,key[1],:]), dim=1)
    lpairs_data.append(concatonated_data)

    lpairs_labels.append(llabel_tensor[:,value,:])

# print(len(lpairs_data), len(lpairs_labels))
# print(lpairs_data[0].shape, lpairs_labels[0].shape)

# %% [markdown]
# ## Making training data ##

# %%
train_data_list = []
val_data_list = []
test_data_list = []
train_labels_list = []
val_labels_list = []
test_labels_list = []

for pair_idx in range(len(lpairs_data)):
    pair_data = lpairs_data[pair_idx]
    pair_labels = lpairs_labels[pair_idx]

    train_val_data, test_data, train_val_labels, test_labels = train_test_split(pair_data, pair_labels, test_size=0.2, random_state=42)
    train_data, val_data, train_labels, val_labels = train_test_split(train_val_data, train_val_labels, test_size=0.2, random_state=42)

    train_data_list.append(train_data)
    val_data_list.append(val_data)
    test_data_list.append(test_data)
    train_labels_list.append(train_labels)
    val_labels_list.append(val_labels)
    test_labels_list.append(test_labels)

class ParticleDataset(Dataset):
    def __init__(self, data_list, labels_list):
        self.data_list = data_list
        self.labels_list = labels_list
    
    def __len__(self):
        return len(self.data_list[0])
    
    def __getitem__(self, idx):
        return [data[idx] for data in self.data_list], [label[idx] for label in self.labels_list]


# %% [markdown]
# ## Defining Model ##

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class CustomKinematicNet(nn.Module):
    def __init__(self, input_size, hidden_layers, lenoutput, activation_fn=F.relu, dropout_prob=0.2):
        """
        Args:
        - input_size (int): Size of the input layer.
        - hidden_layers (list of int): Sizes of each hidden layer.
        - lenoutput (int): Size of the output layer.
        - activation_fn (callable): Activation function to use.
        """
        super(CustomKinematicNet, self).__init__()
        
        # Create the list of layers
        layers = [nn.Linear(input_size, hidden_layers[0])]
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        layers.append(nn.Linear(hidden_layers[-1], lenoutput))
        
        self.layers = nn.ModuleList(layers)
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
            x = self.dropout(x)
        return self.layers[-1](x)

# hidden_layer_sizes = [128, 200, 300, 250, 128, 64]
lenoutput = l_output2.shape[2]
# model = CustomKinematicNet(input_size=10, hidden_layers=hidden_layer_sizes, lenoutput=lenoutput, activation_fn=F.relu)
# model.to(device)




# %% [markdown]
# ## Training ##

# %%
def custom_loss(y_pred, y_true):
    # Compute MSE loss for each output individually
    mse_loss = F.mse_loss(y_pred, y_true, reduction='none')
    
    # Compute RMSE for specific indices and replace in the MSE loss
    indices = [3, 6, 12]
    for idx in indices:
        RMSE = torch.abs(y_pred[:, idx]**2 - y_true[:, idx]**2) / torch.abs(y_true[:, idx])
        mask = y_true[:, idx] > 1
        mse_loss[mask, idx] = RMSE[mask]
    
    # Calculate average loss of each output
    avg_loss_per_output = mse_loss.mean(dim=0)  # Averaging over the batch dimension
    
    # Normalize the average loss by its maximum loss value
    normalized_loss = avg_loss_per_output / avg_loss_per_output.max()
    
    # Calculate the total loss as the average of the normalized losses
    total_loss = normalized_loss.mean()
    
    return total_loss

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_train_loss = 0
    
    for data, labels in data_loader:
        data = [d.to(device) for d in data]
        labels = [l.to(device) for l in labels]
        
        optimizer.zero_grad()
        total_loss = 0
        for i in range(len(data)):
            y_pred = model(data[i])
            # loss = loss_fn(y_pred, labels[i])
            loss = custom_loss(y_pred, labels[i])
            total_loss += loss
        total_train_loss += total_loss.item()
        
        total_loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(data_loader)
    return avg_train_loss

# def train_one_epoch(model, data_loader, optimizer, device):
#     model.train()
#     total_train_loss=0
    
#     for data, labels in data_loader:
#         data = [d.to(device) for d in data]
#         labels = [l.to(device) for l in labels]
        
#         optimizer.zero_grad()
#         total_loss = 0
#         for i in range(len(data)):
#             y_pred = model(data[i])
#             total_loss = custom_loss(y_pred, labels[i])
            
#             total_batch_loss += curr_loss

        
#         total_batch_loss.backward()
#         optimizer.step()


# def validate_one_epoch(model, data_loader, device):
#     model.eval()
#     total_mse_loss = 0
#     total_rr_loss = 0
    
#     with torch.no_grad():
#         for data, labels in data_loader:
#             data = [d.to(device) for d in data]
#             labels = [l.to(device) for l in labels]
            
#             total_loss = 0
#             for i in range(len(data)):
#                 y_pred = model(data[i])
#                 mse, rr = custom_loss(y_pred, labels[i])
                
#                 total_mse_loss += mse.item()
#                 total_rr_loss += rr.item()

#     avg_mse_loss = total_mse_loss / len(data_loader)
#     avg_rr_loss = total_rr_loss / len(data_loader)
#     return avg_mse_loss, avg_rr_loss

def validate_model(model, data_loader, device):
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for data, labels in data_loader:
            data = [d.to(device) for d in data]
            labels = [l.to(device) for l in labels]

            total_loss = 0
            for i in range(len(data)):
                y_pred = model(data[i])
                # loss = loss_fn(y_pred, labels[i])
                loss = custom_loss(y_pred, labels[i])                
                total_loss += loss
            total_val_loss += total_loss.item()

    avg_val_loss = total_val_loss / len(data_loader)
    return avg_val_loss

# def test_model2(model, data_loader, device):
#     model.eval()
#     total_mse_loss = 0
#     total_rr_loss = 0

#     with torch.no_grad():
#         for data, labels in data_loader:
#             data = [d.to(device) for d in data]
#             labels = [l.to(device) for l in labels]

#             total_loss = 0
#             for i in range(len(data)):
#                 y_pred = model(data[i])
#                 mse, rr = custom_loss(y_pred, labels[i])

#                 total_mse_loss += mse.item()
#                 total_rr_loss += rr.item()
#     avg_mse_loss = total_mse_loss / len(data_loader)
#     avg_rr_loss = total_rr_loss / len(data_loader)
#     return avg_mse_loss, avg_rr_loss


def test_model(model, data_loader, device):
    model.eval()
    total_test_loss = 0

    with torch.no_grad():
        for data, labels in data_loader:
            data = [d.to(device) for d in data]
            labels = [l.to(device) for l in labels]

            total_loss = 0
            for i in range(len(data)):
                y_pred = model(data[i])
                # loss = loss_fn(y_pred, labels[i])
                loss = custom_loss(y_pred, labels[i])
                total_loss += loss
            total_test_loss += total_loss.item()

    avg_test_loss = total_test_loss / len(data_loader)
    return avg_test_loss

def main_training_loop(model, num_epochs, train_data_list, train_labels_list, val_data_list, val_labels_list, optimizer, loss_fn, device, early_stop_patience, batch_size=320):
    epochs_no_improve = 0
    min_val_loss = np.Inf


    train_dataset = ParticleDataset(train_data_list, train_labels_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = ParticleDataset(val_data_list, val_labels_list)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = ParticleDataset(test_data_list, test_labels_list)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        test_loss = train_one_epoch(model, train_loader, optimizer, device)
        # model, data_loader, optimizer, device
        # train_loss = train_one_epoch(model, train_data_list, train_labels_list, optimizer, loss_fn, device)
        val_loss = validate_model(model, val_loader, device)
        # val_loss = validate_model(model, val_loader, loss_fn, device)
        # val_loss = validate_model(model, val_data_list, val_labels_list, loss_fn, device)
        
        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            saved_model = model.state_dict()
            # torch.save(model.state_dict(), 'fnn_FeatureRegression/fnn_try4.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stop_patience:
                print('Early stopping!')
                return saved_model
        if (epoch +1) % 5 == 0:
            # test_loss = test_model(model, test_data_list, test_labels_list, loss_fn, device)
            # test_loss = test_model(model, test_loader, loss_fn, device)
            test_loss = test_model(model, test_loader, device)
            print(f"Epoch [{epoch + 1}], "
            f"Train Loss: ({test_loss*1000:.4f}), "
                f"Val Loss: ({val_loss*1000:.4f}), "
                f"Test Loss: ({test_loss*1000:.4f})")
        
        else:
            print(f"Epoch [{epoch + 1}], "
                f"Train Loss: ({test_loss*1000:.4f}), "
                f"Val Loss: ({val_loss*1000:.4f})")

# %%
batch_size=1000
num_epochs = 10000
early_stop_patience = 20
loss_fn = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hidden_layers_list= [[128, 200, 300, 250, 128, 64], 
#                      [32,42,52,64,74,52,42,32,16], 
#                      [32,42,52,64,42,32], 
#                      [32,42,52,64,42,16], 
#                      [32,32,32,32,32,32,32,32,32,32], 
#                      [32,32,32,32,32,32], 
#                      [32,32,32]]
# hidden_layers_list= [[128, 200, 300, 250, 128, 64]]
# hidden_layers_list= [[32,42,52,64,74,52,42,32,32]]
# hidden_layers_list= [[32,42,52,64,74,82,74,64,64,50]]
# hidden_layers_list= [[32,32,32,32,32,32,32,32,32,32,32,32,32],
#                      [32,32,32,32,32,32,32,32,32,32,32,32,32],
#                      [32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32],
#                      [32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32],
#                      [32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32]]

# activation_fn_list=[F.relu, F.sigmoid, F.relu, F.sigmoid, F.tanh]

with open(yaml_file_path, 'r') as yaml_file:
    loaded_data = yaml.load(yaml_file, Loader=yaml.SafeLoader)

# Accessing data from the loaded dictionary
hidden_layers_list_loaded = loaded_data['hidden_layers_list']
activation_fn_names_loaded = loaded_data['activation_fn_list']

# Map function names back to their actual functions
activation_fn_mapping = {
    'relu': F.relu,
    'sigmoid': F.sigmoid,
    'tanh': F.tanh
}

# Convert function names to actual functions
activation_fn_list_loaded = [activation_fn_mapping[name] for name in activation_fn_names_loaded]

hidden_layers_list = hidden_layers_list_loaded
activation_fn_list = activation_fn_list_loaded

pbar = tqdm(hidden_layers_list, total=len(hidden_layers_list))

for i, hidden_layers_curr in enumerate(pbar):
    pbar.set_description(f"model [{i+1}/{len(hidden_layers_list)}]")
    # print(f"[{i+1}/{len(hidden_layers_list)}] Hidden Layers: {hidden_layers_curr}")
    curr_model= CustomKinematicNet(input_size=10, hidden_layers=hidden_layers_curr, lenoutput=lenoutput, activation_fn=activation_fn_list[i])
    curr_model.to(device)
    l2_reg_strength = 1e-4
    optimizer = torch.optim.Adam(curr_model.parameters(), lr=0.001, weight_decay=l2_reg_strength)
    curr_saved_model = main_training_loop(curr_model, num_epochs, train_data_list, train_labels_list, val_data_list, val_labels_list, optimizer, loss_fn, device, early_stop_patience, batch_size=batch_size)
    save_path = os.path.join(base_path, "saved_files","saved_models", "FNN_FeatureRegression",f"fnn_try_aug15_{i+1}.pt")
    torch.save(curr_saved_model, save_path)




