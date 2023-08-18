# %%
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
import sys
sys.path.append('../FakeDatasetMaking/')
from pair_nomet_creation import KinematicDataset, EpochSampler
from copy import deepcopy
from tqdm import tqdm

# %%
modelsavepath='/home/ddemler/HNLclassifier/fnn_FeatureRegression/fnn_aug17all.pt'
pdsavepath='/home/ddemler/HNLclassifier/fnn_FeatureRegression/fnn_aug17all.csv'


# %%
train_dataset = KinematicDataset(num_events=1000000, seed=0)
input_dim, output_dim = train_dataset.usefulvariables()

print(input_dim, output_dim)

train_sampler = EpochSampler(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=320, sampler=train_sampler)


# %%
val_dataset = KinematicDataset(num_events=500000, seed=10000)
input_dim, output_dim = val_dataset.usefulvariables()

val_sampler = EpochSampler(val_dataset)

val_loader = DataLoader(val_dataset, batch_size=320, sampler=val_sampler)

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class KinematicNet(nn.Module):
    def __init__(self):
        super(KinematicNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim*1.5))
        self.fc2 = nn.Linear(int(input_dim*1.5), int(input_dim//3))
        self.fc3 = nn.Linear(int(input_dim//3), output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CustomKinematicNet(nn.Module):
    def __init__(self, input_size, hidden_layers, lenoutput, activation_fn=F.relu):
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
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        return self.layers[-1](x)
    




def custom_train_loss(y_pred, y_true):
    se_loss = (y_pred - y_true) ** 2
    # print(se_loss.shape)
    num_features = int(output_dim / 3)
    y_pred_reshaped = y_pred.reshape(-1, 3, num_features)
    y_true_reshaped = y_true.reshape(-1, 3, num_features)

    
    indice=[3,6,12]
    for i in indice:
        for j in range(3):
            y_p = y_pred_reshaped[:, j, i]
            y_t = y_true_reshaped[:, j, i]
            RMSE = torch.abs(y_p - y_t)**2 / torch.abs(y_t)
            mask =  (y_t > 1)
            # print("mask shape", mask.shape)
            # print("RMSE shape", RMSE.shape)
            # print("se_loss shape", se_loss.shape)
            se_loss[mask, int(3*i+j)] = RMSE[mask]
    mse_loss = torch.mean(se_loss)

    return mse_loss

def custom_val_loss(y_pred, y_true):
    se_loss = (y_pred - y_true) ** 2
    # print(se_loss.shape)
    num_features = int(output_dim / 3)
    y_pred_reshaped = y_pred.reshape(-1, 3, num_features)
    y_true_reshaped = y_true.reshape(-1, 3, num_features)
    loss_list=[]
    indice=[3,6,12]
    for i in range(num_features):
        if i in indice:
            pairlosses=[]
            for j in range(3):
                y_p = y_pred_reshaped[:, j, i]
                y_t = y_true_reshaped[:, j, i]
                RMSE = torch.abs(y_p - y_t)**2 / torch.abs(y_t)
                mask = (y_t > 1)
                se_loss[mask, int(3*i+j)] = RMSE[mask]
                pairlosses.append(torch.mean(se_loss[:, int(3*i+j)]))
            loss_list.append(sum(pairlosses)/3)
        else:
            y_p = y_pred_reshaped[:, :, i].flatten()
            y_t = y_true_reshaped[:, :, i].flatten()
            loss = torch.mean((y_p - y_t) ** 2)
            loss_list.append(loss.item())
    mse= torch.mean(se_loss)
    return loss_list, mse
            




    
    # for i in indice:
    #     for j in range(3):
    #         y_p = y_pred_reshaped[:, j, i].flatten()
    #         y_t = y_true_reshaped[:, j, i].flatten()
    #         RMSE = torch.abs(y_p - y_t)**2 / torch.abs(y_t)
    #         mask = y_true[:] > 1
    #         se_loss[mask, int(3*i+j)] = RMSE[mask]
    # mse_loss = torch.mean(se_loss)

    # return mse_loss
    
    

# def custom_val_loss(y_pred, y_true):
#     num_features = int(output_dim / 3)
#     y_pred_reshaped = y_pred.reshape(-1, 3, num_features)
#     y_true_reshaped = y_true.reshape(-1, 3, num_features)
    
#     loss_list = []
    
#     for i in range(num_features):
#         y_p = y_pred_reshaped[:, :, i].flatten()
#         y_t = y_true_reshaped[:, :, i].flatten()
#         loss = torch.mean((y_p - y_t) ** 2)
#         loss_list.append(loss.item())

#     mse_loss = torch.mean((y_pred - y_true) ** 2)
    
#     return loss_list, mse_loss

        
        
        
def l2_regularization(model, lambda_reg):
    l2_reg = 0.0
    for W in model.parameters():
        l2_reg += torch.sum(W ** 2)
    return l2_reg * lambda_reg      
        



# %%
# hidden_layers=[64,72,82,92,102,112,122,132,142,132,122,112,102,92,82,72,64]
hidden_layers=[64,72,82,92,102,112,122,132,142,132,102,92,82]

model = CustomKinematicNet(input_size=input_dim, hidden_layers=hidden_layers, lenoutput=output_dim)
model.to(device)

# %%



optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)
# loss_fn=nn.MSELoss()


out_feats=['deltaphi', 'deltaeta', 'deltaR', 'mt', 'norm_mt', 'mass', 'pt', 'eta' , 'phi',  'px', 'py', 'pz', 'energy']
losses_cols=['train_loss', 'val_loss', 'l2sum']+out_feats
losses_df=pd.DataFrame(columns=losses_cols)


numepochs=10000
best_loss=np.inf
for epoch in range(numepochs):
    model.train()
    train_loss=0
    l2sum=0
    for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, position=0, disable=True):
        x=x.to(device)
        y=y.to(device)
        y_pred=model(x)
        original_loss = custom_train_loss(y_pred, y)
        l2_loss = l2_regularization(model, lambda_reg=1e-7)
        loss = original_loss + l2_loss
        l2sum+=l2_loss.item()
        train_loss += original_loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    model.eval()
    # patience=20
    with torch.no_grad():
        x,y=next(iter(val_loader))
        # x=val_loader
        x=x.to(device)
        y=y.to(device)
        y_pred=model(x)
        feats_loss, valloss = custom_val_loss(y_pred, y)
        # valloss=sum(feats_loss)/len(feats_loss)
        

        if valloss<best_loss:
            best_loss=valloss
            patience=30
            modelsave=deepcopy(model.state_dict())
            torch.save(modelsave, modelsavepath)
        else:
            patience-=1
            if patience==0:
                print('early stopping')
                break
    indice=[3,6,12]
    for idx in indice:
        feats_loss[idx]=feats_loss[idx].cpu().float()

    valloss=valloss.cpu().float()
    valloss2=valloss.item()
    loss_strings = [f"{out_feats[i]}: {feats_loss[i]:.4e}" for i in range(len(out_feats))]
    loss_summary = ", ".join(loss_strings)
    loss_values = [train_loss/len(train_loader), valloss2, l2sum]
    loss_values.extend(feats_loss)
    for losses in loss_values:
        print(type(losses))
    losses_df.loc[epoch] = loss_values
    losses_df.to_csv(pdsavepath)
    # losses_df.loc[epoch]=[train_loss/len(train_loader), valloss2, feats_loss[0], feats_loss[1]]
    print(f"epoch: {epoch}, train: {train_loss/len(train_loader):.4e}, val: {valloss2:.4e}, {loss_summary}")



