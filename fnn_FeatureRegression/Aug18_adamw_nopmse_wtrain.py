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
from copy import deepcopy
from tqdm import tqdm
import sys
sys.path.append('../FakeDatasetMaking/')
from pair_nomet_creation2 import KinematicDataset, EpochSampler


# %%
modelsavepath='/home/ddemler/HNLclassifier/saved_files/saved_models/FNN_FeatureRegression/fnn_aug18_adamw_nomse.pt'
pdsavepath='/home/ddemler/HNLclassifier/saved_files/fnn_featregr/fnn_aug18_adamw_sm.csv'
pd_train_savepath='/home/ddemler/HNLclassifier/saved_files/fnn_featregr/fnn_aug18_nomse2_train.csv'

out_feats=['deltaphi', 'deltaeta', 'deltaR', 'mt', 'norm_mt', 'mass', 'pt', 'eta' , 'phi',  'px', 'py', 'pz', 'energy']
tryrel=[ 'mt','mass', 'pt', 'px', 'py', 'pz', 'energy']
customlossindices=[idx for idx, feat in enumerate(out_feats) if feat in tryrel]

hidden_layers = [32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32]



# %%
train_dataset = KinematicDataset(num_events=1000000, seed=0)
input_dim, output_dim = train_dataset.usefulvariables()
# print(input_dim, output_dim)

# print(input_dim, output_dim)

train_sampler = EpochSampler(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=320, sampler=train_sampler)


# %%
val_dataset = KinematicDataset(num_events=500000, seed=100000)
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
    




# def custom_loss(y_pred, y_true):
#     se_loss = (y_pred - y_true) ** 2
#     MSE_loss=se_loss.clone()
#     # print(se_loss.shape)
#     num_features = int(output_dim)

#     loss_list=[]

#     for i in range(num_features):

#         if i in customlossindices:
#             RMSE=((y_pred[:,i]-y_true[:,i])/y_true[:,i])**2
#             mask = (y_true[:,i] > 1)

#             RMSE_meanloss=torch.mean(RMSE[mask])
#             MSE_meanloss = torch.mean(se_loss[:,i][~mask])

#             MSE_loss[:, i] = 0  
#             MSE_loss[mask, i] = RMSE[mask]

#             loss_list.append(MSE_meanloss.item())
#             loss_list.append(RMSE_meanloss.item())
#         else:
#             loss = torch.mean(se_loss[:,i])
#             loss_list.append(loss.item())
    
#     full_loss = torch.mean(se_loss)
#     return loss_list, full_loss


def custom_loss(y_pred, y_true):
    se_loss = (y_pred - y_true) ** 2
    MSE_loss = torch.zeros_like(se_loss)  # Initialize with zeros
    
    num_features = int(output_dim)
    loss_list = []

    for i in range(num_features):
        if i in customlossindices:
            RMSE = ((y_pred[:, i] - y_true[:, i]) / y_true[:, i]) ** 2
            mask = (y_true[:, i] > 1)
            
            RMSE_meanloss = torch.mean(RMSE[mask])
            MSE_meanloss = torch.mean(se_loss[:, i][~mask])

            # Weighted contribution of the RMSE for the masked values to the final loss tensor
            MSE_loss[:, i] = RMSE * mask.float()

            loss_list.append(MSE_meanloss.item())
            loss_list.append(RMSE_meanloss.item())
        else:
            loss = torch.mean(se_loss[:, i])
            loss_list.append(loss.item())
            MSE_loss[:, i] = se_loss[:, i]  # Copy over the entire squared error for this feature

    full_loss = torch.mean(MSE_loss)  # Calculate the final average loss
    return loss_list, full_loss



        
        
# def l2_regularization(model, lambda_reg):
#     l2_reg = 0.0
#     for W in model.parameters():
#         l2_reg += torch.sum(W ** 2)
#     return l2_reg * lambda_reg      
        



# %%
# hidden_layers=[64,72,82,92,102,112,122,132,142,132,122,112,102,92,82,72,64]
# hidden_layers=[64,72,82,92,102,112,122,132,142,132,102,92,82]

model = CustomKinematicNet(input_size=input_dim, hidden_layers=hidden_layers, lenoutput=output_dim, activation_fn=F.tanh)
model.to(device)

# %%
out_feats=['deltaphi', 'deltaeta', 'deltaR', 'mt', 'norm_mt', 'mass', 'pt', 'eta' , 'phi',  'px', 'py', 'pz', 'energy']

df_outfeats=[]
for i, feat in enumerate(out_feats):
    if i in customlossindices:
        df_outfeats.append(feat +"_MSE")
        df_outfeats.append(feat +"_RMSE")
    else:
        df_outfeats.append(feat +"_MSE")

losses_cols=['train_loss', 'val_loss']+df_outfeats

print(losses_cols)



optimizer=torch.optim.AdamW(model.parameters(), lr=0.0001)
# loss_fn=nn.MSELoss()


out_feats=['deltaphi', 'deltaeta', 'deltaR', 'mt', 'norm_mt', 'mass', 'pt', 'eta' , 'phi',  'px', 'py', 'pz', 'energy']

df_outfeats=[]
for i, feat in enumerate(out_feats):
    if i in customlossindices:
        df_outfeats.append(feat +"_MSE")
        df_outfeats.append(feat +"_RMSE")
    else:
        df_outfeats.append(feat +"_MSE")

losses_cols=['train_loss', 'val_loss']+df_outfeats
losses_df=pd.DataFrame(columns=losses_cols)
losses_train_df=pd.DataFrame(columns=df_outfeats)


numepochs=10000
best_loss=np.inf
for epoch in range(numepochs):
    model.train()
    train_loss=0
    train_featsloss_list=[]
    for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, position=0, disable=True):
        x=x.to(device)
        y=y.to(device)
        y_pred=model(x)
        train_featslosssep, original_loss = custom_loss(y_pred, y)
        train_featsloss_list.append(train_featslosssep)
        loss = original_loss
        train_loss += original_loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_featsloss_array = np.array(train_featsloss_list)
    avg_train_featsloss = np.mean(train_featsloss_array, axis=0)
    
    model.eval()
    # patience=20
    with torch.no_grad():
        x,y=next(iter(val_loader))
        # x=val_loader
        x=x.to(device)
        y=y.to(device)
        y_pred=model(x)
        feats_loss, valloss = custom_loss(y_pred, y)
        # valloss=sum(feats_loss)/len(feats_loss)
        

        if valloss<best_loss:
            best_loss=valloss
            patience=100
            modelsave=deepcopy(model.state_dict())
            torch.save(modelsave, modelsavepath)
        else:
            patience-=1
            if patience==0:
                print('early stopping')
                break
    # indice=[3,6,12]
    # for idx in indice:
    #     feats_loss[idx]=feats_loss[idx].cpu().float()
    
    loss_strings = [f"{df_outfeats[i]}: {feats_loss[i]:.4e}" for i in range(len(df_outfeats))]
    loss_summary = ", ".join(loss_strings)
    loss_values = [train_loss/len(train_loader), valloss.item()]
    loss_values.extend(feats_loss)

    losses_df.loc[epoch] = loss_values
    losses_df.to_csv(pdsavepath)

    losses_train_df.loc[epoch] = avg_train_featsloss
    losses_train_df.to_csv(pd_train_savepath)
    


    # print(f"epoch: {epoch}, train: {train_loss/len(train_loader):.4e}, val: {valloss.item():.4e}, {loss_summary}")
    # print("train loss features:", avg_train_featsloss)
    print(f"epoch: {epoch}, train: {train_loss/len(train_loader):.4e}, val: {valloss.item():.4e}")

    # valloss=valloss.cpu().float()
    # valloss2=valloss.item()
    # loss_strings = [f"{out_feats[i]}: {feats_loss[i]:.4e}" for i in range(len(out_feats))]
    # loss_summary = ", ".join(loss_strings)
    # loss_values = [train_loss/len(train_loader), valloss2, l2sum]
    # loss_values.extend(feats_loss)
    # # for losses in loss_values:
    #     # print(type(losses))
    # losses_df.loc[epoch] = loss_values
    # losses_df.to_csv(pdsavepath)
    # # losses_df.loc[epoch]=[train_loss/len(train_loader), valloss2, feats_loss[0], feats_loss[1]]
    # print(f"epoch: {epoch}, train: {train_loss/len(train_loader):.4e}, val: {valloss2:.4e}, {loss_summary}")


