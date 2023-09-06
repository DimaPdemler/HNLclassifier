# %%
import numpy as np
import sys
import os
import vector
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from kinematic_custom import *
from fnn_datagenerator import BatchedFakeParticleDataset_All, flat_output_vars
import pickle

# %%
train_batch_size = 320
activation=F.relu
hidden_layers = [30, 100,300]+[512 for i in range(20)]
prefix='try2'

#! copied from try 1. 


dfpath='/home/ddemler/HNLclassifier/saved_files/fnn_featregr/all_data/' + prefix + '_losses.csv'
modelsavepath='/home/ddemler/HNLclassifier/saved_files/fnn_featregr/all_data/' + prefix + '_model.pt'

# %%

train_dataset = BatchedFakeParticleDataset_All(batch_size=train_batch_size, length=500_000)

train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=8)

val_dataset = BatchedFakeParticleDataset_All(batch_size=1000, length=200_000)
val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=True, num_workers=2)


input_dim = train_dataset.input_dim
output_dim = train_dataset.output_dim
print(input_dim, output_dim)


# %%
# for batch_idx, (input_tensors, output_tensors) in enumerate(train_dataloader):
#     print(f"Batch {batch_idx + 1}:")
#     print(f"Input tensors keys: {list(input_tensors.keys())}")
#     print(f"Output tensors keys: {list(output_tensors.keys())}")
#     print(f"Example input tensor ('MET_mass') shape: {input_tensors['MET_mass'].shape}")
#     print(f"Example output tensor ('deltaphi_12') shape: {output_tensors['deltaphi_12'].shape}")
#     # For demonstration purposes, we'll break after printing the first batch
#     break

# %%
class CustomKinematicNet(nn.Module):
    def __init__(self, input_size, hidden_layers, lenoutput, activation_fn=F.relu):
        super(CustomKinematicNet, self).__init__()
        
        layers = []
        for i in range(len(hidden_layers)):
            # Check if this is a layer where we reintroduce the inputs
            if (i % 3 == 0) and (i != 0):
                in_features = hidden_layers[i-1] + input_size
            else:
                in_features = hidden_layers[i-1] if i > 0 else input_size
            out_features = hidden_layers[i]
            layers.append(nn.Linear(in_features, out_features))
        
        # Adjusting the input size for the final layer
        if len(hidden_layers) % 3 == 0:
            layers.append(nn.Linear(hidden_layers[-1] + input_size, lenoutput))
        else:
            layers.append(nn.Linear(hidden_layers[-1], lenoutput))
        
        self.layers = nn.ModuleList(layers)

        for layer in self.layers:
            # Apply He Initialization
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


        self.activation_fn = activation_fn
        
    def forward(self, x):
        inputs = x
        for idx, layer in enumerate(self.layers[:-1]):
            if (idx % 3 == 0) and (idx != 0):
                x = self.activation_fn(layer(torch.cat((x, inputs), dim=-1)))
            else:
                x = self.activation_fn(layer(x))
        
        # Check if the output layer needs the original inputs
        if (len(self.layers) - 1) % 3 == 0:
            return self.layers[-1](torch.cat((x, inputs), dim=-1))
        return self.layers[-1](x)

# %%
def custom_loss(y_pred, y_true):
    se_loss = (y_pred - y_true) ** 2
    # print(se_loss.shape)
    num_features = int(output_dim)

    loss_list=[]

    for i in range(num_features -1):

        MSE_meanloss = torch.mean(se_loss[:,i])

        

        loss_list.append(MSE_meanloss.item())
    
    #RMSE for the last feature
    # ((y_pred[:,i]-y_true[:,i])/y_true[:,i])**2
    RMSE_loss = ((y_pred[:,i]-y_true[:,i])/ y_true[:,i])**2
    loss_list.append(torch.mean(RMSE_loss).item())
    se_loss[:,num_features-1] = RMSE_loss
        
    
    full_loss = torch.mean(se_loss)
    return loss_list, full_loss

# %%
model = CustomKinematicNet(input_dim, hidden_layers, output_dim, activation_fn=activation)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.5, verbose=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs=100000

# %%
dfcols=flat_output_vars
dfcols = ['train_loss', 'val_loss'] + dfcols
# print(len(dfcols))
if __name__ == "__main__":
    losses_df=pd.DataFrame(columns=dfcols)

    patience=100
    min_loss=np.inf
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_dataloader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)

            loss_list, loss = custom_loss(outputs, target)
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(train_dataloader.dataset)
        

        model.eval()
        with torch.no_grad():
            val_loss=0
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                loss_list, loss = custom_loss(outputs, target)
                val_loss += loss.item()
            val_loss /= len(val_dataloader.dataset)
            if val_loss < min_loss:
                min_loss=val_loss
                torch.save(model.state_dict(), modelsavepath)
                best_modelchanged=True
                patience=100
            else:
                patience-=1
                if patience==0:
                    print("Early stopping!!", epoch)
                    break
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if old_lr != new_lr and best_modelchanged:
            print("Loading best model due to learning rate decrease...")
            model.load_state_dict(torch.load(modelsavepath))
            best_modelchanged=False

        loss_feats=np.array(loss_list)
        # losses_full=np.array([total_loss]+list(loss_feats))
        losses_full=np.array([total_loss, val_loss]+list(loss_feats))
        
        losses_df.loc[epoch]=losses_full
        # print("loss feats before mean:", loss_feats.shape)
        # loss_feats=np.mean(loss_feats, axis=0)
        # print("loss feats after mean:", loss_feats.shape)
        pd.DataFrame(losses_df).to_csv(dfpath, index=False)
        print("Epoch: {}, Training Loss: {:.4e}, Validation Loss: {:.4e}".format(epoch, total_loss, val_loss))



