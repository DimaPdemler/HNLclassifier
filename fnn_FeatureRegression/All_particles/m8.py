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

activation=F.gelu
hidden_layers = [512 for i in range(20)]
prefix='m8'


train_batch_size = 320
dfpath='/home/ddemler/HNLclassifier/saved_files/fnn_featregr/all_data/wt_' + prefix + '_losses.csv'
modelsavepath='/home/ddemler/HNLclassifier/saved_files/fnn_featregr/all_data/wt_' + prefix + '_model.pt'

# %%

train_dataset = BatchedFakeParticleDataset_All(batch_size=train_batch_size, length=1_000_000)

train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=False, num_workers=8)

val_dataset = BatchedFakeParticleDataset_All(batch_size=1000, length=500_000)
val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=2)


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
    def __init__(self, input_size, hidden_layers, lenoutput, activation_fn):
        super(CustomKinematicNet, self).__init__()
        
        layers = []
        activations = []  
        for i in range(len(hidden_layers)):
            if (i % 3 == 0) and (i != 0):
                in_features = hidden_layers[i-1] + input_size
            else:
                in_features = hidden_layers[i-1] if i > 0 else input_size
            out_features = hidden_layers[i]
            
            layers.append(nn.Linear(in_features, out_features))
            activations.append(nn.PReLU())  # Add a PReLU activation for each layer
        
        if len(hidden_layers) % 3 == 0:
            layers.append(nn.Linear(hidden_layers[-1] + input_size, lenoutput))
        else:
            layers.append(nn.Linear(hidden_layers[-1], lenoutput))
        
        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)  # Convert activations list to ModuleList
        
    def forward(self, x):
        inputs = x
        for idx, layer in enumerate(self.layers[:-1]):
            if (idx % 3 == 0) and (idx != 0):
                x = self.activations[idx](layer(torch.cat((x, inputs), dim=-1)))  # Use corresponding PReLU activation
            else:
                x = self.activations[idx](layer(x))
        
        if (len(self.layers) - 1) % 3 == 0:
            return self.layers[-1](torch.cat((x, inputs), dim=-1))
        return self.layers[-1](x)

# %%
def custom_loss(y_pred, y_true):
    se_loss = (y_pred - y_true) ** 2
    # print(se_loss.shape)
    num_features = int(output_dim)

    # loss_list=[]

    
    #RMSE for the last feature
    # ((y_pred[:,i]-y_true[:,i])/y_true[:,i])**2
    RMSE_loss = ((y_pred[:,num_features-1]-y_true[:,num_features-1])/ y_true[:,num_features-1])**2
    # loss_list.append(torch.mean(RMSE_loss).item())
    se_loss[:,num_features-1] = RMSE_loss
        
    full_loss= torch.mean(se_loss, axis=0)

    loss_list = full_loss.detach().cpu().numpy()

    full_loss = torch.sum(full_loss)
    return loss_list, full_loss

# %%
model = CustomKinematicNet(input_dim, hidden_layers, output_dim, activation_fn=activation)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, betas=(0.99,0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5, verbose=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs=100000

# %%
dfcols=flat_output_vars
dfcols = ['train_loss', 'val_loss', 'learning_rate'] + dfcols
# print(len(dfcols))
if __name__ == "__main__":
    losses_df=pd.DataFrame(columns=dfcols)

    patience=300
    min_loss=np.inf
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # E_tot_arr=np.array((train_batch_size*len(train_dataloader.dataset)))
        for batch_idx, (data, target) in enumerate(train_dataloader):

            # print("min, max of data[-1]", torch.min(data[:,-1]), torch.max(data[:,-1]))
            # print("min, max of target[-1]", torch.min(target[:,-1]), torch.max(target[:,-1]))
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)

            loss_list_t, loss = custom_loss(outputs, target)
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(train_dataloader.dataset)
        
        f_shape=(output_dim)
        full_loss_list=np.empty(f_shape)
        model.eval()
        with torch.no_grad():
            val_loss=0
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                loss_list, loss = custom_loss(outputs, target)
                # loss_list=np.array(loss_list)
                # print("loss_list shape", loss_list.shape)
                full_loss_list=np.vstack((full_loss_list, loss_list))
                val_loss += loss.item()
            val_loss /= len(val_dataloader.dataset)
            full_loss_list=np.mean(full_loss_list, axis=0)

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

        loss_feats=full_loss_list

        # loss_feats=np.mean(loss_feats, axis=0)
        # losses_full=np.array([total_loss]+list(loss_feats))
        # print("full loss list shape",full_loss_list.shape)
        losses_full=np.array([total_loss, val_loss, old_lr]+list(loss_feats))
        
        losses_df.loc[epoch]=losses_full
        # print("loss feats before mean:", loss_feats.shape)
        # loss_feats=np.mean(loss_feats, axis=0)
        # print("loss feats after mean:", loss_feats.shape)
        pd.DataFrame(losses_df).to_csv(dfpath, index=False)
        print("Epoch: {}, Training Loss: {:.4e}, Validation Loss: {:.4e}".format(epoch, total_loss, val_loss))




