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


sys.path.append('../../utils/')
from DD_data_extractor_git import generate_random_data



# %%
class FakeParticleDataset(Dataset):


    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        phi, eta, mass, pt = self.generate_input_data()
        px, py, pz, energy = self.generate_output_data(phi, eta, mass, pt)
        
        input_tensor = torch.tensor([phi, eta, mass, pt], dtype=torch.float32)
        output_tensor = torch.tensor([px, py, pz, energy], dtype=torch.float32)
        
        return input_tensor, output_tensor

    
    def generate_input_data(self):
        eta_low, eta_high = -2.5, 2.5
        mass_low, mass_high = 0, 11
        phi_low, phi_high = -np.pi, np.pi
        pt_low, pt_high = 0, 1000

        eta= np.random.uniform(eta_low, eta_high)
        mass = np.random.uniform(mass_low, mass_high)
        phi = np.random.uniform(phi_low, phi_high)
        pt = np.random.uniform(pt_low, pt_high)

        return phi, eta, mass, pt
    
    def generate_output_data(self, phi, eta, mass, pt):
        particle=vector.obj(pt=pt, phi=phi, eta=eta, mass=mass)

        px = particle.px
        py = particle.py
        pz = particle.pz
        energy = particle.e

        return px, py, pz, energy

def worker_init_fn(worker_id):
    numpy_seed = int(torch.initial_seed()) % (2**32 - 1)
    np.random.seed(numpy_seed + worker_id)

# %%
dataset= FakeParticleDataset()
train_loader = DataLoader(dataset, batch_size=3200, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
input_dim,output_dim=4,4

# %%
class CustomKinematicNet(nn.Module):
    def __init__(self, input_size, hidden_layers, lenoutput, activation_fn=F.relu):
        
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

def custom_loss(y_pred, y_true):
    # print("y_pred:", y_pred.shape)
    se_loss = (y_pred - y_true) ** 2
    MSE_loss = torch.zeros_like(se_loss)  # Initialize with zeros
    
    num_features = int(output_dim)
    loss_list = []

    for i in range(num_features):
    
        RMSE = ((y_pred[:, i] - y_true[:, i]) / y_true[:, i]) ** 2
        mask = (y_true[:, i] > 1)
        
        RMSE_meanloss = torch.mean(RMSE[mask])
        MSE_meanloss = torch.mean(se_loss[:, i][~mask])

        # Weighted contribution of the RMSE for the masked values to the final loss tensor
        MSE_loss[:, i] = RMSE * mask.float()

        loss_list.append(MSE_meanloss.item())
        loss_list.append(RMSE_meanloss.item())
        
    full_loss = torch.mean(MSE_loss)  # Calculate the final average loss
    return loss_list, full_loss


hidden_layers = [16 for i in range(10)]
model = CustomKinematicNet(input_dim, hidden_layers, output_dim, activation_fn=F.tanh)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
epochs=100000

dfpath='/home/ddemler/HNLclassifier/saved_files/fnn_featregr/single_particle/losses1.csv'
dfcols=["train_loss","px_mse", "px_rmse", "py_mse", "py_rmse", "pz_mse", "pz_rmse", "energy_mse", "energy_rmse"]
losses_df=pd.DataFrame(columns=dfcols)

patience=100
min_loss=np.inf
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        loss_list, loss = custom_loss(outputs, target)
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()

    total_loss /= len(train_loader.dataset)
    

    model.eval()
    with torch.no_grad():
        val_loss=0
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            loss_list, loss = custom_loss(outputs, target)
            val_loss += loss.item()
        val_loss /= len(val_loader.dataset)
        if val_loss < min_loss:
            min_loss=val_loss
            patience=100
        else:
            patience-=1
            if patience==0:
                break
    loss_feats=np.array(loss_list)
    losses_full=np.array([total_loss]+list(loss_feats))
    
    losses_df.loc[epoch]=losses_full
    # print("loss feats before mean:", loss_feats.shape)
    # loss_feats=np.mean(loss_feats, axis=0)
    # print("loss feats after mean:", loss_feats.shape)
    pd.DataFrame(losses_df).to_csv(dfpath, index=False)
    print("Epoch: {}, Training Loss: {:.4e}, Validation Loss: {:.4e}".format(epoch, total_loss, val_loss))
# %%



