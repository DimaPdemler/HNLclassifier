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


batchsize=320
activation=F.relu
hidden_layers = [8,16,32,64,64,64,64,50,32,32]
# hidden_layers = [128 for i in range(10)]
prefix='itry1'

#! copied from try 12. Using prelu instead of relu


dfpath='/home/ddemler/HNLclassifier/saved_files/fnn_featregr/single_particle/inv_' + prefix + '_losses.csv'
modelsavepath='/home/ddemler/HNLclassifier/saved_files/saved_models/FNN_FeatureRegression/Single_particle/inv_' + prefix + '_model.pt'



# %%
class FakeParticleDataset(Dataset):


    def __len__(self):
        return 100000
    
    def __getitem__(self, idx):
        phi, eta, mass, pt = self.generate_input_data()
        px, py, pz, energy = self.generate_output_data(phi, eta, mass, pt)
        
        input_tensor = torch.tensor([phi, eta, mass, pt], dtype=torch.float32)
        output_tensor = torch.tensor([px, py, pz, energy], dtype=torch.float32)
        
        return output_tensor, input_tensor

    
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
train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
input_dim,output_dim=4,4

val_dataset= FakeParticleDataset()
val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)

# %%

class CustomKinematicNet(nn.Module):
    def __init__(self, input_size, hidden_layers, lenoutput, activation_fn):
        super(CustomKinematicNet, self).__init__()
        
        layers = []
        activations = []  # List to store PReLU activations
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



def custom_loss(y_pred, y_true):
    se_loss = (y_pred - y_true) ** 2
    # print(se_loss.shape)
    num_features = int(output_dim)

    loss_list=[]

    # for i in range(num_features):

        
    #     RMSE=((y_pred[:,i]-y_true[:,i])/y_true[:,i])**2
    #     mask = (y_true[:,i] > 1)

    #     RMSE_meanloss=torch.mean(RMSE[mask])
    #     MSE_meanloss = torch.mean(se_loss[:,i][~mask])

    #     se_loss[mask, i] = RMSE[mask]

    #     loss_list.append(MSE_meanloss.item())
    #     loss_list.append(RMSE_meanloss.item())
        
    
    full_loss = torch.mean(se_loss)
    return loss_list, full_loss


model = CustomKinematicNet(input_dim, hidden_layers, output_dim, activation_fn=activation)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.006)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, verbose=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
epochs=1000000


dfcols=["train_loss","val_loss","px_mse", "px_rmse", "py_mse", "py_rmse", "pz_mse", "pz_rmse", "energy_mse", "energy_rmse"]
if __name__ == "__main__":
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
                torch.save(model.state_dict(), modelsavepath)
                patience=100
            else:
                patience-=1
                if patience==0:
                    print("Early stopping!!", epoch)
                    break
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr < old_lr:  # Learning rate has decreased
            print("Loading best model due to learning rate decrease...")
            model.load_state_dict(torch.load(modelsavepath))
        loss_feats=np.array(loss_list)
        # losses_full=np.array([total_loss]+list(loss_feats))
        losses_full=np.array([total_loss, val_loss]+list(loss_feats))
        
        losses_df.loc[epoch]=losses_full
        # print("loss feats before mean:", loss_feats.shape)
        # loss_feats=np.mean(loss_feats, axis=0)
        # print("loss feats after mean:", loss_feats.shape)
        pd.DataFrame(losses_df).to_csv(dfpath, index=False)
        print("Epoch: {}, Training Loss: {:.4e}, Validation Loss: {:.4e}".format(epoch, total_loss, val_loss))

# %%



