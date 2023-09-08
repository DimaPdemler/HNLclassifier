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
from tqdm import tqdm
import pickle
from scipy import stats
from torch.nn.utils import prune
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is
yaml_path = os.path.join(script_dir, 'simpleRegressionmodel.yaml')  # Construct the full path to the YAML file

with open(yaml_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Map the activation function string to the actual function
activation_function_str = config.get('activation', 'F.relu')
activation = eval(activation_function_str)

# Other configurations
hidden_layers = config.get('hidden_layers', [512 for _ in range(20)])
prefix = config.get('prefix', 'defaultmodel')
lr_start = config.get('lr_start', 4e-4)
# lr_patience = config.get('lr_patience', 25)
patience_model = config.get('patience_model', 75)
train_batch_size = config.get('train_batch_size', 320)
optimizer_str = config.get('optimizer', 'torch.optim.Adam')
optimizer_params = config.get('optimizer_params', {})
scheduler_str = config.get('scheduler', 'plateau')
scheduler_params = config.get('scheduler_params', {})
cuda = config.get('device', 'cuda:0')
train_length=config.get('training_size', 2_000_000)
val_length=config.get('validation_size', 2_000_000)
test_length=config.get('testing_size', 5_000_000)

dfpath= '../saved_files/fnn_featregr/' + prefix + '/losses.csv'
print(dfpath)
modelsavepath='../saved_files/fnn_featregr/' + prefix + '/model.pt'
pdfpath='../saved_files/fnn_featregr/' + prefix + '/plots.pdf'

if not os.path.exists('../saved_files/fnn_featregr/' + prefix):
    os.makedirs('../../saved_files/fnn_featregr/' + prefix)









train_dataset = BatchedFakeParticleDataset_All(batch_size=train_batch_size, length=train_length)

train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=False, num_workers=3)

val_dataset = BatchedFakeParticleDataset_All(batch_size=10000, length=val_length)
val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=3)


input_dim = train_dataset.input_dim
output_dim = train_dataset.output_dim
print(input_dim, output_dim)


# class CustomKinematicNet(nn.Module):
#     def __init__(self, input_size, hidden_layers, lenoutput, activation_fn=F.relu):
#         super(CustomKinematicNet, self).__init__()
        
#         layers = []
#         for i in range(len(hidden_layers)):
#             # Check if this is a layer where we reintroduce the inputs
#             if (i % 3 == 0) and (i != 0):
#                 in_features = hidden_layers[i-1] + input_size
#             else:
#                 in_features = hidden_layers[i-1] if i > 0 else input_size
#             out_features = hidden_layers[i]
#             layers.append(nn.Linear(in_features, out_features))
        
#         # Adjusting the input size for the final layer
#         if len(hidden_layers) % 3 == 0:
#             layers.append(nn.Linear(hidden_layers[-1] + input_size, lenoutput))
#         else:
#             layers.append(nn.Linear(hidden_layers[-1], lenoutput))
        
#         self.layers = nn.ModuleList(layers)

#         for layer in self.layers:
#             # Apply He Initialization
#             nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
#             if layer.bias is not None:
#                 nn.init.constant_(layer.bias, 0)


#         self.activation_fn = activation_fn
        
#     def forward(self, x):
#         inputs = x
#         for idx, layer in enumerate(self.layers[:-1]):
#             if (idx % 3 == 0) and (idx != 0):
#                 x = self.activation_fn(layer(torch.cat((x, inputs), dim=-1)))
#             else:
#                 x = self.activation_fn(layer(x))
        
#         # Check if the output layer needs the original inputs
#         if (len(self.layers) - 1) % 3 == 0:
#             return self.layers[-1](torch.cat((x, inputs), dim=-1))
#         return self.layers[-1](x)

class CustomKinematicNet(nn.Module):
    def __init__(self, input_size, hidden_layers, lenoutput, activation_fn=F.relu, dropout_p=0.0):
        super(CustomKinematicNet, self).__init__()
        
        layers = []
        for i in range(len(hidden_layers)):
            # Check if this is a layer where we reintroduce the inputs
            if (i % 3 == 0) and (i != 0):
                in_features = hidden_layers[i-1] + input_size
            else:
                in_features = hidden_layers[i-1] if i > 0 else input_size
            out_features = hidden_layers[i]
            
            # Add the linear layer
            layers.append(nn.Linear(in_features, out_features))
            
            # Add dropout layer
            layers.append(nn.Dropout(dropout_p))
        
        # Adjusting the input size for the final layer
        if len(hidden_layers) % 3 == 0:
            layers.append(nn.Linear(hidden_layers[-1] + input_size, lenoutput))
        else:
            layers.append(nn.Linear(hidden_layers[-1], lenoutput))
        
        self.layers = nn.ModuleList(layers)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Apply He Initialization
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        self.activation_fn = activation_fn

        # for i, layer in enumerate(self.layers):
        #     print(i, layer)
        
    def forward(self, x):
        inputs = x
        for idx, layer in enumerate(self.layers[:-1]):
            if isinstance(layer, nn.Dropout):
                x = layer(x)
            elif (idx % 6 == 0) and (idx != 0):
                # print(idx, layer, x.shape, inputs.shape)
                x = self.activation_fn(layer(torch.cat((x, inputs), dim=-1)))
            else:
                x = self.activation_fn(layer(x))
        
        # Check if the output layer needs the original inputs
        # if (len(self.layers) - 1) % 4 == 0:
        #     return self.layers[-1](torch.cat((x, inputs), dim=-1))
        return self.layers[-1](x)


def custom_loss(y_pred, y_true):
    se_loss = (y_pred - y_true) ** 2
    # print(se_loss.shape)
    num_features = int(output_dim)

    RMSE_loss = ((y_pred[:,num_features-1]-y_true[:,num_features-1])/ y_true[:,num_features-1])**2
    # loss_list.append(torch.mean(RMSE_loss).item())
    se_loss[:,num_features-1] = RMSE_loss
        
    full_loss= torch.mean(se_loss, axis=0)

    loss_list = full_loss.detach().cpu().numpy()

    full_loss = torch.sum(full_loss)
    return loss_list, full_loss


model = CustomKinematicNet(input_dim, hidden_layers, output_dim, activation_fn=activation)

optimizer = eval(optimizer_str)(model.parameters(), lr=lr_start, **optimizer_params)

if scheduler_str == 'plateau':
    lr_patience = scheduler_params.get('lr_patience', 25)
    factor = scheduler_params.get('factor', 0.3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=factor, verbose=True)
elif scheduler_str == 'exponential':
    gamma = scheduler_params.get('gamma', 0.99)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=-1, verbose=False)


device = torch.device(cuda if torch.cuda.is_available() else "cpu")
model.to(device)

epochs=100000

# %%
dfcols=flat_output_vars
dfcols = ['train_loss', 'val_loss', 'learning_rate'] + dfcols
# print(len(dfcols))
if __name__ == "__main__":
    losses_df=pd.DataFrame(columns=dfcols)

    min_loss=np.inf
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_dataloader):


            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)

            loss_list_t, loss = custom_loss(outputs, target)
            loss.backward()

            clip_norm = 1.0
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            
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

                full_loss_list=np.vstack((full_loss_list, loss_list))
                val_loss += loss.item()
            val_loss /= len(val_dataloader.dataset)
            full_loss_list=np.mean(full_loss_list, axis=0)

            if val_loss < min_loss:
                min_loss=val_loss
                torch.save(model.state_dict(), modelsavepath)
                savedmodel=model
                best_modelchanged=True
                patience=patience_model
            else:
                patience-=1
                if patience==0:
                    print("Early stopping!!", epoch)
                    break

        old_lr = optimizer.param_groups[0]['lr']
        if scheduler_str == 'plateau':
            scheduler.step(val_loss)
        elif scheduler_str == 'exponential':
            scheduler.step()

        loss_feats=full_loss_list

        losses_full=np.array([total_loss, val_loss, old_lr]+list(loss_feats))
        
        losses_df.loc[epoch]=losses_full

        pd.DataFrame(losses_df).to_csv(dfpath, index=False)
        print("Epoch: {}, Training Loss: {:.4e}, Validation Loss: {:.4e}, LR: {:.4e}".format(epoch, total_loss, val_loss, old_lr))

    from matplotlib.backends.backend_pdf import PdfPages 

    torch.cuda.empty_cache()

    test_dataset = BatchedFakeParticleDataset_All(batch_size=1_000, length=test_length)
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False, num_workers=2)
    input_dim, output_dim=14,112


    model=CustomKinematicNet(input_dim, hidden_layers, output_dim, activation_fn=activation)
    model.load_state_dict(torch.load(modelsavepath))
    model.to(device)



    import matplotlib.pyplot as plt

    pdf_path = pdfpath

    print("beginning plotting...")
    # Define a function to compute moving average
    def moving_average(data, window_size):
        return data.rolling(window=window_size).mean()

    with PdfPages(pdf_path) as pdf:
        fig= plt.figure(figsize=(8, 6))

        plt.text(0.1, 0.9, f"hidden_layers: {hidden_layers}")
        plt.text(0.1, 0.8, f"activation: {activation}")
        plt.text(0.1, 0.7, f"dfpath: {dfpath}")
        plt.text(0.1, 0.6, f"modelsavepath: {modelsavepath}")
        plt.text(0.1, 0.5, f"prefix: {prefix}")
        plt.axis('off')
        pdf.savefig(fig)
        plt.close()

        out_feats=flat_output_vars

        # List to collect data batch-wise
        y_total_list = []
        y_pred_total_list = []

        # for i, (x, y) in enumerate(test_loader):
        for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Predicting"):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            
            y_pred_total_list.append(y_pred.cpu().detach().numpy())
            y_total_list.append(y.cpu().detach().numpy())

        # Convert lists to arrays
        y_pred_total = np.vstack(y_pred_total_list)
        y_total = np.vstack(y_total_list)

        numfeatures = len(out_feats)

        residuals = [y_pred_total[:, i] - y_total[:, i] for i in range(numfeatures)]
        label_values = [y_total[:, i] for i in range(numfeatures)]
        residual_std_devs = [np.std(res) for res in residuals]
        residual_means = [np.mean(res) for res in residuals]


        # Number of plots per page
        plots_per_page = 8
        num_rows_per_page = 4
        num_cols_per_page = 2

        # Calculate the number of pages
        num_pages = -(-numfeatures // plots_per_page)  # Ceiling division

        # for page in range(num_pages):
        for page in tqdm(range(num_pages), desc="Plotting residuals"):
            start_idx = page * plots_per_page
            end_idx = start_idx + plots_per_page
            
            fig, axes = plt.subplots(nrows=num_rows_per_page, ncols=num_cols_per_page, figsize=(15, 20))  # Adjust the figsize as needed
            flat_axes = axes.flatten()
            
            for i, ax in enumerate(flat_axes):
                idx = start_idx + i
                if idx >= numfeatures:
                    ax.axis('off')  # Turn off extra subplots
                    continue
                
                ax.hist(residuals[idx], bins=100, edgecolor='k', alpha=0.65)
                ax.axvline(x=residual_means[idx] + residual_std_devs[idx], color='r', linestyle='--', label=f'+1 std = {residual_means[idx] + residual_std_devs[idx]:.2f})')
                ax.axvline(x=residual_means[idx] - residual_std_devs[idx], color='b', linestyle='--', label=f'-1 std = {residual_means[idx] - residual_std_devs[idx]:.2f})')
                ax.set_title(f'Residuals for {out_feats[idx]}')
                ax.set_yscale('log')
                ax.set_xlabel('Residual Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                
                # Display the mean value on the plot
                mean_text = f"Mean: {residual_means[idx]:.2f}, std: {residual_std_devs[idx]:.5f}"
                ax.text(0.4, 0.85, mean_text, transform=ax.transAxes)
            
            pdf.savefig(fig)
            plt.close()



        df=pd.read_csv(dfpath)

        plt.figure(figsize=(12, 6))
        plt.style.use('ggplot')

        # Plot train_loss and val_loss
        plt.plot(df.index, df['train_loss'], label='Train Loss', marker='o', linestyle='-')
        plt.plot(df.index, df['val_loss'], label='Validation Loss', marker='o', linestyle='-')

        # Set the title and labels
        plt.title('Train Loss vs Validation Loss of ' + prefix)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()

        # Display the plot
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        selected_features = ['E_tot']
        selected_indices = [111]

        fig, axes = plt.subplots(nrows=1, ncols=len(selected_features), figsize=(15, 5))

        # Ensure axes is a list or array
        if len(selected_features) == 1:
            axes = [axes]

        for idx, ax in enumerate(axes):
            feature_label_values = label_values[selected_indices[idx]]
            # print(feature_label_values)
            # bins = np.linspace(min(feature_label_values), max(feature_label_values), num=10)
            bins = np.logspace(np.log10(min(feature_label_values)), np.log10(max(feature_label_values)), num=10)
            
            # Use relative residuals for 'energy' and normal residuals for others
            if selected_features[idx] == 'E_tot':
                # print("using relative residuals")
                y_values = residuals[idx] / label_values[idx]
            else:
                y_values = residuals[idx]
            
            # Compute binned statistics (standard deviation)
            bin_stds, bin_edges, binnumber = stats.binned_statistic(feature_label_values,
                                                                    y_values,
                                                                    statistic='std',
                                                                    bins=bins)


            # Compute the uncertainty in standard deviation
            bin_counts, _, _ = stats.binned_statistic(label_values[idx],
                                                    y_values,
                                                    statistic='count',
                                                    bins=bins)

            bin_std_uncertainties = bin_stds / np.sqrt(2 * (bin_counts - 1))

            # Midpoints of the bins
            bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
            x_errors = np.diff(bin_edges) / 2

            # Scatter plot with error bars
            ax.errorbar(bin_midpoints, bin_stds, xerr=x_errors, yerr=bin_std_uncertainties, fmt='o', capsize=5)
            
            ax.set_yscale('log')
            ax.set_title(f'{selected_features[idx]}')
            ax.set_xlabel('True Value')
            ax.set_ylabel('Standard Deviation')

        pdf.savefig(fig)
        plt.close()




        # Window size for the moving average
        window_size = 5  # you can adjust this value as needed

        # Compute moving averages
        train_loss_smoothed = moving_average(df['train_loss'], window_size)
        val_loss_smoothed = moving_average(df['val_loss'], window_size)

        # Plotting
        fig = plt.figure(figsize=(12, 6))
        plt.style.use('ggplot')

        # Plot original data points
        plt.plot(df.index, df['train_loss'], 'o', label='Train Loss Points', alpha=0.2, color='blue')
        plt.plot(df.index, df['val_loss'], 'o', label='Validation Loss Points', alpha=0.2, color='red')

        # Plot smoothed lines
        plt.plot(df.index, train_loss_smoothed, label='Train Loss Smoothed', linestyle='-', color='blue')
        plt.plot(df.index, val_loss_smoothed, label='Validation Loss Smoothed', linestyle='-', color='red')

        # Set the title and labels
        plt.title('Train Loss vs Validation Loss (Smoothed)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()

        # Display the plot
        plt.grid(True)
        plt.tight_layout()
        # plt.show()

        pdf.savefig(fig)
        plt.close()

    print(f"Plots and information saved to {pdf_path}")