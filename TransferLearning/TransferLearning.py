import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
import os
import yaml

parentdir = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(parentdir, 'FeatureRegression'))

from regression_train import hidden_layers, CustomKinematicNet, modelsavepath, cuda, activation, flat_output_vars
from fnn_datagenerator import GeV_outputvars

sys.path.append(os.path.join(parentdir, 'DNN/'))
from pytorch2python import create_data_loaders, create_test_loader
from tqdm import tqdm

sys.path.append(os.path.join(parentdir, 'SignificancePlotting/'))

from Significance_func import find_significance2, find_significance, bin_uncertainty2,binmaker_rightleft,   error_boolean, binmaker, process_channels, process_dataframe
from copy import deepcopy

# from ..FeatureRegression import regression_train



yaml_path=os.path.join(parentdir, 'TransferLearning/Transfer_model1.yaml')
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

tf_dropout = config.get('dropout', 0.0)
numepochs = config.get('num_epochs', 1000)
tf_activation_str=config.get('activation', 'F.relu')
tf_activation = eval(tf_activation_str)
tf_hidden_layers = config.get('hidden_layers', [100, 100])
tf_optimizer_str = config.get('optimizer', 'torch.optim.AdamW')
tf_optimizer_params = config.get('optimizer_params', {'lr': 0.001, 'weight_decay': 0.0})
tf_saveprefix=config.get('saveprefix', 'generic_tfmodel')
transfermodelsave = os.path.join(parentdir, 'saved_files/transfer_learning/'+tf_saveprefix+'/')
datasetpath = config.get('datasetpath', 'Aug29')
scheduler_patience = config.get('scheduler_patience', 5)
scheduler_factor = config.get('scheduler_factor', 0.1)
unfreeze_epoch = config.get('unfreeze_epoch', 5)

model_patience = config.get('model_patience', 20)

if not os.path.exists(transfermodelsave):
    os.makedirs(transfermodelsave)

newmodelsavepath=os.path.join(transfermodelsave, 'model.pt')

data_base_path = os.path.join(parentdir, 'saved_files/extracted_data')

train_datapath = os.path.join(data_base_path, 'TEST10_train_' + datasetpath)
val_datapath = os.path.join(data_base_path, 'TEST10_val_' + datasetpath)
test_datapath = os.path.join(data_base_path, 'TEST10_test_' + datasetpath)

traindata = pd.read_pickle(train_datapath)
valdata = pd.read_pickle(val_datapath)

renamed_old_input_names=['eta_1', 'mass_1', 'phi_1', 'pt_1', 'eta_2', 'mass_2', 'phi_2', 'pt_2', 'eta_3', 'mass_3', 'phi_3', 'pt_3', 'phi_MET', 'pt_MET']
additionalinput_vars=['charge_1', 'charge_2', 'charge_3', 'channel','n_tauh', 'mass_hyp']
# output=traindata['signal_label']

def datamaker(data):
    outputdata_shape= (len(data['pt_MET']), len(flat_output_vars))
    inputdata_shape= (len(data['pt_MET']), len(renamed_old_input_names))

    inputdata=np.empty(inputdata_shape)
    outputdata=np.empty(outputdata_shape)

    for i, outvar in enumerate(flat_output_vars):
        if outvar in GeV_outputvars:
            outputdata[:,i]=data[outvar]/data['E_tot']
        else:
            outputdata[:,i]=data[flat_output_vars[i]]

    for i in range(len(renamed_old_input_names)):
        inputdata[:,i]=data[renamed_old_input_names[i]]

    input_tensor = torch.tensor(inputdata, dtype=torch.float32)
    output_tensor = torch.tensor(outputdata, dtype=torch.float32)

    return input_tensor, output_tensor


def additional_datamaker(data):
    n_samples = len(data['pt_MET'])
    outputdata_shape = (n_samples, 1)
    
    # Create a list to store the individual feature arrays
    inputdata_list = []
    
    for var in additionalinput_vars:
        feature_data = data[var]
        
        # Check if this is the 'channel' variable to one-hot encode
        if var == 'channel':
            # Perform one-hot encoding
            onehot_channel = np.eye(5)[feature_data.astype(int)]
            inputdata_list.append(onehot_channel)
        else:
            inputdata_list.append(feature_data.to_numpy().reshape(-1, 1))
            
    # Concatenate all the feature arrays horizontally
    inputdata = np.hstack(inputdata_list)
    
    outputdata = np.empty(outputdata_shape)
    outputdata[:, 0] = data['signal_label']
    
    input_tensor = torch.tensor(inputdata, dtype=torch.float32)
    output_tensor = torch.tensor(outputdata, dtype=torch.float32)
    
    return input_tensor, output_tensor


train_weight=traindata['weightNorm'].to_numpy()
val_weight=valdata['weightNorm'].to_numpy()

# print(type(train_weight))

train_weight_tensor=torch.tensor(train_weight, dtype=torch.float32)
val_weight_tensor=torch.tensor(val_weight, dtype=torch.float32)


train_inputdata, train_outputdata=datamaker(traindata)
val_inputdata, val_outputdata=datamaker(valdata)

ad_train_inputdata, ad_train_outputdata=additional_datamaker(traindata)
ad_val_inputdata, ad_val_outputdata=additional_datamaker(valdata)

train_inputfull=torch.cat((train_inputdata, ad_train_inputdata), dim=1)
val_inputfull=torch.cat((val_inputdata, ad_val_inputdata), dim=1)

device = torch.device(cuda if torch.cuda.is_available() else "cpu")

pretrained_model= CustomKinematicNet(input_size=14, hidden_layers=hidden_layers, lenoutput=len(flat_output_vars), activation_fn=activation, dropout_p = tf_dropout)
pretrained_model.load_state_dict(torch.load(modelsavepath))
pretrained_model.to(device)

import torchsummary

torchsummary.summary(pretrained_model, input_size=(14,))

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super(FeatureExtractor, self).__init__()
        self.pretrained = pretrained_model
        self.pretrained.layers = self.pretrained.layers[:-1]  # Remove the last layer

    def forward(self, x):
        return self.pretrained(x)
    


feature_extractor = FeatureExtractor(pretrained_model)
feature_extractor.to(device)

class TransferCustomKinematicNet(nn.Module):
    def __init__(self, feature_extractor, additional_input_size, new_hidden_layers):
        super(TransferCustomKinematicNet, self).__init__()
        
        self.feature_extractor = feature_extractor
        
        # Freeze the pre-trained layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Define new hidden layers
        input_size = feature_extractor.pretrained.layers[-3].out_features + additional_input_size
        # print("input size", input_size)
        layer_sizes = [input_size] + new_hidden_layers + [1]  # Final output size is 1 for binary classification
        # print("layer sizes", layer_sizes)
        
        self.new_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.new_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        self.activation_fn = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, x, additional_input):
        # print("x shape",x.shape)
        features = self.feature_extractor(x)
        combined_input = torch.cat((features, additional_input), dim=-1)
        
        out = combined_input
        for i in range(len(self.new_layers) - 1):
            # print("out shape", out.shape)
            # print("new layer shape", self.new_layers[i])
            out = self.activation_fn(self.new_layers[i](out))
        
        out = self.output_activation(self.new_layers[-1](out))
        return out
    


train_dataset = torch.utils.data.TensorDataset(train_inputfull, ad_train_outputdata, train_weight_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=320, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(val_inputfull, ad_val_outputdata, val_weight_tensor)
val_dataloader = DataLoader(val_dataset, batch_size=320, shuffle=False)

# Initialize the feature extractor with the pre-trained model
feature_extractor = FeatureExtractor(pretrained_model=pretrained_model)

# Initialize the new model
# Assuming YOUR_ADDITIONAL_INPUT_SIZE is the number of additional features you have
new_hidden_layers=[128, 128]
new_model = TransferCustomKinematicNet(feature_extractor, additional_input_size=10, new_hidden_layers=new_hidden_layers)
new_model.to(device)
# print("hidden layers: ", new_hidden_layers)

optimizer = eval(tf_optimizer_str)(new_model.parameters(), **tf_optimizer_params)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, factor=scheduler_factor, verbose=True)

best_valloss = np.inf

for epoch in range(numepochs):
    if epoch == unfreeze_epoch:
        print("Unfreezing feature extractor layers...")
        for param in new_model.feature_extractor.parameters():
            param.requires_grad = True
    new_model.train()
    weightedfull_trainloss=0

    
    for i, (combined_input, target, sample_weight) in enumerate(train_dataloader):
        # Split the combined input into input for the pre-trained model and additional input
        pretrained_input = combined_input[:, :train_inputdata.shape[1]].to(device)
        additional_input = combined_input[:, train_inputdata.shape[1]:].to(device)
        target = target.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = new_model(pretrained_input, additional_input)
        
        # Compute Loss without reduction
        loss_unreduced = criterion(outputs, target)
        
        # Apply sample weights and compute the final loss
        weighted_loss = (loss_unreduced * sample_weight.to(device)).sum()

        weightedfull_trainloss += weighted_loss.item()
        
        # Backward pass and optimization
        weighted_loss.backward()
        optimizer.step()
    
    weightedfull_trainloss /= len(train_dataloader.dataset)
    new_model.eval()
    with torch.no_grad():
        val_loss = 0
        for i, (combined_input, target, sample_weight) in enumerate(val_dataloader):
            # Split the combined input into input for the pre-trained model and additional input
            pretrained_input = combined_input[:, :train_inputdata.shape[1]].to(device)
            additional_input = combined_input[:, train_inputdata.shape[1]:].to(device)
            target = target.to(device)

            # Forward pass
            outputs = new_model(pretrained_input, additional_input)

            #debugging
            # print("sum outputs", torch.sum(outputs))
            # print("outputs", outputs[:10])
            
            # Compute Loss without reduction
            loss_unreduced = criterion(outputs, target)
            
            # Apply sample weights and compute the final loss
            weighted_loss = (loss_unreduced * sample_weight.to(device)).sum()
            
            val_loss += weighted_loss.item()
        
        val_loss /= len(val_dataloader.dataset)
        
        if val_loss < best_valloss:
            patience=model_patience
            best_valloss = val_loss
            epoch_best=epoch
            torch.save(new_model.state_dict(), newmodelsavepath)
        else:
            patience-=1
            if patience==0:
                print("Early stopping, saving epoch ", epoch_best)
                break
        old_lr=optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
    

    print(f'Epoch {epoch + 1}, Train Loss: {weightedfull_trainloss:.4e}, Val Loss: {val_loss:.4e}, lr: {old_lr:.4e}')

testdata=pd.read_pickle(test_datapath)
test_inputdata, test_outputdata=datamaker(testdata)
ad_test_inputdata, ad_test_outputdata=additional_datamaker(testdata)
test_inputfull=torch.cat((test_inputdata, ad_test_inputdata), dim=1)

test_weight=testdata['weightNorm'].to_numpy()
test_weight_tensor=torch.tensor(test_weight, dtype=torch.float32)

test_dataset = torch.utils.data.TensorDataset(test_inputfull, ad_test_outputdata, test_weight_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

new_model.load_state_dict(torch.load(newmodelsavepath))
outputlist=[]
with torch.no_grad():
    for i, (combined_input, target, sample_weight) in enumerate(test_dataloader):
        # print("combined input shape: ", combined_input.shape)
        # Split the combined input into input for the pre-trained model and additional input
        pretrained_input = combined_input[:, :train_inputdata.shape[1]].to(device)
        additional_input = combined_input[:, train_inputdata.shape[1]:].to(device)
        target = target.to(device)

        # Forward pass
        outputs = new_model(pretrained_input, additional_input)
        outputlist.append(outputs)



outputlist_tens=torch.cat(outputlist, dim=0)
channels_names=['tee', 'tem', 'tmm', 'tte', 'ttm']
channelsd={'tee': 0, 'tem': 1, 'tmm': 2, 'tte': 3, 'ttm': 4}
# channels=channelsd=[0,1, 2, 3,4]
# channels=list(channelsd.keys())
print("outputlist_tens.shape: ", outputlist_tens.shape)
output_np=outputlist_tens.cpu().numpy()

All_channel_dict = {}
pbar = tqdm(channelsd.keys())

data_pd = deepcopy(testdata)
# data_pd['scores'] = output_np
for channel in pbar:
    pbar.set_description(f"Processing {channel}")
    background_data = {}
    signal_data = {}

    channel_indices = np.where(data_pd['channel'] == channelsd[channel])
    # print("channel_indices: ", channel_indices)
    data_dict4={}
    for key in data_pd.columns:
        data_dict4[key]=np.array(data_pd[key])[channel_indices]
    # print(data_dict4['phi_1'])
    # break
    # for key in data_dict_all.co
    #     data_dict4[key]=np.array(data_dict_all[key])[channel_indices].flatten()

    # print(data_dict4['channel'])
    # Get indices of signal and background data
    signal_indices = np.where(data_dict4['signal_label'] == 1)
    background_indices = np.where(data_dict4['signal_label'] == 0)

    for feature in list(data_dict4.keys()):
        # Skip unwanted features
        if feature == 'signal_labels' or feature == 'event_type' or feature =='channel':
            continue

        # Populate signal and background data for the feature
        signal_data[feature] = np.array(data_dict4[feature])[signal_indices].flatten()
        background_data[feature] = np.array(data_dict4[feature])[background_indices].flatten()

    # Add the background and signal data to the all channel dictionary
    All_channel_dict[channel] = {'background': background_data, 'signal': signal_data}


# def get_transferLearning_score_dict(data_dict_dnn, model_class, vars_list,masshyp, scaler=None):
#     """
#     Given a dictionary containing data for various channels and a deep learning model, this method computes the model's
#     scores for each event (particle collision). The method also modifies the original dictionary to 
#     include these scores.

#     Parameters:
#     data_dict_dnn (dict): A dictionary where each key is a channel and its corresponding value is a nested dictionary 
#                           with keys 'background' and 'signal', each associated with a DataFrame containing features for 
#                           each instance in the corresponding category. 
#     model_name (str): The name of the trained model to load for score calculation.
#     path (str): The directory where the trained model is located.
#     vars_list (list): A list of feature names that the model uses for prediction. 
#                       It should include 'signal_label' and 'weightNorm' but they will be removed inside the function.
#     masshyp (float): The mass hypothesis under consideration.
#     scaler (object, optional): An instance of a preprocessing scaler if the data needs to be scaled. 
#                                The default is None, indicating that no scaling is required.

#     Returns:
#     dict: The modified dictionary where each 'background' and 'signal' DataFrame now includes a new 'scores' column 
#           containing the model's scores for each event.
#     """

    
#     dict_copy = deepcopy(data_dict_dnn)
#     vars_list_copy= vars_list.copy()
#     vars_list_copy.remove('signal_label')
#     vars_list_copy.remove('weightNorm')

   
#     model=model_class
#     model.to(device)
#     model.eval()
#     # for channel in  tqdm(dict_copy.keys(), desc='channel', disable=True):
        
#     #     data_background = pd.DataFrame.from_dict(dict_copy[channel]['background'])
#     #     data_signal = pd.DataFrame.from_dict(dict_copy[channel]['signal'])
#     #     data_background['mass_hyp']=masshyp
#     #     data_all_concat = pd.concat([data_background, data_signal])

#     #     # data_all_concat['channel']=channelsd[channel]
#     #     for ch_int in channelsd.values():
#     #         data_all_concat[f'channel_{ch_int}'] = 0
#     #     channel_int = channelsd[channel]
#     #     data_all_concat[f'channel_{channel_int}'] = 1
#     for channel in tqdm(dict_copy.keys(), desc='channel', disable=True):
            
#         data_background = pd.DataFrame.from_dict(dict_copy[channel]['background'])
#         data_signal = pd.DataFrame.from_dict(dict_copy[channel]['signal'])
#         data_background['mass_hyp'] = masshyp
#         data_all_concat = pd.concat([data_background, data_signal])
        
#         # Initialize one-hot encoding columns for channels to zero
#         chname=['channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4']
#         for ch_int in channelsd.values():  # Loop over all possible channel integers
#             data_all_concat[f'channel_{ch_int}'] = 0
        
#         # Set the column corresponding to the current channel to 1
#         channel_int = channelsd[channel]  # Convert channel name to integer
#         data_all_concat[f'channel_{channel_int}'] = 1

#         # print("data_all_concat channel shape: ", data_all_concat['channel'].shape)
#         addinputvars=['charge_1', 'charge_2', 'charge_3']+ chname +['n_tauh', 'mass_hyp']
#         additional=data_all_concat[addinputvars]
#         additional=additional.to_numpy()

#         pretrained=data_all_concat[renamed_old_input_names]
#         pretrained=pretrained.to_numpy()

#         # data_all_concat = data_all_concat[vars_list_copy]
#         data_all_concat = data_all_concat.to_numpy()

#         pretrained_input = torch.tensor(pretrained).float().to(device)
#         additional_input = torch.tensor(additional).float().to(device)

#         with torch.no_grad():
#             output=model(pretrained_input, additional_input)
#         scores=output.cpu().numpy()

#         dict_copy[channel]['background']['scores']=scores[:len(data_background)].flatten()
#         dict_copy[channel]['signal']['scores']=scores[len(data_background):].flatten()

#     return dict_copy


plt.style.use('default')
def plotsignificance(All_channel_dict, xvars ):
    mass_hyp_values = np.unique(All_channel_dict['tte']['signal']['mass_hyp'])
    model_class= TransferCustomKinematicNet(feature_extractor, additional_input_size=10, new_hidden_layers=[128, 128])
    model_class.load_state_dict(torch.load(newmodelsavepath))

    # save_name='transfer'
    # save_path='lolxd'

    input_vars=list(All_channel_dict['tee']['background'].keys())

    fig, ax = plt.subplots(constrained_layout=True)
    # legend_ax = fig.add_axes([0, 0, 1, 0.1])
    for xvar in xvars:
        avg_scores=[]
        avg_uncertainties=[]
        for mass_hyp_value in tqdm(mass_hyp_values, total=len(mass_hyp_values)):
            if xvar=="scores":
                sig_curr, uncer_curr = find_significance2(All_channel_dict, channels_names, xvar, mass_hyp_value, model_class, input_vars, X=0.15, modeltype= 'transfer') 
            else:
                sig_curr, uncer_curr = find_significance(All_channel_dict, channels_names, xvar, mass_hyp_value, X=0.15)
            avg_score = sig_curr.mean(axis=1).values[0]
            avg_uncertainty = uncer_curr.mean(axis=1).values[0]

            avg_scores.append(avg_score)
            avg_uncertainties.append(avg_uncertainty)
            
        ax.plot(mass_hyp_values, avg_scores, label=f' average ({xvar})')
        ax.fill_between(mass_hyp_values, np.subtract(avg_scores, avg_uncertainties), np.add(avg_scores, avg_uncertainties), alpha=0.2)
        print("avg scores for ", xvar, ": ", avg_scores)

    old_model_score = [1.820613890442892, 9.361420462976184, 20.039911049182045, 28.27756725671741, 50.52404764866883, 82.29467697585164, 121.31897199748639, 158.09875583127587, 193.87855479919511, 222.62642118178218, 246.25142681233356, 311.0890536868693, 365.7611525709605, 381.6536122005936, 383.34540371834, 403.83211635956314]
    old_model_uncertainty= [0.24433219285320082, 0.8580808825921137, 1.8202986544403879, 1.8941909500045746, 4.15670847063436, 7.205530510526907, 9.98181579622748, 13.044901735371345, 15.88139382751271, 18.120637465653793, 19.818314869117277, 24.76411286652696, 29.00353115142327, 29.461533545033994, 33.83943027949647, 31.444884323038593]
    ax.plot(mass_hyp_values, old_model_score, label='old model')
    ax.fill_between(mass_hyp_values, np.subtract(old_model_score, old_model_uncertainty), np.add(old_model_score, old_model_uncertainty), alpha=0.2)
    ax.set_xlabel('Mass hypothesis')
    ax.set_ylabel('Average significance')
    ax.set_title('Average significance vs mass hypothesis for transfer learning model')
    ax.grid()
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    # Add legend to the existing ax
    ax.legend(loc='upper left')
    figsavepath= os.path.join(transfermodelsave, 'significance_plot.png')
    fig.savefig(figsavepath)
    plt.show()

datacopy=deepcopy(All_channel_dict)
plotsignificance(datacopy, [ 'Mt_tot', 'scores'])
