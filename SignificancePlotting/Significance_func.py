import sys
sys.path.append('../utils/')
sys.path.append('../DNN/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DD_data_extractor_git import Data_extractor_v4, output_vars_v4, flatten_2D_list
from copy import deepcopy
import torch
from tqdm import tqdm
import pickle
from DNN_models import DNN_flexible
from torch import load


# from sklearn.preprocessing import StandardScaler






def process_dataframe(df, flat_features):
    channels_data = {}
    channel_mapping = {'tee': 0, 'tem': 1, 'tmm': 2, 'tte': 3, 'ttm': 4}

    # Rename 'weightOriginal' to 'genWeight' in flat_features
    flat_features = ['genWeight' if feature=='weightOriginal' else feature for feature in flat_features]

    # Iterate over each unique channel name in the dictionary
    for channel_name, channel_number in channel_mapping.items():
        channel_data = {}

        # Filter the dataframe for the current channel
        channel_df = df[df['channel'] == channel_number]

        # Split the channel data into signal and background data
        for signal_label in ['background', 'signal']:
            signal_label_df = channel_df[channel_df['signal_label'] == (0 if signal_label == 'background' else 1)]
            signal_label_data = {}

            # For each feature, convert the data to a numpy array
            for feature in flat_features:
                if feature == 'signal_label' or feature == 'event_type':
                    continue

                if feature == 'genWeight':
                    # Rename 'weightOriginal' to 'genWeight' in the dataframe
                    signal_label_data[feature] = signal_label_df['weightOriginal'].to_numpy()
                else:
                    signal_label_data[feature] = signal_label_df[feature].to_numpy()

            channel_data[signal_label] = signal_label_data

        channels_data[channel_name] = channel_data

    return channels_data

def process_channels(channels, flat_features, path, relative_path, data):
    """
    This function processes a list of channels and returns a dictionary. 
    For each channel, it creates a sub-dictionary that contains two dictionaries: 
    'background' and 'signal'. Each of these dictionaries contains 40 features as keys, 
    and corresponding numpy arrays as values.
    
    The resulting dictionary structure looks like:
    
    channels_data = {
        'channel1': {
            'background': {feature1: numpy array, feature2: numpy array, ..., feature40: numpy array},
            'signal': {feature1: numpy array, feature2: numpy array, ..., feature40: numpy array},
        },
        'channel2': {
            'background': {feature1: numpy array, feature2: numpy array, ..., feature40: numpy array},
            'signal': {feature1: numpy array, feature2: numpy array, ..., feature40: numpy array},
        },
        ...
    }
    
    Args:
    channels (list): List of channels to be processed.
    flat_features (list): List of features to be extracted.
    path (str): Path to the data directory.
    relative_path (str): Relative path from the data directory to the channel data.
    data (any): Additional data required for the extractor.
    
    Returns:
    dict: A dictionary with the described structure.
    """
    All_channel_dict = {}
    pbar = tqdm(channels)

    for channel in pbar:
        pbar.set_description(f"Processing {channel}")
        # print(channel)

        # Initialize data extractor
        extractor = Data_extractor_v4(channel)

        # Extract data
        data_dict4 = extractor(path+channel+relative_path, data=data)

        # Initialize dictionaries to store background and signal data
        background_data = {}
        signal_data = {}

        # Get indices of signal and background data
        signal_indices = np.where(data_dict4['signal_label'] == 1)
        background_indices = np.where(data_dict4['signal_label'] == 0)

        for feature in flat_features:
            # Skip unwanted features
            if feature == 'signal_labels' or feature == 'event_type':
                continue

            # Populate signal and background data for the feature
            signal_data[feature] = np.array(data_dict4[feature])[signal_indices].flatten()
            background_data[feature] = np.array(data_dict4[feature])[background_indices].flatten()

        # Add the background and signal data to the all channel dictionary
        All_channel_dict[channel] = {'background': background_data, 'signal': signal_data}

    return All_channel_dict

def binmaker(channel_specific_dict, numbins_start, xvariable, masshyp, X=0.3,  plot=False, Channelname= "not specified"):
    """
    Function to create bins for the data and plot histograms with error bars. It makes sure the signal height remains constant in each bin. 
    This function first calculates the optimal number of bins and then creates the bins.
    It also computes the errors for each bin and visualizes the histograms with error bars if plot is set to True.

    Parameters:
    channel_specific_dict (dict): Dictionary containing channel specific data for signal and background.
    numbins_start (int): Initial number of bins.
    xvariable (str): Name of the variable to be plotted on x-axis.
    masshyp (float): Mass hypothesis value.
    X (float, optional): Maximum allowable fractional error. Default is 0.3.
    plot (bool, optional): If True, the function will plot histograms. Default is False.
    Channelname (str, optional): Name of the channel. Default is "not specified".
    
    Returns:
    tuple: A tuple containing bin indices, signal height, background height, signal error and background error.
    """

    features=list(channel_specific_dict['signal'].keys())
    signal_dict_ind= np.where(channel_specific_dict['signal']['mass_hyp'] == masshyp)
    signal_dict = {}
    for feature in features:
        signal_dict[feature] = np.array(channel_specific_dict['signal'][feature])[signal_dict_ind].flatten()
    sorted_indices = np.argsort(signal_dict[xvariable])

    currbins=numbins_start
    while currbins>2:
        total_genweight = np.sum(signal_dict['genWeight']- 1e-10)
        target_genweight=total_genweight/currbins
        numevents=len(signal_dict['genWeight'])

        bin_indices = [signal_dict[xvariable][sorted_indices[0]]]
  
        bin_start_index = 0
        cumulative_genWeight = 0.0
        j=0
        for i in range(currbins):
            for ind in sorted_indices[bin_start_index:]:
                cumulative_genWeight += signal_dict['genWeight'][ind]
                j+=1
                if cumulative_genWeight > target_genweight:
                    bin_indices.append(signal_dict[xvariable][ind]- 1e-10)
                    bin_start_index = j
                    cumulative_genWeight = 0.0
                    break
        signal_height, signal_bins = np.histogram(signal_dict[xvariable], bins=bin_indices, weights=signal_dict['genWeight'])
        background_height, background_bins= np.histogram(channel_specific_dict['background'][xvariable], bins=bin_indices, weights=channel_specific_dict['background']['genWeight'])
        


         # Compute bin widths and centers
        bin_widths = np.diff(signal_bins)
        bin_centers = (signal_bins[:-1] + signal_bins[1:]) / 2
        
        # Calculate the error for each bin
        signal_bins_id=np.digitize(signal_dict[xvariable], signal_bins)
        background_bins_id=np.digitize(channel_specific_dict['background'][xvariable], signal_bins)

        signal_weights_squared = signal_dict['genWeight']**2
        signal_error = np.array([np.sqrt(np.sum(signal_weights_squared[signal_bins_id == i])) for i in range(1, len(signal_bins))])
        
        background_weights_squared = channel_specific_dict['background']['genWeight']**2
        background_error = np.array([np.sqrt(np.sum(background_weights_squared[background_bins_id == i])) for i in range(1, len(background_bins))])

        frac_error_background = np.abs(background_error/background_height)
        frac_error_signal = np.abs(signal_error/signal_height)
        
        if error_boolean(frac_error_background, frac_error_signal, signal_height, background_height, currbins, X=X) and currbins>2:
            break
        else:
            currbins-=1

    forcestop=0
    while error_boolean(frac_error_background, frac_error_signal, signal_height, background_height, currbins, X=X) == False:
        bin_indices[1]=bin_indices[1]*0.99
        signal_height, signal_bins = np.histogram(signal_dict[xvariable], bins=bin_indices, weights=signal_dict['genWeight'])
        background_height, background_bins= np.histogram(channel_specific_dict['background'][xvariable], bins=bin_indices, weights=channel_specific_dict['background']['genWeight'])

        bin_widths = np.diff(signal_bins)
        bin_centers = (signal_bins[:-1] + signal_bins[1:]) / 2
        
        # Calculate the error for each bin
        signal_bins_id=np.digitize(signal_dict[xvariable], signal_bins)
        background_bins_id=np.digitize(channel_specific_dict['background'][xvariable], signal_bins)
        
        signal_weights_squared = signal_dict['genWeight']**2
        signal_error = np.array([np.sqrt(np.sum(signal_weights_squared[signal_bins_id == i])) for i in range(1, len(signal_bins))])
        
        background_weights_squared = channel_specific_dict['background']['genWeight']**2
        background_error = np.array([np.sqrt(np.sum(background_weights_squared[background_bins_id == i])) for i in range(1, len(background_bins))])

        frac_error_background = np.abs(background_error/background_height)
        frac_error_signal = np.abs(signal_error/signal_height)

        forcestop+=1
        if forcestop>100:
            print('forcestop')
            break
        
    # Plot the histograms with error bars
    if plot:
        plt.figure()
        ax = plt.gca()

        ax.bar(bin_centers, signal_height, width=bin_widths, align='center', color='blue', edgecolor='blue', alpha=0.4, label='Signal')
        ax.errorbar(bin_centers, signal_height, yerr=signal_error, fmt='o', color='black', label='Signal error')

        ax.bar(bin_centers, background_height, width=bin_widths, align='center', color='red', edgecolor='red', alpha=0.4, label='Background')
        ax.errorbar(bin_centers, background_height, yerr=background_error, fmt='o', color='green', label='Background error')

        ax.set_yscale('log')
        ax.set_ylabel('Weighted number of events')
        ax.set_xlabel(xvariable)
        ax.set_title('Mass hypothesis: {}'.format(masshyp) + ' GeV, channel: {}'.format(Channelname) + ', numbins: {}'.format(currbins))
        ax.legend()
        # plt.savefig(f"plot_{masshyp}.png")
        plt.show()
    
    return bin_indices, signal_height, background_height, signal_error, background_error


def error_boolean(frac_error_background, frac_error_signal, signal_height, background_height, currbins, X=0.3):
    """
    Function to check if the fractional errors and signal and background heights meet the specified criteria.

    Parameters:
    frac_error_background (np.ndarray): Array of fractional errors for the background.
    frac_error_signal (np.ndarray): Array of fractional errors for the signal.
    signal_height (np.ndarray): Array of signal heights.
    background_height (np.ndarray): Array of background heights.
    currbins (int): Current number of bins.
    X (float, optional): Maximum allowable fractional error. Default is 0.3.

    Returns:
    bool: True if all conditions are met, False otherwise.
    """
    return frac_error_background.max() < X and frac_error_signal.max() < X and all(signal_height>0) and all(background_height>0)


def binmaker_rightleft(channel_specific_dict, xvariable, masshyp, X=0.3,  plot=False, Channelname = 'not specified'):
    """
    A function to bin data from signal and background datasets. It starts at the rightmost xvalue and increases bin width until fractional uncertainty (X)
     is satisfied.
    Parameters:
    channel_specific_dict (dict): The dictionary containing signal and background data. 
                                  This dictionary should have two keys: 'signal' and 'background', 
                                  each containing another dictionary with the features as keys.
    xvariable (str): The variable on which to bin the data.
    masshyp (float): The mass hypothesis used for the binning process.
    X (float, optional): The fractional uncertainty threshold. Default is 0.3.
    plot (bool, optional): If True, the function will plot a histogram with the signal and background data. Default is False.
    Channelname (str, optional): A string representing the channel name, which will be displayed in the plot's title. Default is 'not specified'.

    Returns:
    bin_indices (list): The list of bin edges used.
    signal_height (list): The sum ofweights of the signal data within each bin.
    background_height (list): The sum of weights of the background data within each bin.
    signal_error (list): The square root of the sum of the squares of the weights of the signal data within each bin.
    background_error (list): The square root of the sum of the squares of the weights of the background data within each bin.

    Note:
    The function prints and plots results based on the `plot` parameter. If `plot` is True, it will display a bar plot of the signal 
    and background distributions with error bars, and the resulting plot is saved as an image file.
    """

    


    signal_weights_squared_list = []
    background_weights_squared_list = []

    features=list(channel_specific_dict['signal'].keys())
    signal_dict_ind= np.where(channel_specific_dict['signal']['mass_hyp'] == masshyp)
    signal_dict = {}
    for feature in features:
        signal_dict[feature] = np.array(channel_specific_dict['signal'][feature])[signal_dict_ind].flatten()
    sorted_indices = np.argsort(signal_dict[xvariable])[::-1]

    background_sorted_indices=np.argsort(channel_specific_dict['background'][xvariable])[::-1]
    bin_indices = [signal_dict[xvariable][sorted_indices[0]]]
    j=0
    background_j=0

    prevbinheight_signal=0
    
    currbinheight_signal=0
    currbinheight_background=0
    curr_signalweights_squared=0
    curr_backgroundweights_squared=0

    bin_start_index=0   
    for j, ind in tqdm(enumerate(sorted_indices), desc='binmaker_rightleft', disable=True):
        
        run=False
        
        currbinheight_signal +=signal_dict['genWeight'][ind]
        curr_signalweights_squared +=signal_dict['genWeight'][ind]**2
        for b_jval in range(background_j, len(background_sorted_indices)):
            if channel_specific_dict['background'][xvariable][background_sorted_indices[b_jval]] < signal_dict[xvariable][ind]:
                background_j=b_jval
                break
            else:
                curr_backgroundweights_squared +=channel_specific_dict['background']['genWeight'][background_sorted_indices[b_jval]]**2
                currbinheight_background +=channel_specific_dict['background']['genWeight'][background_sorted_indices[b_jval]]
        

        frac_error_signal= np.sqrt(curr_signalweights_squared)/currbinheight_signal
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_error_background = np.sqrt(curr_backgroundweights_squared)/currbinheight_background

        if frac_error_signal < X and frac_error_background < X and currbinheight_signal > prevbinheight_signal and currbinheight_background>0:
            
            bin_indices.append(signal_dict[xvariable][ind])
            
            prevbinheight_signal=currbinheight_signal

            signal_weights_squared_list.append(curr_signalweights_squared)
            background_weights_squared_list.append(curr_backgroundweights_squared)

            currbinheight_signal=0
            currbinheight_background=0
            curr_signalweights_squared=0
            curr_backgroundweights_squared=0
            run=True
    bin_indices.append(signal_dict[xvariable][sorted_indices[-1]])     
    bin_indices=bin_indices[::-1]
    signalwsquared=0
    backgroundwsquared=0
    for indice in sorted_indices[::-1]:
        if signal_dict[xvariable][indice] < bin_indices[1]:
            signalwsquared += signal_dict['genWeight'][indice]**2
            backgroundwsquared += channel_specific_dict['background']['genWeight'][indice]**2
        else:
            break
    signal_weights_squared_list.append(signalwsquared)
    background_weights_squared_list.append(backgroundwsquared)
    
    signal_height, signal_bins = np.histogram(signal_dict[xvariable], bins=bin_indices, weights=signal_dict['genWeight'])
    background_height, background_bins= np.histogram(channel_specific_dict['background'][xvariable], bins=bin_indices, weights=channel_specific_dict['background']['genWeight'])

    bin_widths = np.diff(signal_bins)
    bin_centers = (signal_bins[:-1] + signal_bins[1:]) / 2

    if plot:

        plt.figure()
        ax = plt.gca()

        ax.bar(bin_centers, signal_height, width=bin_widths, align='center', color='blue', edgecolor='blue', alpha=0.4, label='Signal')
        ax.errorbar(bin_centers, signal_height, fmt='o', color='black', label='Signal error')

        ax.bar(bin_centers, background_height, width=bin_widths, align='center', color='red', edgecolor='red', alpha=0.4, label='Background')
        ax.errorbar(bin_centers, background_height, fmt='o', color='green', label='Background error')
        
        ax.set_yscale('log')
        ax.set_ylabel('Weighted number of events')
        ax.set_xlabel(xvariable)
        ax.set_title('Mass hypothesis: {}, numbins: {}, channel: {}'.format(masshyp, len(bin_indices)-1, Channelname))

        ax.legend()
        # plt.savefig(f"plot_{masshyp}.png")
        plt.show()
    
    signal_error=np.sqrt(np.array(signal_weights_squared_list))
    background_error=np.sqrt(np.array(background_weights_squared_list))
    return bin_indices, signal_height, background_height, np.flip(signal_error), np.flip(background_error)

def get_dnn_score_dict_torch_simple(data_dict_dnn, model_class, vars_list,masshyp, scaler=None):
    """
    Given a dictionary containing data for various channels and a deep learning model, this method computes the model's
    scores for each event (particle collision). The method also modifies the original dictionary to 
    include these scores.

    Parameters:
    data_dict_dnn (dict): A dictionary where each key is a channel and its corresponding value is a nested dictionary 
                          with keys 'background' and 'signal', each associated with a DataFrame containing features for 
                          each instance in the corresponding category. 
    model_name (str): The name of the trained model to load for score calculation.
    path (str): The directory where the trained model is located.
    vars_list (list): A list of feature names that the model uses for prediction. 
                      It should include 'signal_label' and 'weightNorm' but they will be removed inside the function.
    masshyp (float): The mass hypothesis under consideration.
    scaler (object, optional): An instance of a preprocessing scaler if the data needs to be scaled. 
                               The default is None, indicating that no scaling is required.

    Returns:
    dict: The modified dictionary where each 'background' and 'signal' DataFrame now includes a new 'scores' column 
          containing the model's scores for each event.
    """

    
    dict_copy = deepcopy(data_dict_dnn)
    vars_list_copy= vars_list.copy()
    vars_list_copy.remove('signal_label')
    vars_list_copy.remove('weightNorm')

   
    model=model_class
    model.eval()

    for channel in  tqdm(dict_copy.keys(), desc='channel', disable=True):
        
        data_background = pd.DataFrame.from_dict(dict_copy[channel]['background'])
        data_signal = pd.DataFrame.from_dict(dict_copy[channel]['signal'])
        data_background['mass_hyp']=masshyp
        data_all_concat = pd.concat([data_background, data_signal])
        data_all_concat = data_all_concat[vars_list_copy]

        data_all_concat = data_all_concat.to_numpy()
        if scaler is not None:
            data_all_concat=scaler.transform(data_all_concat)

        data_all_concat_tensor = torch.tensor(data_all_concat).float()
        
        with torch.no_grad():
            output=model(data_all_concat_tensor)
        scores=output.numpy()

        dict_copy[channel]['background']['scores']=scores[:len(data_background)].flatten()
        dict_copy[channel]['signal']['scores']=scores[len(data_background):].flatten()

    return dict_copy


def bin_uncertainty2(signal_height, background_height, sig_std, back_std, significance):
    """
    Calculates the uncertainty in the significance calculation. 

    Parameters:
    signal_height (float): The signal height of the bin.
    background_height (float): The background height of the bin.
    sig_std (float): The standard deviation of the signal.
    back_std (float): The standard deviation of the background.
    significance (float): The calculated significance.

    Returns:
    float: The calculated uncertainty in the significance.
    """    
    numbins=len(signal_height)
    sumdfds=0
    sumdfdb=0

    for i in range(numbins):
        
        dfds=signal_height[i] / (background_height[i] * significance + 1e-8)
        sumdfds+=(dfds * sig_std[i])**2

        dfdb=-0.5*(signal_height[i]**2) / (background_height[i]**2 * significance + 1e-8)
        sumdfdb+=(dfdb * back_std[i])**2
    uncertainty=np.sqrt(sumdfds+sumdfdb)
    return uncertainty
    

def find_significance(data, channels, xvariable, masshyp, X=0.2, plot=False, binmakertype='binmaker_rightleft'):
    """
    For each channel, calculates the significance and its uncertainty for a specific mass hypothesis. 

    Parameters:
    data (dict): A dictionary where each key is a channel and its corresponding value is a nested 
                 dictionary with keys 'background' and 'signal', each associated with a DataFrame containing 
                 features for each instance in the corresponding category.
    channels (list): A list of channels to calculate significance for.
    xvariable (str): A string denoting the variable to be plotted on the x-axis.
    masshyp (float): The mass hypothesis under consideration.
    X (float, optional): A scalar value used in the calculation of significance. Default is 0.2.
    plot (bool, optional): A boolean value indicating whether to plot the results. Default is False.
    binmakertype (str, optional): A string specifying the method used to create bins for calculating 
                                  significance. Can be either 'binmaker_rightleft' or 'binmaker_constsignal'. 
                                  Default is 'binmaker_rightleft'.

    Returns:
    significance_pd (DataFrame): A DataFrame with calculated significances for each channel.
    uncertainty_pd (DataFrame): A DataFrame with calculated uncertainties for each channel.
    """
    significance_pd=pd.DataFrame(columns=channels)
    uncertainty_pd=pd.DataFrame(columns=channels)
    for channel in channels:
        if binmakertype=='binmaker_rightleft':
            bin_indices, signal_height, background_height, signal_error, background_error =binmaker_rightleft(data[channel], xvariable, masshyp, X=X,  plot=False, Channelname =channel)
        elif binmakertype == 'binmaker_constsignal':
            bin_indices, signal_height, background_height, signal_error, background_error =binmaker(data[channel], 30, xvariable, masshyp, X=X,  plot=False)
        else:
            raise ValueError("binmakertype not recognized: " + binmakertype)
        
        s1 = signal_height / np.sqrt(background_height + 1e-8)  # Adding a small constant

        significance2=np.sum(s1**2)
        significance_pd.at[0, channel]=np.sqrt(significance2)

        uncertainty=bin_uncertainty2(signal_height, background_height, signal_error, background_error, np.sqrt(significance2))
        uncertainty_pd.at[0, channel]=uncertainty



    return significance_pd, uncertainty_pd

def find_significance2(data, channels, xvariable, masshyp, model_name, model_class, path, vars_list, X=0.2, plot=False, scaler=None, binmakertype='binmaker_rightleft'):
    """
    Similar to find_significance, but uses a deep learning model to calculate scores, 
    and subsequently the significance and its uncertainty, for a specific mass hypothesis.

    Parameters:
    data (dict): A dictionary where each key is a channel and its corresponding value is a nested 
                 dictionary with keys 'background' and 'signal', each associated with a DataFrame containing 
                 features for each instance in the corresponding category.
    channels (list): A list of channels to calculate significance for.
    xvariable (str): A string denoting the variable to be plotted on the x-axis.
    masshyp (float): The mass hypothesis under consideration.
    model_name (str): The name of the trained model to load for score calculation.
    path (str): The directory where the trained model is located.
    vars_list (list): A list of feature names that the model uses for prediction. 
    X (float, optional): A scalar value used in the calculation of significance. Default is 0.2.
    plot (bool, optional): A boolean value indicating whether to plot the results. Default is False.
    scaler (object, optional): An instance of a preprocessing scaler if the data needs to be scaled. 
                               Default is None, indicating that no scaling is required.
    binmakertype (str, optional): A string specifying the method used to create bins for calculating 
                                  significance. Can be either 'binmaker_rightleft' or 'binmaker_constsignal'. 
                                  Default is 'binmaker_rightleft'.

    Returns:
    significance_pd (DataFrame): A DataFrame with calculated significances for each channel.
    uncertainty_pd (DataFrame): A DataFrame with calculated uncertainties for each channel.
    """
    significance_pd=pd.DataFrame(columns=channels)
    uncertainty_pd=pd.DataFrame(columns=channels)

    dnn_score_dict=get_dnn_score_dict_torch_simple(data,model_class, vars_list,masshyp, scaler=scaler)
    
    for channel in tqdm(channels, desc='channel find_significance2, masshyp:' + str(masshyp), disable=True):
        try:
            if binmakertype=='binmaker_rightleft':
                bin_indices, signal_height, background_height, signal_error, background_error=binmaker_rightleft(dnn_score_dict[channel], xvariable, masshyp, X=X,  plot=False, Channelname =channel)
            elif binmakertype == 'binmaker_constsignal':
                bin_indices, signal_height, background_height, signal_error, background_error=binmaker(dnn_score_dict[channel], 30, xvariable, masshyp, X=X,  plot=False)
            else:
                raise ValueError("binmaker type not recognized: " + binmakertype)
        except ValueError as ve:
            print(f"Skipping binmaker for channel {channel} due to error: {str(ve)}")
            continue
        
        s1 = signal_height / np.sqrt(background_height + 1e-8)  # Adding a small constant

        significance2=np.sum(s1**2)
        significance_pd.at[0, channel]=np.sqrt(significance2)  # Set value at specific cell

        uncertainty=bin_uncertainty2(signal_height, background_height, signal_error, background_error, np.sqrt(significance2))
        uncertainty_pd.at[0, channel]=uncertainty


    return significance_pd, uncertainty_pd


def plot_average_significance_withpd(data, xvariables, model_info_df, binmakers, X=0.3, hide_errorbars=False, print_AUC=False):
    """
    This function calculates the average significance for different mass hypotheses for various machine learning models, 
    plots the average significance for each model, and prints the AUC if desired. The function is flexible and allows for 
    the use of different bin making methods and feature variables.

    Parameters:
    data (dict): A dictionary where each key is a channel and its corresponding value is a nested dictionary with keys 
                'background' and 'signal', each associated with a DataFrame containing features for each instance in 
                the corresponding category.
    xvariables (list): A list of feature names to be used in the calculation of significance. 
    model_info_df (DataFrame): A DataFrame containing information about the models to be used. Each row should contain 
                            the following columns: 'save_path' (directory where the trained model is saved), 'save_name' 
                            (name of the model), 'input_variables' (features used by the model), 'hidden_layers' 
                            (the architecture of the model's hidden layers), and 'scaler_path' (path to the model's 
                            preprocessing scaler).
    binmakers (list): A list of strings specifying the methods used to create bins for calculating significance. 
    X (float, optional): A scalar value used in the calculation of significance. Default is 0.3.
    hide_errorbars (bool, optional): If True, the function will not plot the error bars. Default is False.
    print_AUC (bool, optional): If True, the function will print the Area Under Curve (AUC) for each model. Default is False.

    Returns:
    None

    Note:
    The function plots the average significance and its uncertainty against the mass hypotheses for each model. The plots are 
    displayed in a single figure with log scales on both axes. A separate legend is created below the plot.

    Usage:
    The function should be called with the data dictionary, a list of feature names to be used as 'xvariables', a DataFrame 
    containing model information as 'model_info_df', and a list of bin making methods as 'binmakers'. The scalar value 'X', 
    'hide_errorbars', and 'print_AUC' can be specified if needed. If 'print_AUC' is True, the function will print the AUC for 
    each model after plotting the significance. If 'hide_errorbars' is True, the error bars will not be displayed in the plot.
    """

    mtot_run=False
    auc_dict = {}
    channels = data.keys()

    mass_hyp_values = np.unique(data['tte']['signal']['mass_hyp'])

    # Create separate axes for the legend
    fig, ax = plt.subplots()
    legend_ax = fig.add_axes([0, 0, 1, 0.1])

    # Loop over the models
    for _, row in tqdm(model_info_df.iterrows(), desc='model', disable=True):
        save_path, save_name, input_vars, hidden_layers, scaler_path = row['save_path'], row['save_name'], row['input_variables'], row['hidden_layers'], row['scaler_path']

        # Load model
        model_class = DNN_flexible(input_vars, hidden_layers)
        model_class.load_state_dict(load(save_path + save_name + '.pt'))

        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Loop over the binmakers
        for binmakertype in binmakers:
            # Loop over the xvariables
            for xvariable in xvariables:
                if xvariable == 'Mt_tot':
                    if not mtot_run:
                        mtot_run=True
                    else: 
                        continue
                avg_scores = []
                avg_uncertainties = []

                pbar = tqdm(mass_hyp_values, disable=False)
                for mass_hyp_value in pbar:
                    pbar.set_description(f'Mass_hyps: {mass_hyp_value}, xvariable: {xvariable}, binmaker: {binmakertype}, model: {save_name}')

                    if xvariable == 'scores':
                        sig_curr, uncer_curr = find_significance2(data, channels, xvariable, mass_hyp_value, save_name, model_class, save_path, input_vars, X=X, plot=False, scaler=scaler, binmakertype=binmakertype)
                    else:
                        sig_curr, uncer_curr = find_significance(data, channels, xvariable, mass_hyp_value, X=X, plot=False, binmakertype=binmakertype)

                    avg_score = sig_curr.mean(axis=1).values[0]
                    avg_uncertainty = uncer_curr.mean(axis=1).values[0]

                    avg_scores.append(avg_score)
                    avg_uncertainties.append(avg_uncertainty)

                if print_AUC:
                    # Calculate the AUC for the current model
                    auc = np.trapz(avg_scores, x=mass_hyp_values)
                    auc_dict[save_name] = auc  # Save the AUC

                # Plot the average significance and its uncertainty
                ax.plot(mass_hyp_values, avg_scores, label=f'{save_name}, average ({xvariable}, {binmakertype})')
                if not hide_errorbars:
                    ax.fill_between(mass_hyp_values, np.subtract(avg_scores, avg_uncertainties), np.add(avg_scores, avg_uncertainties), alpha=0.2)
                

    if print_AUC:
        print("Ordered list of models by AUC:")
        for model, auc in sorted(auc_dict.items(), key=lambda item: item[1], reverse=True):
            print(f"Model: {model}, AUC: {auc}")


    ax.set_xlabel('Mass hypothesis')
    ax.set_ylabel('Average significance')
    ax.set_title('Average significance vs mass hypothesis for different xvariables and binmakers')
    ax.set_xscale('log')
    ax.set_yscale('log')

    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='center')

    # Hide the axes of the legend
    legend_ax.axis('off')

    # Adjust subplot parameters and then call tight_layout
    plt.subplots_adjust(bottom=-1)  # Adjust this value to suit your needs
    plt.tight_layout()

    plt.show()