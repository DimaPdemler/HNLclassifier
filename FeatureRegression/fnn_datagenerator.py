
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
import pickle
import yaml



parent_path = os.path.dirname(os.getcwd())

#Name of functions to run on the input variables
functions =[
                    deltaphi, deltaphi, deltaphi, 
                    deltaphi, deltaphi, deltaphi,
                    deltaphi3,
                    deltaeta, deltaeta, deltaeta, 
                    deltaeta3,
                    deltaR, deltaR, deltaR, 
                    deltaR3,
                    sum_pt, 
                    transverse_mass, transverse_mass, transverse_mass, 
                    transverse_mass, transverse_mass, transverse_mass,
                    transverse_mass3,
                    invariant_mass, invariant_mass, invariant_mass,
                    invariant_mass,
                    total_transverse_mass, 
                    HNL_CM_angles_with_MET, 
                    W_CM_angles_to_plane, W_CM_angles_to_plane_with_MET,
			        HNL_CM_masses,
                    HNL_CM_masses_with_MET, 
                    W_CM_angles,
                    p4calc,
                    motherpair_vals,
                    Energy_tot
                    ]

# Names of input variables
input_vars = [ 
			       
			        ['1_phi', '2_phi'], ['1_phi', '3_phi'], ['2_phi', '3_phi'], 
			        ['1_phi', 'MET_phi'], ['2_phi', 'MET_phi'], ['3_phi', 'MET_phi'], 
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_eta', '2_eta'], ['1_eta', '3_eta'], ['2_eta', '3_eta'], 
			        ['1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_eta', '2_eta', '1_phi', '2_phi'], ['1_eta', '3_eta', '1_phi', '3_phi'], ['2_eta', '3_eta', '2_phi', '3_phi'], 
			        ['1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        [['1_pt', '2_pt', '3_pt'],['1_phi', '2_phi', '3_phi'],['1_eta', '2_eta', '3_eta'], ['1_mass', '2_mass', '3_mass']], 
			        ['1_pt', '2_pt', '1_phi', '2_phi'], ['1_pt', '3_pt', '1_phi', '3_phi'], ['2_pt', '3_pt', '2_phi', '3_phi'],
			        ['1_pt', 'MET_pt', '1_phi', 'MET_phi'], ['2_pt', 'MET_pt', '2_phi', 'MET_phi'], ['3_pt', 'MET_pt', '3_phi', 'MET_phi'],
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        [['1_pt', '2_pt'],['1_phi', '2_phi'],['1_eta', '2_eta'], ['1_mass', '2_mass']], [['1_pt', '3_pt'],['1_phi', '3_phi'],['1_eta', '3_eta'], ['1_mass', '3_mass']], [['2_pt', '3_pt'],['2_phi', '3_phi'],['2_eta', '3_eta'], ['2_mass', '3_mass']], 	
                    [['1_pt', '2_pt', '3_pt'],['1_phi', '2_phi', '3_phi'],['1_eta', '2_eta', '3_eta'], ['1_mass', '2_mass', '3_mass']], 
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi'],
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        [ '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], 
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
                    ['1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
                    ['1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
                    ['1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass', 'MET_pt']

			        ]

# Names of output variables that are calculated from the input variables and functions
outputvars = [ 
                   
                  'deltaphi_12', 'deltaphi_13', 'deltaphi_23', 
                  'deltaphi_1MET', 'deltaphi_2MET', 'deltaphi_3MET',
                  ['deltaphi_1(23)', 'deltaphi_2(13)', 'deltaphi_3(12)', 
                  'deltaphi_MET(12)', 'deltaphi_MET(13)', 'deltaphi_MET(23)',
                  'deltaphi_1(2MET)', 'deltaphi_1(3MET)', 'deltaphi_2(1MET)', 'deltaphi_2(3MET)', 'deltaphi_3(1MET)', 'deltaphi_3(2MET)'],
                  'deltaeta_12', 'deltaeta_13', 'deltaeta_23', 
                  ['deltaeta_1(23)', 'deltaeta_2(13)', 'deltaeta_3(12)'],
                  'deltaR_12', 'deltaR_13', 'deltaR_23', 
                  ['deltaR_1(23)', 'deltaR_2(13)', 'deltaR_3(12)'],
                  'pt_123',
                  'mt_12', 'mt_13', 'mt_23', 
                  'mt_1MET', 'mt_2MET', 'mt_3MET',
                  ['mt_1(23)', 'mt_2(13)', 'mt_3(12)',
                  'mt_MET(12)', 'mt_MET(13)', 'mt_MET(23)',
                  'mt_1(2MET)', 'mt_1(3MET)', 'mt_2(1MET)', 'mt_2(3MET)', 'mt_3(1MET)', 'mt_3(2MET)'],
                  'mass_12', 'mass_13', 'mass_23',
                  'mass_123',
                  'Mt_tot',
                  ['HNL_CM_angle_with_MET_1', 'HNL_CM_angle_with_MET_2', 'HNL_CM_angle_with_MET_3'], 
                  ['W_CM_angle_to_plane_1', 'W_CM_angle_to_plane_2', 'W_CM_angle_to_plane_3'], ['W_CM_angle_to_plane_with_MET_1', 'W_CM_angle_to_plane_with_MET_2', 'W_CM_angle_to_plane_with_MET_3'],
                  ['HNL_CM_mass_1', 'HNL_CM_mass_2', 'HNL_CM_mass_3'], 
				  ['HNL_CM_mass_with_MET_1', 'HNL_CM_mass_with_MET_2', 'HNL_CM_mass_with_MET_3'], 
                  ['W_CM_angle_12','W_CM_angle_13', 'W_CM_angle_23', 'W_CM_angle_1MET', 'W_CM_angle_2MET', 'W_CM_angle_3MET'],
                  ['px_1', 'py_1', 'pz_1', 'E_1', 'px_2', 'py_2', 'pz_2', 'E_2', 'px_3', 'py_3', 'pz_3', 'E_3'],
                  ['moth_mass_12', 'moth_mass_13', 'moth_mass_23', 'moth_pt_12', 'moth_pt_13', 'moth_pt_23', 'moth_eta_12', 'moth_eta_13', 'moth_eta_23', 'moth_phi_12', 'moth_phi_13', 'moth_phi_23', 'moth_px_12', 'moth_px_13', 'moth_px_23', 'moth_py_12', 'moth_py_13', 'moth_py_23', 'moth_pz_12', 'moth_pz_13', 'moth_pz_23', 'moth_E_12', 'moth_E_13', 'moth_E_23'],
                  'E_tot'
                  ]

#The list of GeV output variables that need to be divided by the total energy
GeV_outputvars = [
    'pt_123', 'mt_12', 'mt_13', 'mt_23', 'mt_1MET', 'mt_2MET', 'mt_3MET', 'mt_1(23)', 
    'mt_2(13)', 'mt_3(12)', 'mt_MET(12)', 'mt_MET(13)', 'mt_MET(23)', 'mt_1(2MET)', 'mt_1(3MET)', 
    'mass_12', 'mass_13', 'mass_23', 'mass_123','Mt_tot',
    'mt_2(1MET)', 'mt_2(3MET)', 'mt_3(1MET)', 'mt_3(2MET)', 'px_1', 'py_1', 'pz_1', 'px_2', 
    'py_2', 'pz_2', 'px_3', 'py_3', 'pz_3', 'moth_pt_12', 'moth_pt_13', 'moth_pt_23', 'moth_px_12',  'moth_px_12', 'moth_px_13', 'moth_px_23', 'moth_py_12', 'moth_py_13', 'moth_py_23', 'moth_pz_12', 'moth_pz_13', 'moth_pz_23', 'moth_E_12', 'moth_E_13', 'moth_E_23',
    'moth_mass_12', 'moth_mass_13', 'moth_mass_23', 'moth_pt_12', 'moth_pt_13', 'moth_pt_23',
    'HNL_CM_mass_1', 'HNL_CM_mass_2', 'HNL_CM_mass_3', 'HNL_CM_mass_with_MET_1','HNL_CM_mass_with_MET_2', 'HNL_CM_mass_with_MET_3',
    'E_1', 'E_2', 'E_3'
]

def generate_pt(lambd, c, batch_size=1):
    """Generate random data from the approximate CDF."""
    p = np.random.uniform(0, 1, batch_size)
    return (c - np.log(1 - p)) / lambd

def flatten_2dlist(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_2dlist(item))
        else:
            flat_list.append(item)
    return flat_list

def call_dict_with_list(dictionary, list_):
    """
    Input :
        -python dictionary
        -python list (potentially multidimensional) of entries
    Output :
        -list with the same structure as the input list, but with the keys replaced by the values of the dictionary at the corresponding keys 
    """
    if type(list_) != list:
        return dictionary[list_]
    else:
        sublist = []
        for el in list_:
            sublist.append(call_dict_with_list(dictionary, el))
        return sublist




flat_output_vars=flatten_2dlist(outputvars)


base_path = os.path.dirname(os.path.dirname(os.getcwd()))
raw_data_pickle_file = os.path.join(base_path, 'saved_files', 'extracted_data', 'TEST10_data_Aug3')
dontremove_outliers=['event', 'genWeight', 'MET_phi', '1_phi', '1_genPartFlav', '2_phi', '2_genPartFlav', '3_phi', '3_genPartFlav', 'charge_1', 'charge_2', 'charge_3', 'pt_1', 'pt_2', 'pt_3', 'pt_MET', 'eta_1', 'eta_2', 'eta_3', 'mass_1', 'mass_2', 'mass_3',
                     ]


def load_thresholds_from_yaml(filename):
    with open(filename, 'r') as f:
        thresholds = yaml.safe_load(f)
    
    lower_thresholds = {k: v['min'] for k, v in thresholds.items()}
    upper_thresholds = {k: v['max'] for k, v in thresholds.items()}
    
    return lower_thresholds, upper_thresholds
thresholdpath= os.path.join(parent_path, 'FeatureRegression', 'threshold_limits.yaml')
lower_thresholds, upper_thresholds = load_thresholds_from_yaml(thresholdpath)
# lower_thresholds, upper_thresholds = load_thresholds_from_yaml('./fnn_FeatureRegression/All_particles_git/thresholds.yaml')


def remove_outliers(data):
    removal_stats = {}
    
    outlier_mask = np.zeros(len(next(iter(data.values()))), dtype=bool)
    for feature_name, values in data.items():
        if (feature_name not in dontremove_outliers) and (feature_name in lower_thresholds):
            lower_value = lower_thresholds[feature_name]
            upper_value = upper_thresholds[feature_name]
            feature_outlier_mask = (np.array(values) < lower_value) | (np.array(values) > upper_value)
            outlier_mask |= feature_outlier_mask  

            removal_stats[feature_name] = np.sum(feature_outlier_mask)



    
    stay_mask = ~outlier_mask
    

    cleaned_data = {k: np.array(v)[stay_mask] for k, v in data.items()}
    

    sorted_removal_stats = dict(sorted(removal_stats.items(), key=lambda item: item[1], reverse=True))

    return cleaned_data, stay_mask






class BatchedFakeParticleDataset_All(Dataset):
    """
    Class to generate "fake" particle physics data for multivariate regression models. Extends the PyTorch Dataset class.
    """

    def __init__(self, batch_size, length):
        """
        Initialize the dataset with batch size and total data length.
        
        Parameters:
        - batch_size (int): Size of each batch.
        - length (int): Total number of data points to generate.
        """
        self.batch_size = batch_size
        self.raw_batch_size = batch_size*2
        self.length = length

        self.input_dim=14
        self.output_dim=len(flat_output_vars)









       


    def __len__(self):
        return self.length // self.batch_size

    def __getitem__(self, batch_idx):
        """
        Generate and return a single batch based on the batch index.
        
        Parameters:
        - batch_idx (int): Index of the batch to generate.
        
        Returns:
        - tuple: A tuple containing input and output tensors.
        """

        raw_input_data = self.generate_input_data()
        output_data, input_data = self.generate_output_data(raw_input_data)

        
        input_data=  {k: v[:self.batch_size] for k, v in input_data.items()}
        output_data= {k: v[:self.batch_size] for k, v in output_data.items()}


        input_data_numpy= np.array([input_data[key] for key in input_data.keys()]).T
        output_data_numpy= np.array([output_data[key] for key in output_data.keys()]).T

        


        input_tensor=torch.tensor(input_data_numpy, dtype=torch.float32)
        output_tensor=torch.tensor(output_data_numpy, dtype=torch.float32)


        
        
        return input_tensor, output_tensor
    
        
    
    def generate_input_data(self):
        """
        Generate a batch of fake input data.
        
        Returns:
        - dict: Dictionary containing fake input data as NumPy arrays.
        """

        eta_low, eta_high = -2.5, 2.5
        mass_low, mass_high = 0, 11
        phi_low, phi_high = -np.pi, np.pi


        #These parameters are from the fit to the pt distribution of the 3 leptons and the MET (using exponential function)
        pt_simplifier_params = [[0.02536545873792836, 0.4934279110259645], [0.019151151336495566, 0.3995434049215345], [0.023038543045718854, 0.31375795899486003], [0.014081741982300087, 0.13542242088536358]]


        input_data = {}
        lepton_specific = ['_eta', '_mass', '_phi', '_pt']
        lepton_specific_names = ['eta', 'mass', 'phi', 'pt']
        MET_specific = [ 'phi', 'pt']
        for i in range(1,4):
            eta= np.random.uniform(eta_low, eta_high, self.raw_batch_size)
            mass = np.random.uniform(mass_low, mass_high, self.raw_batch_size)
            phi = np.random.uniform(phi_low, phi_high, self.raw_batch_size)
            pt = generate_pt(pt_simplifier_params[i-1][0], pt_simplifier_params[i-1][1], batch_size = self.raw_batch_size)



            for j in range(len(lepton_specific)):
                input_data[str(i)+lepton_specific[j]] = eval(lepton_specific_names[j])
        

        phi = np.random.uniform(phi_low, phi_high, self.raw_batch_size)
        pt = generate_pt(pt_simplifier_params[3][0], pt_simplifier_params[3][1], batch_size = self.raw_batch_size)
        METvals=[ phi, pt]
        for feat, val in zip(MET_specific, METvals):
            input_data["MET_"+feat] = val
        
        return input_data
   

    def generate_output_data(self, input_data=None):
        """
        Generate a batch of fake output data based on given input data.
        
        Parameters:
        - input_data (dict): Dictionary containing input data.
        
        Returns:
        - dict: Dictionary containing fake output data. Also returns the "cleaned" input data.
        """

        value_list = []
        for i in range(len(flat_output_vars)):
                value_list.append(np.empty((0,)))

        data = {var: np.empty((0,)) for var in flat_output_vars}


        
        for i,var in enumerate(outputvars):


            
            function_in1=[]
            for var2 in input_vars[i]:
                if isinstance(var2, list):
                    nested_list = [input_data[v] for v in var2]
                    function_in1.append(nested_list)
                else:
                    function_in1.append(input_data[var2])



            outputs = functions[i](*function_in1)



            if type(var) == list:
                for j, subvar in enumerate(var):
                    data[subvar] = np.append(data[subvar], outputs[j])
            else:
                data[var] = np.concatenate((data[var], outputs))


        cleaned_data, stay_mask = remove_outliers(data)

        smaller_input_data=  {k: np.array(v)[stay_mask] for k, v in input_data.items()}
        

        Etot= cleaned_data['E_tot']

        for featname in GeV_outputvars:
            cleaned_data[featname] = cleaned_data[featname]/Etot
            

        return cleaned_data, smaller_input_data


