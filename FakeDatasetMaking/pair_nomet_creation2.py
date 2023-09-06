import numpy as np
import sys
import os

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import vector
import torch
import pickle
from torch.utils.data import Dataset, DataLoader, TensorDataset, Sampler
sys.path.append('../utils/')
from DD_data_extractor_git import Data_generator, outlier_normalization, generate_random_data, exponential_cdf, outlier_normalization, remove_outliers,flatten_2D_list,bucketize, call_dict_with_list

from DDkinematic_final import *


# Same as previous but trains on each pair seperately


output_vars_pair_nomet = [ 
                  'pt_1', 'pt_2', 'pt_3', 'pt_MET', 
                  'eta_1', 'eta_2', 'eta_3',
                  'mass_1', 'mass_2', 'mass_3',
                   
                  'deltaphi_12', 'deltaphi_13', 'deltaphi_23', 
                  
                  'deltaeta_12', 'deltaeta_13', 'deltaeta_23', 

                  'deltaR_12', 'deltaR_13', 'deltaR_23', 
                  'mt_12', 'mt_13', 'mt_23', 
                  
                  'mass_12', 'mass_13', 'mass_23']

input_data_names_ordered = [
    ['MET_phi', 'pt_MET'], 
    ['1_phi', 'pt_1', 'eta_1', 'mass_1'], 
    ['2_phi', 'pt_2', 'eta_2', 'mass_2'], 
    ['3_phi', 'pt_3', 'eta_3', 'mass_3']
]
input_data_particle_order = ['MET', '1', '2', '3']

pair_order = ["MET_1", "MET_2", "MET_3", "1_2", "1_3", "2_3"]
# used_labels2 = [
#     ['deltaphi_1MET', 'mt_1MET'], 
#     ['deltaphi_2MET', 'mt_2MET'], 
#     ['deltaphi_3MET', 'mt_3MET'], 
#     ['deltaphi_12', 'deltaeta_12'], 
#     ['deltaphi_13', 'deltaeta_13'], 
#     [ 'deltaphi_23', 'deltaeta_23']
# ]
used_labels2 = [
    ['deltaphi_1MET', 'mt_1MET'], 
    ['deltaphi_2MET', 'mt_2MET'], 
    ['deltaphi_3MET', 'mt_3MET'], 
    ['deltaphi_12', 'deltaeta_12', 'deltaR_12', 'mt_12', 'norm_mt_12'], 
    ['deltaphi_13', 'deltaeta_13', 'deltaR_13', 'mt_13', 'norm_mt_13'], 
    ['deltaphi_23', 'deltaeta_23', 'deltaR_23', 'mt_23', 'norm_mt_23']
]

features_toadd=[ 'mass', 'pt', 'eta' , 'phi',  'px', 'py', 'pz', 'energy']

class KinematicDataset(Dataset):
    def __init__(self,num_events,seed=0, used_labels=used_labels2, features_toadd=features_toadd, batch_size=320):
        self.num_events = num_events
        self.current_data = None
        self.current_labels = None
        self.seed = seed
        self.output_vars = output_vars_pair_nomet
        self.functions =[None, None, None, None,     # pts
                    None, None, None,           # etas
                    None, None, None,           # masses

                    deltaphi, deltaphi, deltaphi,
                    
                    deltaeta, deltaeta, deltaeta,

                    deltaR, deltaR, deltaR, 

                    transverse_mass, transverse_mass, transverse_mass, 
                    
                    invariant_mass, invariant_mass, invariant_mass
                    ]
        self.raw_vars_general = [ 'MET_pt', 'MET_phi']
       
        self.lepton_input_ordered = input_data_names_ordered[1:]
        self.lepton_output_ordered = used_labels[3:]
        self.lepton_pair_order = pair_order[3:]
        self.lepton_particle_order = input_data_particle_order[1:]
        self.lepton_specific = ['_eta', '_mass', '_phi', '_pt', '_charge', '_genPartFlav']
        self.features_toadd=features_toadd
        self.batch_size=batch_size
        # raw_vars_lepton1 = lepton_specific
        # raw_vars_lepton2 = lepton_specific
        # raw_vars_lepton3 = lepton_specific
        self.input_vars = [['1_pt'], ['2_pt'], ['3_pt'], ['MET_pt'],
			        ['1_eta'], ['2_eta'], ['3_eta'], 
			        ['1_mass'], ['2_mass'], ['3_mass'], 
			        ['1_phi', '2_phi'], ['1_phi', '3_phi'], ['2_phi', '3_phi'], 
			         
			        
                    ['1_eta', '2_eta'], ['1_eta', '3_eta'], ['2_eta', '3_eta'], 

			        ['1_eta', '2_eta', '1_phi', '2_phi'], ['1_eta', '3_eta', '1_phi', '3_phi'], ['2_eta', '3_eta', '2_phi', '3_phi'], 
                     
			        ['1_pt', '2_pt', '1_phi', '2_phi'], ['1_pt', '3_pt', '1_phi', '3_phi'], ['2_pt', '3_pt', '2_phi', '3_phi'],
			        
			        [['1_pt', '2_pt'],['1_phi', '2_phi'],['1_eta', '2_eta'], ['1_mass', '2_mass']], [['1_pt', '3_pt'],['1_phi', '3_phi'],['1_eta', '3_eta'], ['1_mass', '3_mass']], [['2_pt', '3_pt'],['2_phi', '3_phi'],['2_eta', '3_eta'], ['2_mass', '3_mass']]	
                    ]
        base_path = os.path.dirname(os.getcwd())
        raw_data_pickle_file = os.path.join(base_path, 'saved_files', 'extracted_data', 'TEST10_data_Aug3')
        self.lower_percentiles, self.upper_percentiles = compute_percentiles_from_pickle(raw_data_pickle_file)
        self.generate_data()

    def __len__(self):
        return self.num_events
    
    def __getitem__(self, index):
        # self.generate_data()
        x_tensor = torch.from_numpy(self.current_data[index % len(self.current_data)]).float()
        y_tensor = torch.from_numpy(self.current_labels[index % len(self.current_labels)]).float()
        return x_tensor, y_tensor

        
    def set_seed(self, seed):
        self.seed = seed

    def iterate_seed(self):
        self.seed += 15
        self.generate_data()



    def generate_data(self, normalize=True):
        # print(f"Generating data for seed: {self.seed}")
        data_dictlong = self.generate_fake_data2(int(self.num_events*2))

        old_keys = [f"{i}_{var}" for i in range(1, 4) for var in ['pt', 'eta', 'mass']]
        new_keys = [f"{var}_{i}" for i in range(1, 4) for var in ['pt', 'eta', 'mass']]

        for old_key, new_key in zip(old_keys, new_keys):
            if old_key in data_dictlong:
                data_dictlong[new_key] = data_dictlong[old_key]
                del data_dictlong[old_key]  # Remove old key-value pair from the dictionary
            
        if normalize:
            data_dictlong = self.add_norm_features(data_dictlong)

        data_dictlong = remove_outliers(data_dictlong, self.lower_percentiles, self.upper_percentiles)
        # print("data dict len", len(data_dictlong['pt_1']))
        data_dict = {key: value[:self.num_events] for key, value in data_dictlong.items()}
        # print("data dict len", len(data_dict['pt_1']))
        # return data_dict
        
        l_input, l_output=self.convert_to_array(data_dict)
        l_output2= np.concatenate((l_output, self.add_vectorfeats(data_dict)), axis=2)


        pair_input_order=[(0,1),(0,2),(1,2), (1,0),(2,0),(2,1)]

        pairdatashape=(self.num_events,len(pair_input_order), int(l_input.shape[2]*2))
        pairdata=np.empty(pairdatashape)

        for i, pair in enumerate(pair_input_order):
            pairdata[:,i,:l_input.shape[2]] = l_input[:,pair[0],:]
            pairdata[:,i,l_input.shape[2]:] = l_input[:,pair[1],:]

        data=pairdata.reshape(self.num_events*len(pair_input_order), int(pairdata.shape[2]))

        output_half=l_output2.reshape(self.num_events*3, int(l_output2.shape[2]))
        # print("output_half shape:", output_half.shape)
        labels=np.vstack((output_half, output_half))
        
        self.output_dim=labels.shape[1] #? maybe need output to be *6
        self.input_dim=data.shape[1]
        # # datashape=(numevents*len(pair_input_order),l_input.shape[2]*2)
        # datashape=(int(self.num_events),self.input_dim)
        # # print("datashape",datashape)
        # data=np.array(np.zeros(datashape))
        # for i in range(len(pair_input_order)):
        #     combined=np.concatenate((l_input[:,pair_input_order[i][0],:],l_input[:,pair_input_order[i][1],:]),axis=1)
        #     # print(combined.shape)
        #     #add to data
        #     data[:,i*combined.shape[1]:(i+1)*combined.shape[1]]=combined
        
        # # labels=l_output2.reshape((self.num_events,output_dim)
        # labels=l_output2.reshape((self.num_events,self.output_dim))
        self.current_data=data
        self.current_labels=labels
        # self.seed +=15

    def temp_return_data(self):
        return self.current_data, self.current_labels

    @staticmethod
    def worker(instance, start, end, seed=None):
        if seed is not None:
            np.random.seed(seed)
        data_chunk={var: [] for var in (instance.raw_vars_general + [f'{i}_{var}' for i in range(1, 4) for var in ['eta', 'mass', 'phi', 'pt', ]] + instance.flat_output_vars)}


        inputs_chunk= {var: [] for sublist in instance.input_vars for var in (sublist if isinstance(sublist[0], str) else sublist[0])}
        pt_dict={'pt_1': [0.02536545873792836, 0.4934279110259645], 'pt_2': [0.019151151336495566, 0.3995434049215345], 'pt_3': [0.023038543045718854, 0.31375795899486003], 'pt_MET': [0.014081741982300087, 0.13542242088536358]}

        for i in range(start, end):
            sample = {}
          
            eta_low, eta_high = -2.5, 2.5
            mass_low, mass_high = 0, 11
            phi_low, phi_high = -np.pi, np.pi
            # pt_low, pt_high = 0, 1000


            for i in range(1, 4):  # For three leptons
                eta = np.random.uniform(low=eta_low, high=eta_high)
                mass = np.random.uniform(low=mass_low, high=mass_high)
                phi = np.random.uniform(low=phi_low, high=phi_high)
                # pt = np.random.uniform(low=pt_low, high=pt_high)
                pt=generate_random_data(pt_dict[f'pt_{i}'][0], pt_dict[f'pt_{i}'][1])
               

                sample[f'{i}_eta'] = eta
                sample[f'{i}_mass'] = mass
                sample[f'{i}_phi'] = phi
                sample[f'{i}_pt'] = pt
         
            for key in sample:
                inputs_chunk[key].append(sample[key])

            for key, value in sample.items():
                data_chunk[key].append(value)

        return data_chunk, inputs_chunk

    def generate_fake_data2(self, num_samples):
        seed_start = self.seed
        self.flat_output_vars=[]
        for sublist in self.output_vars:
            if isinstance(sublist, list):
                for item in sublist:
                    self.flat_output_vars.append(item)
            else:
                self.flat_output_vars.append(sublist)
        data = {var: [] for var in (self.raw_vars_general + [f'{i}_{var}' for i in range(1, 4) for var in ['eta', 'mass', 'phi', 'pt']] + self.flat_output_vars)}

        inputs = {var: [] for sublist in self.input_vars for var in (sublist if isinstance(sublist[0], str) else sublist[0])}
        pt_dict={'pt_1': [0.02536545873792836, 0.4934279110259645], 'pt_2': [0.019151151336495566, 0.3995434049215345], 'pt_3': [0.023038543045718854, 0.31375795899486003], 'pt_MET': [0.014081741982300087, 0.13542242088536358]}

        num_chunks = os.cpu_count()  # or any other number based on your preference
        if num_chunks > 15: num_chunks = 15
        # print(f'Using {num_chunks} workers')
        chunk_size = num_samples // num_chunks

        futures = []
        # seeds=[1,2,3,5,6,7]
        with ProcessPoolExecutor() as executor:
            for i in range(num_chunks):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i != num_chunks - 1 else num_samples
                futures.append(executor.submit(self.worker, self, start, end, seed=seed_start+i))



        # Collect results from all workers
        for future in tqdm(futures, desc='Collecting results', disable=True):
            chunk_data, chunk_inputs = future.result()
            for key, value in chunk_data.items():
                data[key].extend(value)
            for key, value in chunk_inputs.items():
                inputs[key].extend(value)
        
        tq2=tqdm(enumerate(self.functions), desc='Applying functions', disable=True)
        for i, func in tq2:
            if func is not None:
                func_inputs = [np.array(call_dict_with_list(inputs, var)) for var in self.input_vars[i]]


                func_outputs = func(*func_inputs)

                # Add outputs to data
                if isinstance(self.output_vars[i], list):
                    for j, v in enumerate(self.output_vars[i]):
                        if len(data[v]) == 0:
                            data[v] = func_outputs[j]
                        else:
                            data[v] = np.concatenate((data[v], func_outputs[j]))
                else:
                    if len(data[self.output_vars[i]]) == 0:
                        data[self.output_vars[i]] = func_outputs
                    else:
                        data[self.output_vars[i]] = np.concatenate((data[self.output_vars[i]], func_outputs))
        # for key in sample:
        #     data[key].append(sample[key])
        
        for key in data:
            data[key] = np.array(data[key])
        return data
    
    def add_norm_features(self,data_dict):
        feat_toadd=[ 'norm_mt_12', 'norm_mt_13', 'norm_mt_23']
        feat_orig=feat_toadd.copy()
        feat_orig = [i.replace('norm_', '') for i in feat_orig]
        for i, feat in enumerate(feat_toadd):
            # fake_ptMet=np.asarray
            shape_of_other_arrays =data_dict['pt_1'].shape
            data_dict['pt_MET'] = np.zeros(shape_of_other_arrays)
            # print(data_dict['pt_1'].shape, data_dict['pt_2'].shape, data_dict['pt_3'].shape, data_dict['pt_MET'].shape,data_dict[feat_orig[i]].shape)
            data_dict[feat] = outlier_normalization(data_dict['pt_1'], data_dict['pt_2'], data_dict['pt_3'], data_dict['pt_MET'], data_dict[feat_orig[i]])
        return data_dict
    
    def convert_to_array(self, data_dict):
        
        l_input_shape=(self.num_events,len(self.lepton_input_ordered), len(self.lepton_input_ordered[0]))
        # print("events, particles, input features: ",l_input_shape)
        l_input= np.empty(l_input_shape)

        for i in range(len(self.lepton_input_ordered)):
            for j, feature in enumerate(self.lepton_input_ordered[i]):
                l_input[:,i,j] = data_dict[feature]
        l_output_shape=(self.num_events, len(self.lepton_output_ordered), len(self.lepton_output_ordered[0]))
        # print("events, particle pairs, output kin. features: ",l_output_shape)
        l_output= np.empty(l_output_shape)

        for i in range(len(self.lepton_output_ordered)):
            for j, feature in enumerate(self.lepton_output_ordered[i]):
                l_output[:,i,j] = data_dict[feature]

        return l_input, l_output
    
    def add_vectorfeats(self, data_dict):
        p1_pt=data_dict['pt_1']
        p2_pt=data_dict['pt_2']
        p3_pt=data_dict['pt_3']

        p1_phi=data_dict["1_phi"]
        p2_phi=data_dict["2_phi"]
        p3_phi=data_dict["3_phi"]

        p1_eta=data_dict["eta_1"]
        p2_eta=data_dict["eta_2"]
        p3_eta=data_dict["eta_3"]

        p1_mass=data_dict["mass_1"]
        p2_mass=data_dict["mass_2"]
        p3_mass=data_dict["mass_3"]

        particle1=vector.arr({"pt": p1_pt, "phi": p1_phi, "eta": p1_eta, "mass": p1_mass})
        particle2=vector.arr({"pt": p2_pt, "phi": p2_phi, "eta": p2_eta, "mass": p2_mass})
        particle3=vector.arr({"pt": p3_pt, "phi": p3_phi, "eta": p3_eta, "mass": p3_mass})

        p4_mother12=particle1+particle2
        p4_mother23=particle2+particle3
        p4_mother13=particle1+particle3

        pairs=['12','13','23']
        motherpairs=[p4_mother12, p4_mother13, p4_mother23]
        # features_toadd=[ 'mass', 'pt', 'eta' , 'phi',  'px', 'py', 'pz', 'energy']
        # features_toadd=[ 'mass', 'pt', 'eta']

        add_feat_size=(len(data_dict['pt_1']), len(pairs), len(self.features_toadd))
        add_feat_array= np.empty(add_feat_size)

        for feature in self.features_toadd:
            for i, pair in enumerate(pairs):
                add_feat_array[:, i, self.features_toadd.index(feature)] = getattr(motherpairs[i], feature)
        return add_feat_array

    def usefulvariables(self):
        return self.input_dim, self.output_dim

dontremove_outliers=['event', 'genWeight', 'MET_phi', '1_phi', '1_genPartFlav', '2_phi', '2_genPartFlav', '3_phi', '3_genPartFlav', 'charge_1', 'charge_2', 'charge_3', 'pt_1', 'pt_2', 'pt_3', 'pt_MET', 'eta_1', 'eta_2', 'eta_3', 'mass_1', 'mass_2', 'mass_3']

def compute_percentiles_from_pickle(filename):
    with open(filename, 'rb') as f:
        raw_data_dict = pickle.load(f)
    
    numeric_data_dict = {k: v for k, v in raw_data_dict.items() if (k not in dontremove_outliers) and np.issubdtype(type(v[0]), np.number)}

    # Compute the required percentiles for each numeric feature
    lower_percentiles = {k: np.percentile(v, 0.03) for k, v in numeric_data_dict.items()}
    upper_percentiles = {k: np.percentile(v, 99.7) for k, v in numeric_data_dict.items()}

    return lower_percentiles, upper_percentiles

def remove_outliers(data, lower_percentiles, upper_percentiles):
    del data['MET_phi']
    del data['MET_pt']
    outlier_mask = np.zeros(len(next(iter(data.values()))), dtype=bool)
    for feature_name, values in data.items():
        if (feature_name not in dontremove_outliers) and (feature_name in lower_percentiles):
            # print(feature_name)
            lower_value = lower_percentiles[feature_name]
            upper_value = upper_percentiles[feature_name]
            feature_outlier_mask = (np.array(values) < lower_value) | (np.array(values) > upper_value)
            outlier_mask |= feature_outlier_mask  # update the outlier mask
    
    # Remove rows with outliers from all features in the data dictionary
    cleaned_data = {k: np.array(v)[~outlier_mask].tolist() for k, v in data.items()}
    return cleaned_data


# from torch.utils.data import 

class EpochSampler(Sampler):
    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self.seed = seed

    def __iter__(self):
        # Use self.seed to ensure a consistent order within this epoch
        np.random.seed(self.seed)
        indices = list(np.random.permutation(len(self.data_source)))
        # Ensure the dataset generates new data for the next epoch by modifying the seed
        self.data_source.iterate_seed()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)








