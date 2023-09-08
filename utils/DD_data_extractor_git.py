# from kinematic import *
# from DDkinematic_final import *
from uproot import open
from os import listdir
from fnmatch import filter
from numpy import ravel, unique, array, empty, concatenate, ones, logical_and
from numpy import abs as np_abs
from numpy.random import choice
# from DD_utils_final import isolate_int, count_tauh, call_dict_with_list, replace_prefix_in_list, flatten_2D_list, RandomGenerate_count_tauh
from copy import deepcopy
import numpy as np
from scipy.optimize import brentq
from functools import reduce
from operator import iconcat
from numbers import Number
from pandas import DataFrame
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import sys
sys.path.append('./FeatureRegression/')
from kinematic_custom import *

# /home/ddemler/HNLclassifier/fnn_FeatureRegression/All_particles/kinematic_custom.py

# p4calc, motherpair_vals, Energy_tot

# np.random.seed(39)
# np.rand

# import yaml

# Global variables
output_vars_v1 = ['event', 'genWeight', 'deltaR_12', 'deltaR_13', 'deltaR_23', 'pt_123', 'mt_12', 'mt_13', 'mt_23', 'Mt_tot', 'n_tauh']
output_vars_v2 = ['event', 'genWeight', 'deltaphi_12', 'deltaphi_13', 'deltaphi_23', 'deltaeta_12', 'deltaeta_13', 'deltaeta_23', 
                  'deltaR_12', 'deltaR_13', 'deltaR_23', 'pt_123', 'mt_12', 'mt_13', 'mt_23', 'Mt_tot', 'n_tauh']
output_vars_v3 = ['event', 'genWeight', 'deltaphi_12', 'deltaphi_13', 'deltaphi_23', 'deltaeta_12', 'deltaeta_13', 'deltaeta_23',
                   'deltaR_12', 'deltaR_13', 'deltaR_23', 'pt_123', 'mt_12', 'mt_13', 'mt_23', 'Mt_tot',
                    ['HNL_CM_angle_with_MET_1', 'HNL_CM_angle_with_MET_2'], ['W_CM_angle_HNL_1', 'W_CM_angle_HNL_2'], 
                    ['W_CM_angle_HNL_with_MET_1', 'W_CM_angle_HNL_with_MET_2'], ['HNL_CM_mass_1', 'HNL_CM_mass_2'],
                    ['HNL_CM_mass_with_MET_1', 'HNL_CM_mass_with_MET_2'], 'n_tauh']
output_vars_v4 = ['event', 'genWeight', 
                  'charge_1', 'charge_2', 'charge_3', 
                  'pt_1', 'pt_2', 'pt_3', 'pt_MET', 
                  'eta_1', 'eta_2', 'eta_3',
                  'mass_1', 'mass_2', 'mass_3',
                  'phi_1', 'phi_2', 'phi_3', 'phi_MET', 
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
                  ['HNL_CM_angle_with_MET_1', 'HNL_CM_angle_with_MET_2'], 
                  ['W_CM_angle_to_plane_1', 'W_CM_angle_to_plane_2'], ['W_CM_angle_to_plane_with_MET_1', 'W_CM_angle_to_plane_with_MET_2'],
                  ['HNL_CM_mass_1', 'HNL_CM_mass_2'], 
				  ['HNL_CM_mass_with_MET_1', 'HNL_CM_mass_with_MET_2'], 
                  ['W_CM_angle_12','W_CM_angle_13', 'W_CM_angle_23', 'W_CM_angle_1MET', 'W_CM_angle_2MET', 'W_CM_angle_3MET'],
                  'n_tauh']

output_vars_v5 = ['event', 'genWeight', 
                  'charge_1', 'charge_2', 'charge_3', 
                  'pt_1', 'pt_2', 'pt_3', 'pt_MET', 
                  'eta_1', 'eta_2', 'eta_3',
                  'mass_1', 'mass_2', 'mass_3',
                  'phi_1', 'phi_2', 'phi_3', 'phi_MET', 
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
                  'n_tauh',
                  ['px_1', 'py_1', 'pz_1', 'E_1', 'px_2', 'py_2', 'pz_2', 'E_2', 'px_3', 'py_3', 'pz_3', 'E_3'],
                  ['moth_mass_12', 'moth_mass_13', 'moth_mass_23', 'moth_pt_12', 'moth_pt_13', 'moth_pt_23', 'moth_eta_12', 'moth_eta_13', 'moth_eta_23', 'moth_phi_12', 'moth_phi_13', 'moth_phi_23', 'moth_px_12', 'moth_px_13', 'moth_px_23', 'moth_py_12', 'moth_py_13', 'moth_py_23', 'moth_pz_12', 'moth_pz_13', 'moth_pz_23', 'moth_E_12', 'moth_E_13', 'moth_E_23'],
                  'E_tot']


#===================================================================================================

class Data_extractor():
    """
    A Data_extractor extracts data from a folder of root files containing the anatuples.
    It takes a channel as argument : channel = "tee" "tem" "tmm" "tte" or "ttm"
    When called, it returns the variables of interest for the DNN training
    """
    def __init__(self, channel, raw_vars_general, raw_vars_lepton1, raw_vars_lepton2, raw_vars_lepton3, output_vars, functions, input_vars):
        """
        -channel : flavour of the 3 prompt leptons present in the decay. channel = "tee" "tem" "tmm" "tte" or "ttm"
        -raw_vars_general : names of variables in the root files that will be loaded and which are present only once, and not for each lepton
        -raw_vars_lepton(1,2,3) : end of names of variables in the root files that will be loaded and which are defined for a specific lepton.
                                The naming convention for such variables is L_X where L = Electron(1,2), Muon(1,2), Tau(1,2). Only specify
                                _X, since L will be deduced from the channel
        -output_vars : names of variable of interest that will be created by the data extractor
        -functions : functions that will be used to compute the output_vars (one function for each output_vars in the right order). If the 
                     corresponding output variable is already present as raw variable, put None as a function.
        -input_vars : list of lists of variables that are passed to the functions to compute the output_vars. If the variable in question 
                      is specific to one lepton, then "(1,2,3)_X" will be converted to lepton(1,2,3)_X. 
                      For example, in tee channel "3_mass"->"Electron2_mass"

        """
        self.channel = channel
        if self.channel == "tee":
            self.n_taus = 1
            self.lepton1 = "Tau"
            self.lepton2 = "Electron1"
            self.lepton3 = "Electron2"
        elif self.channel == "tem":
            self.n_taus = 1
            self.lepton1 = "Tau"
            self.lepton2 = "Electron"
            self.lepton3 = "Muon"
        elif self.channel == "tmm":
            self.n_taus = 1
            self.lepton1 = "Tau"
            self.lepton2 = "Muon1"
            self.lepton3 = "Muon2"
        elif self.channel == "tte":
            self.n_taus = 2
            self.lepton1 = "Tau1"
            self.lepton2 = "Tau2"
            self.lepton3 = "Electron"
        elif self.channel == "ttm":
            self.n_taus = 2
            self.lepton1 = "Tau1"
            self.lepton2 = "Tau2"
            self.lepton3 = "Muon"
        else:
            raise ValueError("The channel name \""+channel+"\" is not valid")
        self.raw_vars = raw_vars_general
        for var in raw_vars_lepton1:
            self.raw_vars.append(self.lepton1+var)
        for var in raw_vars_lepton2:
            self.raw_vars.append(self.lepton2+var)
        for var in raw_vars_lepton3:
            self.raw_vars.append(self.lepton3+var)
        
        self.input_vars = replace_prefix_in_list(input_vars, to_replace=['1','2','3'], replace_by=[self.lepton1, self.lepton2, self.lepton3])

        self.functions = functions
        self.output_vars = output_vars
        self.flat_output_vars = flatten_2D_list(output_vars)


    def __call__(self, path, signal_prefix = ['HNL'], real_data_prefix = ['EGamma', 'SingleMuon', 'Tau'], data = None, file_list = None, with_mass_hyp = True):
        """
        Arguments :
            -path : the path to the root files
            -signal_prefix : beginning of names of the files containing the signal (here "HNL"). It can be a string or a list of strings
            -real_data_prefix : beginning of filenames that correspond to real data, and that will be ignored
            -data : dictionnary to which the extracted data will be appended (if None, the dictionary will be created)
            -file_list : list of root files from which data will be extracted (if None, all root files present in path will be used).
            -with_mass_hyp : if True, the data will contain , the HNL mass hypothesis in GeV for the signal events, and a random choice 
                             among the different hypothesis for background events
        Output : 
            -data : dictionary containing the event indices, the variables of interest, the label of the event, and the type of event.
                    By default, data will contain the entries "signal_label" (1 for signal, 0 for background), "channel" and "event_type" (name of the 
                    file in which the events were taken)
        """
        total_keys = deepcopy(self.flat_output_vars)
        total_keys.extend(['signal_label', 'channel', 'event_type'])
        if with_mass_hyp:
            total_keys.append('mass_hyp')
        value_list = []
        for i in range(len(self.flat_output_vars)):
            value_list.append(empty((0,)))
        data = dict(zip(self.flat_output_vars, value_list))

        if with_mass_hyp:
            total_keys.append('mass_hyp')
            data['mass_hyp'] = []

        data['signal_label'] = []
        data['channel'] = []
        data['event_type'] = []


        if set(list(data.keys())) != set(total_keys):
            raise KeyError("The data keys don't match the names of the variable created by the data extractor : ", list(data.keys()), total_keys)

        if file_list == None:
            file_list = filter(listdir(path), '*.root')

        # Create a list of all considered HNL mass hypothesis
        if type(signal_prefix) != list:
                signal_prefix = [signal_prefix]

        mass_hyps = []
        if with_mass_hyp:
            for filename in file_list:
                for prefix in signal_prefix:
                    if filename[:len(prefix)] == prefix:
                        mass_hyps.append(isolate_int(filename, separators=['-', '_'])[0])
            mass_hyps = unique(array(mass_hyps))
        weightsum1=0
        weightsum2=0
        numsum2=0
        

        for filename in file_list:
            RealData = False
            for prefix in real_data_prefix:
                if filename[:len(prefix)] == prefix:
                    RealData = True
            if RealData:
                continue

            # Raw data loading
            limit_charge = 3
            limit_tau_jet = 5
            limit_em_iso = 0.15

            cut = ''
            if self.channel == 'tte':
                cut = '(abs(Tau1_charge + Tau2_charge + Electron_charge) < {}) & (Tau1_idDeepTau2018v2p5VSjet >= {}) & (Tau2_idDeepTau2018v2p5VSjet >= {}) & (Electron_pfRelIso03_all < {})'.format(limit_charge, limit_tau_jet, limit_tau_jet, limit_em_iso)

            if self.channel == 'tee':
                cut = '(abs(Tau_charge + Electron1_charge + Electron2_charge) < {}) & (Tau_idDeepTau2018v2p5VSjet >= {}) & (Electron1_pfRelIso03_all < {}) & (Electron2_pfRelIso03_all < {})'.format(limit_charge, limit_tau_jet, limit_em_iso, limit_em_iso)

            if self.channel == 'tem':
                cut = '(abs(Tau_charge + Electron_charge + Muon_charge) < {}) & (Tau_idDeepTau2018v2p5VSjet >= {}) & (Electron_pfRelIso03_all < {}) & (Muon_pfRelIso03_all < {})'.format(limit_charge, limit_tau_jet, limit_em_iso, limit_em_iso)

            if self.channel == 'tmm':
                cut = '(abs(Tau_charge + Muon1_charge + Muon2_charge) < {}) & (Tau_idDeepTau2018v2p5VSjet >= {}) & (Muon1_pfRelIso03_all < {}) & (Muon2_pfRelIso03_all < {})'.format(limit_charge, limit_tau_jet, limit_em_iso, limit_em_iso)

            if self.channel == 'ttm':
                cut = '(abs(Tau1_charge + Tau2_charge + Muon_charge) < {}) & (Tau1_idDeepTau2018v2p5VSjet >= {}) & (Tau2_idDeepTau2018v2p5VSjet >= {}) & (Muon_pfRelIso03_all < {})'.format(limit_charge, limit_tau_jet, limit_tau_jet, limit_em_iso)            

            anatuple_before_cut = open(path+filename)['Event;1'].arrays(self.raw_vars, library='np') # type: ignore
            weightsum_before_cut = anatuple_before_cut['genWeight'].sum()
            weightsum1 += weightsum_before_cut
            # print('weightsum before cut : ', weightsum_before_cut)
            anatuple = open(path+filename)['Event;1'].arrays(self.raw_vars, cut=cut, library='np') # type: ignore
            weightsum_after_cut = anatuple['genWeight'].sum()
            weightsum2 += weightsum_after_cut
            numsum2 += len(anatuple['genWeight'])

            n = len(anatuple[list(anatuple.keys())[0]])

            if n==0:
                continue

            anatuple['channel'] = [self.channel]*n


            # Creation of the data
            for i, var in enumerate(self.output_vars):
                if self.functions[i] == None:
                    data[var] = concatenate((data[var], anatuple[self.input_vars[i][0]]))
                else:
                    outputs = self.functions[i](*call_dict_with_list(anatuple, self.input_vars[i]))
                    if type(var) == list:
                        for j,v in enumerate(var):
                            data[v] = concatenate((data[v], outputs[j]))
                    else:
                        data[var] = concatenate((data[var], outputs))

            label = 0
            mass = ones((n,))
            for prefix in signal_prefix:
                if filename[:len(prefix)] == prefix:
                    label = 1
                    if with_mass_hyp:
                        mass *= isolate_int(filename,separators=['-', '_'])[0]
            if label == 0 and with_mass_hyp:
                mass = choice(mass_hyps, n)
            
            # Add mass hypothesis
            if with_mass_hyp:
                if 'mass_hyp' in data.keys():
                    data['mass_hyp'] = concatenate((data['mass_hyp'], mass))
                else:
                    data['mass_hyp'] = mass

            # Add signal label (by default)
            if 'signal_label' in data.keys():
                data['signal_label'] = concatenate((data['signal_label'], ones((n,))*label))
            else:
                data['signal_label'] = ones((n,))*label

            # Add channel (by default)
            if 'channel' in data.keys():
                data['channel'].extend([self.channel]*n)
            else:
                data['channel'] = [self.channel]*n

            # Add event type (by default)
            if 'event_type' in data.keys():
                data['event_type'].extend([filename.replace('.root','')]*n)
            else:
                data['event_type'] = [filename.replace('.root','')]*n
        
        # print('weightsum before cut : ', weightsum1)
        # print('weightsum after cut : ', weightsum2)
        # print('numsum after cut : ', numsum2)
        # weightsum= data['genWeight'].sum()
        # print("weightsum = ", weightsum)
 


        return data
    
#===================================================================================================

class Data_extractor_test(Data_extractor):
    def __init__(self):
        output_vars = ['test1', ['test_mix1', 'test_mix2'], 'test2']
        functions = [None, lambda a : (a[0]*a[1], a[0]+a[1]), lambda a : 2*a]
        raw_vars_general = ['test1', 'test2']
        raw_vars_lepton1 = []
        raw_vars_lepton2 = []
        raw_vars_lepton3 = []
        input_vars = [['test1'], ['test1', 'test2'], ['test2']]
        super().__init__(channel='tte', raw_vars_general=raw_vars_general, raw_vars_lepton1=raw_vars_lepton1, raw_vars_lepton2=raw_vars_lepton2, 
                         raw_vars_lepton3=raw_vars_lepton3, output_vars=output_vars, functions=functions, input_vars=input_vars, ) 

        
class Data_extractor_v1(Data_extractor):
    def __init__(self, channel):
        output_vars = deepcopy(output_vars_v1)
        functions =[None, None, deltaR, deltaR, deltaR, sum_pt, transverse_mass, transverse_mass, transverse_mass, total_transverse_mass, count_tauh]
        raw_vars_general = ['event', 'genWeight', 'MET_pt', 'MET_phi']
        raw_vars_lepton1=['_eta', '_mass', '_phi', '_pt', '_genPartFlav']
        raw_vars_lepton2=['_eta', '_mass', '_phi', '_pt', '_genPartFlav']
        raw_vars_lepton3=['_eta', '_mass', '_phi', '_pt', '_genPartFlav']
        input_vars = [['event'], ['genWeight'], ['1_eta', '2_eta', '1_phi', '2_phi'], ['1_eta', '3_eta', '1_phi', '3_phi'],
                      ['2_eta', '3_eta', '2_phi', '3_phi'], [['1_pt', '2_pt', '3_pt'],['1_phi', '2_phi', '3_phi'],['1_eta', '2_eta', '3_eta'],
                       ['1_mass', '2_mass', '3_mass']], ['1_pt', '2_pt', '1_phi', '2_phi'], ['1_pt', '3_pt', '1_phi', '3_phi'], 
                       ['2_pt', '3_pt', '2_phi', '3_phi'], ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi'], 
                       ['channel', '1_genPartFlav', '2_genPartFlav', '3_genPartFlav']]
        super().__init__(channel, raw_vars_general=raw_vars_general, raw_vars_lepton1=raw_vars_lepton1, raw_vars_lepton2=raw_vars_lepton2, 
                         raw_vars_lepton3=raw_vars_lepton3, output_vars=output_vars, functions=functions, input_vars=input_vars)
        
class Data_extractor_v2(Data_extractor):
    def __init__(self, channel):
        output_vars = deepcopy(output_vars_v2)
        functions =[None, None, deltaphi, deltaphi, deltaphi, deltaeta, deltaeta, deltaeta, deltaR, deltaR, deltaR, sum_pt, transverse_mass, transverse_mass, transverse_mass, total_transverse_mass, count_tauh]
        raw_vars_general = ['event', 'genWeight', 'MET_pt', 'MET_phi']
        raw_vars_lepton1=['_eta', '_mass', '_phi', '_pt', '_genPartFlav']
        raw_vars_lepton2=['_eta', '_mass', '_phi', '_pt', '_genPartFlav']
        raw_vars_lepton3=['_eta', '_mass', '_phi', '_pt', '_genPartFlav']
        input_vars = [['event'], ['genWeight'], ['1_phi', '2_phi'], ['1_phi', '3_phi'], ['2_phi', '3_phi'], ['1_eta', '2_eta'], 
                      ['1_eta', '3_eta'], ['2_eta', '3_eta'], ['1_eta', '2_eta', '1_phi', '2_phi'], ['1_eta', '3_eta', '1_phi', '3_phi'],
                      ['2_eta', '3_eta', '2_phi', '3_phi'], [['1_pt', '2_pt', '3_pt'],['1_phi', '2_phi', '3_phi'],['1_eta', '2_eta', '3_eta'],
                       ['1_mass', '2_mass', '3_mass']], ['1_pt', '2_pt', '1_phi', '2_phi'], ['1_pt', '3_pt', '1_phi', '3_phi'], 
                      ['2_pt', '3_pt', '2_phi', '3_phi'], ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi'], 
                      ['channel', '1_genPartFlav', '2_genPartFlav', '3_genPartFlav']]
        super().__init__(channel, raw_vars_general=raw_vars_general, raw_vars_lepton1=raw_vars_lepton1, raw_vars_lepton2=raw_vars_lepton2, 
                         raw_vars_lepton3=raw_vars_lepton3, output_vars=output_vars, functions=functions, input_vars=input_vars)
        
class Data_extractor_v3(Data_extractor):
    def __init__(self, channel):
        output_vars = deepcopy(output_vars_v3)
        functions =[None, None, deltaphi, deltaphi, deltaphi, deltaeta, deltaeta, deltaeta, deltaR, deltaR, deltaR, sum_pt, transverse_mass,
                     transverse_mass, transverse_mass, total_transverse_mass, HNL_CM_angles_with_MET, W_CM_angles_to_plane, 
                     W_CM_angles_to_plane_with_MET, HNL_CM_masses, HNL_CM_masses_with_MET, count_tauh]
        raw_vars_general = ['event', 'genWeight', 'MET_pt', 'MET_phi']
        lepton_specific = ['_eta', '_mass', '_phi', '_pt', '_charge', '_genPartFlav']
        raw_vars_lepton1 = lepton_specific
        raw_vars_lepton2 = lepton_specific
        raw_vars_lepton3 = lepton_specific
        input_vars = [['event'], ['genWeight'], ['1_phi', '2_phi'], ['1_phi', '3_phi'], ['2_phi', '3_phi'], ['1_eta', '2_eta'], 
                      ['1_eta', '3_eta'], ['2_eta', '3_eta'], ['1_eta', '2_eta', '1_phi', '2_phi'], ['1_eta', '3_eta', '1_phi', '3_phi'],
                      ['2_eta', '3_eta', '2_phi', '3_phi'], [['1_pt', '2_pt', '3_pt'],['1_phi', '2_phi', '3_phi'],['1_eta', '2_eta', '3_eta'],
                       ['1_mass', '2_mass', '3_mass']], ['1_pt', '2_pt', '1_phi', '2_phi'], ['1_pt', '3_pt', '1_phi', '3_phi'], 
                      ['2_pt', '3_pt', '2_phi', '3_phi'], ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi'],
                      ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
                      ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], 
                      ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
                      ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], 
                      ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
                      ['channel', '1_genPartFlav', '2_genPartFlav', '3_genPartFlav']]
        super().__init__(channel, raw_vars_general=raw_vars_general, raw_vars_lepton1=raw_vars_lepton1, raw_vars_lepton2=raw_vars_lepton2, 
                         raw_vars_lepton3=raw_vars_lepton3, output_vars=output_vars, functions=functions, input_vars=input_vars)
        

class Data_extractor_v4(Data_extractor):
    def __init__(self, channel):
        output_vars = deepcopy(output_vars_v4)
        functions =[None, None,                 # event, genWeight
                    None, None, None,           # charges
                    None, None, None, None,     # pts
                    None, None, None,           # etas
                    None, None, None,           # masses
                    None, None, None, None,         # phis
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
                    count_tauh]
        raw_vars_general = ['event', 'genWeight', 'MET_pt', 'MET_phi']
        lepton_specific = ['_eta', '_mass', '_phi', '_pt', '_charge', '_genPartFlav']
        raw_vars_lepton1 = lepton_specific
        raw_vars_lepton2 = lepton_specific
        raw_vars_lepton3 = lepton_specific
        input_vars = [['event'], ['genWeight'], 
			        ['1_charge'], ['2_charge'], ['3_charge'], 
			        ['1_pt'], ['2_pt'], ['3_pt'], ['MET_pt'],
			        ['1_eta'], ['2_eta'], ['3_eta'], 
			        ['1_mass'], ['2_mass'], ['3_mass'],
                    ['1_phi'], ['2_phi'], ['3_phi'], ['MET_phi'],
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
			        ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], 
			        ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['channel', '1_genPartFlav', '2_genPartFlav', '3_genPartFlav']]
        super().__init__(channel, raw_vars_general=raw_vars_general, raw_vars_lepton1=raw_vars_lepton1, raw_vars_lepton2=raw_vars_lepton2, 
                         raw_vars_lepton3=raw_vars_lepton3, output_vars=output_vars, functions=functions, input_vars=input_vars)

class Data_extractor_v5(Data_extractor):
    def __init__(self, channel):
        output_vars = deepcopy(output_vars_v5)
        functions =[None, None,                 # event, genWeight
                    None, None, None,           # charges
                    None, None, None, None,     # pts
                    None, None, None,           # etas
                    None, None, None,           # masses
                    None, None, None, None,         # phis
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
                    count_tauh,
                    p4calc,
                    motherpair_vals,
                    Energy_tot]
        raw_vars_general = ['event', 'genWeight', 'MET_pt', 'MET_phi']
        lepton_specific = ['_eta', '_mass', '_phi', '_pt', '_charge', '_genPartFlav']
        raw_vars_lepton1 = lepton_specific
        raw_vars_lepton2 = lepton_specific
        raw_vars_lepton3 = lepton_specific
        input_vars = [['event'], ['genWeight'], 
			        ['1_charge'], ['2_charge'], ['3_charge'], 
			        ['1_pt'], ['2_pt'], ['3_pt'], ['MET_pt'],
			        ['1_eta'], ['2_eta'], ['3_eta'], 
			        ['1_mass'], ['2_mass'], ['3_mass'],
                    ['1_phi'], ['2_phi'], ['3_phi'], ['MET_phi'],
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
			        [ '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        [ '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        [ '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], 
			        [ '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['channel', '1_genPartFlav', '2_genPartFlav', '3_genPartFlav'],
                    ['1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
                    ['1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
                    ['1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass', 'MET_pt']
                    ]
        super().__init__(channel, raw_vars_general=raw_vars_general, raw_vars_lepton1=raw_vars_lepton1, raw_vars_lepton2=raw_vars_lepton2, 
                         raw_vars_lepton3=raw_vars_lepton3, output_vars=output_vars, functions=functions, input_vars=input_vars)


class Data_generator():
    def __init__(self, numevents, normalize=False):


        


        self.output_vars = deepcopy(output_vars_v4)
        self.functions =[None, None,                 # event, genWeight
                    None, None, None,           # charges
                    None, None, None, None,     # pts
                    None, None, None,           # etas
                    None, None, None,           # masses
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
                    RandomGenerate_count_tauh]
        self.raw_vars_general = ['event', 'genWeight', 'MET_pt', 'MET_phi']
        lepton_specific = ['_eta', '_mass', '_phi', '_pt', '_charge', '_genPartFlav']
        raw_vars_lepton1 = lepton_specific
        raw_vars_lepton2 = lepton_specific
        raw_vars_lepton3 = lepton_specific
        self.input_vars = [['event'], ['genWeight'], 
			        ['1_charge'], ['2_charge'], ['3_charge'], 
			        ['1_pt'], ['2_pt'], ['3_pt'], ['MET_pt'],
			        ['1_eta'], ['2_eta'], ['3_eta'], 
			        ['1_mass'], ['2_mass'], ['3_mass'], 
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
			        ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', '1_phi', '2_phi', '3_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'], 
			        ['1_charge', '2_charge', '3_charge', '1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        ['1_pt', '2_pt', '3_pt', 'MET_pt', '1_phi', '2_phi', '3_phi', 'MET_phi', '1_eta', '2_eta', '3_eta', '1_mass', '2_mass', '3_mass'],
			        [ '1_genPartFlav', '2_genPartFlav', '3_genPartFlav']]
			        # ['channel', '1_genPartFlav', '2_genPartFlav', '3_genPartFlav']]
        self.data = self.generate_fake_data2(numevents)
        old_keys = [f"{i}_{var}" for i in range(1, 4) for var in ['charge', 'pt', 'eta', 'mass']] + ['MET_pt']
        new_keys = [f"{var}_{i}" for i in range(1, 4) for var in ['charge', 'pt', 'eta', 'mass']] + ['pt_MET']

        # Convert key names from "1_charge" to "charge_1" etc.
        for old_key, new_key in zip(old_keys, new_keys):
            if old_key in self.data:
                self.data[new_key] = self.data[old_key]
                del self.data[old_key]  # Remove old key-value pair from the dictionary
            
        # return data
        # self.cleanup_data()
        # print(data.keys())
        if normalize:
            self.add_norm_features()
       
    
    def getData(self):
        return self.data
    
    @staticmethod
    def worker(instance, start, end):
        data_chunk={var: [] for var in (instance.raw_vars_general + [f'{i}_{var}' for i in range(1, 4) for var in ['eta', 'mass', 'phi', 'pt', 'charge', 'genPartFlav']] + instance.flat_output_vars)}

        genPartFlav_options = [1,2,3,4]  # Define the possible values for genPartFlav

        inputs_chunk= {var: [] for sublist in instance.input_vars for var in (sublist if isinstance(sublist[0], str) else sublist[0])}
        pt_dict={'pt_1': [0.02536545873792836, 0.4934279110259645], 'pt_2': [0.019151151336495566, 0.3995434049215345], 'pt_3': [0.023038543045718854, 0.31375795899486003], 'pt_MET': [0.014081741982300087, 0.13542242088536358]}

        for i in range(start, end):
            sample = {}
            for var in instance.raw_vars_general:
                if var == 'event':
                    sample[var] = np.random.randint(0, 10000) 
                elif var == 'genWeight':
                    sample[var] = np.random.uniform(-1, 1) 
                elif var == 'MET_pt':
                    sample[var] = generate_random_data(pt_dict['pt_MET'][0], pt_dict['pt_MET'][1])
                elif var == 'MET_phi':
                    sample[var] = np.random.uniform(-np.pi, np.pi)  # Assuming 'MET_phi' ranges from -pi to pi
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
                charge = np.random.choice([1, -1])
                genPartFlav = np.random.choice(genPartFlav_options)

                sample[f'{i}_eta'] = eta
                sample[f'{i}_mass'] = mass
                sample[f'{i}_phi'] = phi
                sample[f'{i}_pt'] = pt
                sample[f'{i}_charge'] = charge
                sample[f'{i}_genPartFlav'] = genPartFlav
            if sample['1_charge']== sample['2_charge'] == sample['3_charge']:
                numflip = np.random.randint(1,4)
                sample[f'{numflip}_charge'] = -sample[f'{numflip}_charge']
            # Initialize empty lists for the output variables
            # for var in self.output_vars:
            #     if isinstance(var, list):
            #         for v in var:
            #             data[v] = []
            #     else:
            #         data[var] = []
            for key in sample:
                inputs_chunk[key].append(sample[key])

            for key, value in sample.items():
                data_chunk[key].append(value)

        return data_chunk, inputs_chunk

    def generate_fake_data2(self, num_samples):
        self.flat_output_vars=[]
        for sublist in self.output_vars:
            if isinstance(sublist, list):
                for item in sublist:
                    self.flat_output_vars.append(item)
            else:
                self.flat_output_vars.append(sublist)
        data = {var: [] for var in (self.raw_vars_general + [f'{i}_{var}' for i in range(1, 4) for var in ['eta', 'mass', 'phi', 'pt', 'charge', 'genPartFlav']] + self.flat_output_vars)}

        # data = {var: [] for var in (self.raw_vars_general + [f'{i}_{var}' for i in range(1, 4) for var in ['eta', 'mass', 'phi', 'pt', 'charge', 'genPartFlav']] + self.output_vars)}
        genPartFlav_options = [1,2,3,4]  # Define the possible values for genPartFlav

        inputs = {var: [] for sublist in self.input_vars for var in (sublist if isinstance(sublist[0], str) else sublist[0])}
        pt_dict={'pt_1': [0.02536545873792836, 0.4934279110259645], 'pt_2': [0.019151151336495566, 0.3995434049215345], 'pt_3': [0.023038543045718854, 0.31375795899486003], 'pt_MET': [0.014081741982300087, 0.13542242088536358]}

        num_chunks = os.cpu_count()  # or any other number based on your preference
        if num_chunks > 15: num_chunks = num_chunks - 5
        print(f'Using {num_chunks} workers')
        chunk_size = num_samples // num_chunks

        futures = []
        # seeds=[1,2,3,5,6,7]
        with ProcessPoolExecutor() as executor:
            for i in range(num_chunks):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i != num_chunks - 1 else num_samples
                futures.append(executor.submit(self.worker, self, start, end))



        # Collect results from all workers
        for future in tqdm(futures, desc='Collecting results'):
            chunk_data, chunk_inputs = future.result()
            for key, value in chunk_data.items():
                data[key].extend(value)
            for key, value in chunk_inputs.items():
                inputs[key].extend(value)
        
        tq2=tqdm(enumerate(self.functions), desc='Applying functions')
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

    def generate_fake_data(self, num_samples):
        # Initialize a dictionary with each key being a variable and each value being an empty list
        # Flatten the list
        flat_output_vars=[]
        for sublist in self.output_vars:
            if isinstance(sublist, list):
                for item in sublist:
                    flat_output_vars.append(item)
            else:
                flat_output_vars.append(sublist)
        # flat_output_vars = [item for sublist in self.output_vars for item in sublist]

        # Use the flattened list in your dictionary comprehension
        data = {var: [] for var in (self.raw_vars_general + [f'{i}_{var}' for i in range(1, 4) for var in ['eta', 'mass', 'phi', 'pt', 'charge', 'genPartFlav']] + flat_output_vars)}

        # data = {var: [] for var in (self.raw_vars_general + [f'{i}_{var}' for i in range(1, 4) for var in ['eta', 'mass', 'phi', 'pt', 'charge', 'genPartFlav']] + self.output_vars)}
        genPartFlav_options = [1,2,3,4]  # Define the possible values for genPartFlav

        inputs = {var: [] for sublist in self.input_vars for var in (sublist if isinstance(sublist[0], str) else sublist[0])}
        pt_dict={'pt_1': [0.02536545873792836, 0.4934279110259645], 'pt_2': [0.019151151336495566, 0.3995434049215345], 'pt_3': [0.023038543045718854, 0.31375795899486003], 'pt_MET': [0.014081741982300087, 0.13542242088536358]}

        tq = tqdm(range(num_samples), desc='Generating raw data')
        for j in tq:
            sample = {}
            for var in self.raw_vars_general:
                if var == 'event':
                    sample[var] = np.random.randint(0, 10000) 
                elif var == 'genWeight':
                    sample[var] = np.random.uniform(-1, 1) 
                elif var == 'MET_pt':
                    sample[var] = generate_random_data(pt_dict['pt_MET'][0], pt_dict['pt_MET'][1])
                elif var == 'MET_phi':
                    sample[var] = np.random.uniform(-np.pi, np.pi)  # Assuming 'MET_phi' ranges from -pi to pi
            
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
                charge = np.random.choice([1, -1])
                genPartFlav = np.random.choice(genPartFlav_options)

                sample[f'{i}_eta'] = eta
                sample[f'{i}_mass'] = mass
                sample[f'{i}_phi'] = phi
                sample[f'{i}_pt'] = pt
                sample[f'{i}_charge'] = charge
                sample[f'{i}_genPartFlav'] = genPartFlav
            if sample['1_charge']== sample['2_charge'] == sample['3_charge']:
                numflip = np.random.randint(1,4)
                sample[f'{numflip}_charge'] = -sample[f'{numflip}_charge']
            # Initialize empty lists for the output variables
            # for var in self.output_vars:
            #     if isinstance(var, list):
            #         for v in var:
            #             data[v] = []
            #     else:
            #         data[var] = []
            for key in sample:
                inputs[key].append(sample[key])
                # data[key].append(sample[key])

            for key, value in sample.items():
                data[key].append(value)
        tq2=tqdm(enumerate(self.functions), desc='Applying functions')
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
    
    def add_norm_features(self):
        feat_toadd=['norm_mt_1(23)', 'norm_mt_2(13)', 'norm_mt_3(12)',
                  'norm_mt_MET(12)', 'norm_mt_MET(13)', 'norm_mt_MET(23)',
                  'norm_mt_1(2MET)', 'norm_mt_1(3MET)', 'norm_mt_2(1MET)', 'norm_mt_2(3MET)', 'norm_mt_3(1MET)', 'norm_mt_3(2MET)', 'norm_mt_12', 'norm_mt_13', 'norm_mt_23']
        feat_orig=feat_toadd.copy()
        feat_orig = [i.replace('norm_', '') for i in feat_orig]
        for i, feat in enumerate(feat_toadd):
            self.data[feat] = outlier_normalization(self.data['pt_1'], self.data['pt_2'], self.data['pt_3'], self.data['pt_MET'], self.data[feat_orig[i]])
        return
# def inverted_exponential_cdf(p, lambd, c):
#     """Inverted exponential cumulative distribution function."""
#     a = 0
#     b = 10
#     while np.sign(exponential_cdf(a, lambd, c) - p) == np.sign(exponential_cdf(b, lambd, c) - p):
#         b *= 2
#     return brentq(lambda x: exponential_cdf(x, lambd, c) - p, a, b)
def inverted_exponential_cdf(p, lambd, c):
    """Inverted exponential cumulative distribution function."""
    # return (-np.log(1 - p) - c) / lambd
    return (c - np.log(1 - p)) / lambd

def generate_random_data( lambd, c):
    """Generate random data from the approximate CDF."""
    p = np.random.uniform(0, 1)

    return inverted_exponential_cdf(p, lambd, c)

        
def exponential_cdf(x, lambd,c):
    """The exponential cumulative distribution function."""
    return 1 - np.exp(-lambd * x+c)

def outlier_normalization(Pt_1,Pt_2, Pt_3, MET, Xvar):
    Psum=np.sum([Pt_1,Pt_2, Pt_3, MET])
    return Xvar/Psum


def remove_outliers(data, feature_name, limits):
    feature_limits = limits.get(feature_name)
    if feature_limits is None:
        lower_limit, upper_limit = 0.03, 99.7
    elif 'do_not_cut' in feature_limits and feature_limits['do_not_cut']:
        return data[feature_name]
    else:
        lower_limit = feature_limits.get('lower_percentile', 0.03)
        upper_limit = feature_limits.get('upper_percentile', 99.7)

    lower_value, upper_value = np.percentile(data[feature_name], [lower_limit, upper_limit])
    mask = (data[feature_name] >= lower_value) & (data[feature_name] <= upper_value)
    return data[feature_name][mask]

def remove_all_outliers(data, limits):
    for feature_name in data.keys():
        data[feature_name] = remove_outliers(data, feature_name, limits)
    return data

def flatten_2D_list(multi_dim_list):
    new_list = []
    for ele in multi_dim_list:
        if type(ele) is list:
            new_list.append(ele)
        else:
            new_list.append([ele])
    return reduce(iconcat, new_list, [])


def normalize(dataframe, key, sum, weight_name='genWeight'):
    classes = dataframe[key].unique()
    if isinstance(sum, Number):
       sum = dict(zip(classes, [sum]*len(classes)))
    if len(sum)!= len(classes):
        raise ValueError("The number of elements in sum doesn't match the number of classes in the dataframe")
    
    for c in classes:
        mask = dataframe[key] == c
        dataframe.loc[mask, weight_name] *= sum[c] / dataframe.loc[mask, weight_name].sum()

    return dataframe

def bucketize(dataframe, key, return_dict = True):
    """
    Input : 
        -dataframe : pandas dataframe or dictionary
        -key : key of the dataframe representing the classes names, that will be turned into indices
        -return_dict : if True, the function returns the dictionary linking the former class names to the corresponding integer indices
    Output : 
        -output : dataframe with integers replacing the values of dataframe[key] (one index per different value)  
        -class_names : dictionary linking the former class names to the corresponding integer indices    
    """
    dictionary = False
    if type(dataframe) == dict:
        dictionary = True
        dataframe = pd.DataFrame(dataframe)

    class_names = {}
    for i,class_name in enumerate(dataframe[key]):
        if not class_name in class_names:
            class_names[class_name] = len(class_names)
    output = dataframe.copy()
    output[key].replace(list(class_names.keys()), list(class_names.values()), inplace=True)

    if dictionary:
        output = output.to_dict()
    
    if return_dict : 
        return output, class_names
    return output

def count_tauh(channel, genPartFlavs_1, genPartFlavs_2, genPartFlavs_3):
    """
    Input : 
        -channel : string of three characters corresponding to the three prompt leptons in the decay
        -genPartFlavs : 3 (1 for each lepton) arguments describing the flavour of genParticle
    Output :
        -number of hadronic taus present in the event (either 0, 1 or 2) 
    """
    # if len(args) == 1:
    #     if len(args[0]) != 4:
    #         raise TypeError("Wrong number of arguments")
    #     channel = args[0][0][0]
    #     genPartFlavs = args[0][1:]
    # elif len(args) == 4:
    #     channel = args[0][0]
    #     genPartFlavs = args[1:]
    # else:
    #     raise TypeError("Wrong number of arguments")
    channel = channel[0]
    is_list = False
    genPartFlavs = [genPartFlavs_1, genPartFlavs_2, genPartFlavs_3]
    if type(genPartFlavs[0]) == list:
        is_list = True
        for lepton_flav in genPartFlavs:
            lepton_flav = np.array(lepton_flav)
    n_tauh = np.zeros_like(genPartFlavs[0]).astype('int64')
    for i, lepton_flav in enumerate(genPartFlavs):
        if channel[i] == 't':
            n_tauh += (lepton_flav==5).astype('int64')
    
    if is_list:
        n_tauh = n_tauh.tolist()
    
    return n_tauh 

def replace_prefix_in_list(list_, to_replace, replace_by):
    """
    Input :
        -list_ : python list of strings, potentially multidimensional
        -to_replace : list of characters or substrings that will be replaced in each element of the list
        -replace_by : list of characters or substrings that will replace the "to_replace" elements
    Output :
        -list with the same structure as the input list, with the replaced characters
    """
    if type(list_) != list:
        for i,s in enumerate(to_replace):
            if list_[:len(s)] == s:
                list_ = list_.replace(list_[:len(s)],replace_by[i])
        return list_
    else:
        sublist = []
        for el in list_:
            sublist.append(replace_prefix_in_list(el, to_replace, replace_by))
        return sublist
    


def isolate_int(string, separators):
    if type(separators) != list:
       separators = [separators]
    ints = []

    for i in range(1,len(separators)):
       string = string.replace(separators[i], separators[0])

    for z in string.split(separators[0]):
       if z.isdigit():
          ints.append(int(z))

    return ints

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

def RandomGenerate_count_tauh(genPartFlavs_1, genPartFlavs_2, genPartFlavs_3):

    channels = ['tee', 'tem', 'tmm', 'tte', 'ttm']

    channel = random.choice(channels)
    """
    Input : 
        -channel : string of three characters corresponding to the three prompt leptons in the decay
        -genPartFlavs : 3 (1 for each lepton) arguments describing the flavour of genParticle
    Output :
        -number of hadronic taus present in the event (either 0, 1 or 2) 
    """
    is_list = False
    genPartFlavs = [genPartFlavs_1, genPartFlavs_2, genPartFlavs_3]
    if type(genPartFlavs[0]) == list:
        is_list = True
        for lepton_flav in genPartFlavs:
            lepton_flav = np.array(lepton_flav)
    n_tauh = np.zeros_like(genPartFlavs[0]).astype('int64')
    for i, lepton_flav in enumerate(genPartFlavs):
        if channel[i] == 't':
            n_tauh += (lepton_flav==5).astype('int64')
    
    if is_list:
        n_tauh = n_tauh.tolist()
    
    return n_tauh 


def split_dataset(data, ratio_train = 0.75, shuffle = True, print_sizes = True):
    """
    Input : 
        - data : dictionnary containing the variables of interest for each event
        - ratio_train : percentage of train + validation events going in the train dataset
        - shuffle : if True, the training and validation set are shuffled
    Output :
        - data_train : training dataset as pandas dataframe
        - data_val : validation dataset as pandas dataframe
        - data_test : test dataset as pandas dataframe
        - data_meas : measurement dataset as pandas dataframe
    """
    df = DataFrame.from_dict(data)

    data_tv = df.query("(event % 4 == 0) or (event % 4 == 1)")
    data_test = df.query("event % 4 == 2").reset_index(drop=True)
    data_meas = df.query("event % 4 == 3").reset_index(drop=True)

    if shuffle:
        data_tv = data_tv.sample(frac=1).reset_index(drop=True)

    data_train = data_tv.sample(frac = ratio_train)
    data_val = data_tv.drop(data_train.index)


    if print_sizes :
        N = len(df)
        print("Total number of events : ", N)
        print("Train set : {:.2f} %".format(100*len(data_train)/N))
        print("Validation set : {:.2f} %".format(100*len(data_val)/N))
        print("Test set : {:.2f} %".format(100*len(data_test)/N))
        print("Measurement set : {:.2f} %".format(100*len(data_meas)/N))

    return data_train, data_val, data_test, data_meas

def split_dataset2(data, ratio_train = 0.5, ratio_val = 0.1, shuffle = True, print_sizes = True):
    """
    Input : 
        - data : dictionnary containing the variables of interest for each event
        - ratio_train : percentage of events going in the train dataset
        - ratio_val : percentage of events going in the validation dataset
        - shuffle : if True, the training and validation set are shuffled
    Output :
        - data_train : training dataset as pandas dataframe
        - data_val : validation dataset as pandas dataframe
        - data_test : test dataset as pandas dataframe
    """
    df = DataFrame.from_dict(data)

    # Calculate total number of events here
    N = len(df)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    data_train = df.sample(frac = ratio_train)
    df = df.drop(data_train.index)

    data_val = df.sample(frac = ratio_val / (1 - ratio_train))
    data_test = df.drop(data_val.index)


    if print_sizes :
        print("Total number of events : ", N)
        print("Train set : {:.2f} %".format(100*len(data_train)/N))
        print("Validation set : {:.2f} %".format(100*len(data_val)/N))
        print("Test set : {:.2f} %".format(100*len(data_test)/N))

    return data_train, data_val, data_test


def split_dataset_multitrain(data, ratio_train1=0.4, ratio_train2=0.4, ratio_val1=0.1, ratio_val2=0.1, shuffle=True, print_sizes=True):
    """
    Input : 
        - data : dictionary containing the variables of interest for each event
        - ratio_train1 : percentage of events going in the first train dataset
        - ratio_train2 : percentage of events going in the second train dataset
        - ratio_val1 : percentage of events going in the first validation dataset
        - ratio_val2 : percentage of events going in the second validation dataset
        - shuffle : if True, the datasets are shuffled
    Output :
        - data_train1 : first training dataset as pandas dataframe
        - data_train2 : second training dataset as pandas dataframe
        - data_val1 : first validation dataset as pandas dataframe
        - data_val2 : second validation dataset as pandas dataframe
    """
    
    df = DataFrame.from_dict(data)
    N = len(df)
    
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
        
    data_train1 = df.sample(frac=ratio_train1)
    df = df.drop(data_train1.index)
    
    data_train2 = df.sample(frac=ratio_train2 / (1 - ratio_train1))
    df = df.drop(data_train2.index)
    
    data_val1 = df.sample(frac=ratio_val1 / (1 - ratio_train1 - ratio_train2))
    df = df.drop(data_val1.index)
    
    data_val2 = df
    
    if print_sizes:
        print("Total number of events:", N)
        print("Train1 set: {:.2f} %".format(100*len(data_train1)/N))
        print("Train2 set: {:.2f} %".format(100*len(data_train2)/N))
        print("Validation1 set: {:.2f} %".format(100*len(data_val1)/N))
        print("Validation2 set: {:.2f} %".format(100*len(data_val2)/N))
        
    return data_train1, data_train2, data_val1, data_val2
