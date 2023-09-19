<!-- This folder is called "FeatureRegression"
    It trains a model on fake "generated" data of the raw input variables of the particles (eta, phi, mass, pt) 
    It trains to predict the output features of an event such as relative angles, transveerse mass total, etc.

In this folder there are three python files, fnn_datagenerator, kinematic_custom, and regression_train and some accompanying yaml files

kinematic_custom.py:
    This is custom version of kinematic functions that can be found in HNLCLASSIFIER/utils. This one has more kinmetic functions to calculate and does not take charge and an input variable (more combinations). 
    This function is called by the fnn_datagenerator function
    
fnn_datagenerator.py:
    This file extends the pytorch dataset class to generate new "fake" data for training each epoch. This allows for the model to not overfit since the model always uses new data. It can create both training and testing data. It uses the file threshold_limits.yaml to set limits on the kimenatic features so that the data is more similar to the "real" (monte carlo simulated) data. To create the kinematic data it uses the kinematic_custom python file. 

regression_train.py
    This is the multivariate regression model code. This uses a DNN with specifications that can be defined in a yaml file (simpleRegression.yml is an example). The yaml can specify the activation function, the cuda device, the hidden layers (depth and width), learning rate initial, optimizer, patience, training, validation, testing data per epoch size, learning rate scheduler, and batch size -->


# FeatureRegression Subdirectory

Welcome to the `FeatureRegression` subdirectory! Here, we train models on synthetic data of the raw input variables of particles (like eta, phi, mass, pt). Our primary goal is to predict output features of an event, such as relative angles, transverse mass total, and more.

## Overview

- [File Descriptions](#file-descriptions)
  - [kinematic_custom.py](#kinematic_custompy)
  - [fnn_datagenerator.py](#fnn_datageneratorpy)
  - [regression_train.py](#regression_trainpy)
- [Usage](#usage)

## File Descriptions

### kinematic_custom.py
A custom version of the kinematic functions, usually found in `HNLCLASSIFIER/utils`. This variant offers more kinematic functions and omits the charge as an input variable. The `fnn_datagenerator` function utilizes this.

### fnn_datagenerator.py
Extends the PyTorch Dataset class to generate synthetic data for each training epoch. This ensures the model isn't prone to overfitting by always using new data. The kinematic feature limits are set using `threshold_limits.yaml` to make the synthetic data closely resemble the "real" (Monte Carlo simulated) data. For kinematic data creation, it calls upon the `kinematic_custom` Python file.

### regression_train.py
Contains the multivariate regression model's code. It employs a deep neural network (DNN) with configurations that can be defined in YAML files (an example being `simpleRegression.yml`). This configuration caters to:

- Activation function
- CUDA device
- Hidden layers (depth and width)
- Initial learning rate
- Optimizer
- Patience
- Training, validation, testing data sizes per epoch
- Learning rate scheduler
- Batch size

## Usage

1. Navigate to the `FeatureRegression` subdirectory.
2. Modify the YAML configuration files as necessary.
3. Execute the desired Python script.
.