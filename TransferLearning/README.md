<!-- subdirectory name: TransferLearning

There is one python file: TransferLearning.py

TransferLearning.py
    Does transfer learning on the pretrained multivariate regression model. Predicts whether the real (simulated monte carlo) data event is background or an HNL event. Uses yaml file for specifications (example Transfer_model1.yaml)

Transfer_model1.yaml
    you speicfy the pretrained model prefix. Also specify: dorpout (set to 0 if you want it to not do dropout), activation function for the added model, added model hidden layers depth and width, optimizer, scheduler (platue) patience factor, model patience, unfreeze epoch  (when the weights of the pretrained model are unforzen), num epochs total. You also specify the train dataset path. -->


# TransferLearning Subdirectory

Welcome to the `TransferLearning` subdirectory! This section focuses on applying transfer learning to a pre-trained multivariate regression model. The model's task is to predict whether a given event from real (simulated Monte Carlo) data is background noise or an HNL (Heavy Neutral Lepton) event.

## Overview

- [File Descriptions](#file-descriptions)
  - [TransferLearning.py](#transferlearningpy)
  - [Transfer_model1.yaml](#transfer_model1yaml)
- [Usage](#usage)

## File Descriptions

### TransferLearning.py
This script applies transfer learning techniques on the pre-trained multivariate regression model. The prediction task is to determine if a real (simulated Monte Carlo) data event is a background or an HNL event. It uses a YAML configuration file to specify the parameters and settings (e.g., `Transfer_model1.yaml`).

### Transfer_model1.yaml
A configuration file where various model and training parameters are specified:

- Pre-trained model prefix
- Dropout (set to 0 if dropout is not desired)
- Activation function for the appended model
- Added model's hidden layers - both depth and width
- Optimizer choice
- Scheduler (plateau) patience factor
- Model's patience level
- Epoch at which the pre-trained model weights are unfrozen (`unfreeze epoch`)
- Total number of training epochs
- Path to the training dataset

## Usage

1. Navigate to the `TransferLearning` subdirectory.
2. Modify the `Transfer_model1.yaml` configuration file as necessary.
3. Run the `TransferLearning.py` script to commence transfer learning on the specified dataset.

---

This README gives a brief introduction and guide to the `TransferLearning` subdirectory. Adjustments can be made as per specific requirements or additional details.