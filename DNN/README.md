# README for the DNN Folder

This subfolder is part of a larger project on the analysis of Heavy Neutral Lepton (HNL) particles. The main goal of this subfolder is to train a Deep Neural Network (DNN) and run a Neural Architecture Search (NAS) on varying depths and widths.

## Folder Structure

The folder contains the following files:

- `Pytorch_feature_calculationJul26.py`: This Python script contains the main code for training the DNN and running the NAS.
- `dnnconfig.yml`: A YAML file containing the configuration parameters or settings for the DNN and NAS.
- `DNN_models.py`: This Python script contains the definitions of the DNN models used in the project.
- `pytorch2python.py`: This Python script is used for converting PyTorch models to Python or for other PyTorch-related tasks.

## DNN and NAS Pipeline

The `Pytorch_feature_calculationJul26.py` script implements a Neural Architecture Search (NAS) algorithm for fully connected Deep Neural Networks (DNNs). The main steps of the script are:

1. **Load Data**: The script loads training, validation, and test datasets from specified pickle files. The datasets are expected to be Pandas DataFrame objects.
2. **Data Loaders**: PyTorch data loaders are created for these datasets. These loaders are used to provide data to the model during the training and testing phases.
3. **Model Architecture**: The architecture of each model, defined by the depth and width of the network, is determined by the variable `hidden_layer_configs`.
4. **Train and Test Models**: The script trains models using all input features defined in the `selectionlonger` list, and then tests the models.

## Getting Started

To train the DNN and run the NAS, run the `Pytorch_feature_calculationJul26.py` script. Ensure that the necessary data is available at the defined path and that all the required Python packages are installed.

## Additional Information

For more detailed information about the DNN training and NAS process, refer to the documentation within the `Pytorch_feature_calculationJul26.py` script. This script provides a foundation for building and optimizing a Deep Neural Network (DNN) that can be used for further analysis of the synthetic datasets.