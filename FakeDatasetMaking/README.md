# README for the FakeDatasetMaking Folder

This subfolder is part of a larger project on the analysis of Heavy Neutral Lepton (HNL) particles. The main goal of this subfolder is to generate a synthetic dataset that mimics the results of physics events involving multiple particles. These fake datasets are then used to create a Graph Neural Network (GNN) that predicts the kinematic features of the fake data.

## Folder Structure

The folder contains the following files:

- `Generate_fake_dataset.ipynb`: A Jupyter notebook containing the main pipeline for generating and cleaning the synthetic dataset.
- `GenerateDataClean.yaml`: A YAML file containing the settings and thresholds for data cleaning, specifically for removing outliers from the synthetic data.
- `README.md`: This file provides information about the project and its files.

## Data Generation Pipeline

The `Generate_fake_dataset.ipynb` notebook consists of the following sections:

1. **Data Generation**: This section generates synthetic data that represents the results of physics events involving multiple particles. Each particle is characterized by different properties such as `eta`, `mass`, `phi`, `pt`, `charge`, and `genPartFlav`. The data generation process involves creating a dictionary of these variables and generating random data for each variable based on relevant statistical distributions.

2. **Data Cleaning**: This section removes outliers from the synthetic data using the `remove_outliers` function. The outliers are identified based on the limits defined in the `GenerateDataClean.yaml` file.

3. **Data Storage**: This section stores the cleaned synthetic data as a pickle file for later use.

## Getting Started

To generate the synthetic data, open and run the `Generate_fake_dataset.ipynb` notebook. Ensure that the required Python packages are installed.

## Additional Information

For more detailed information about the data generation and cleaning processes, refer to the documentation within the `Generate_fake_dataset.ipynb` notebook. This synthetic data generation process provides a foundation for building a Graph Neural Network (GNN) that predicts the kinematic features of the fake data.