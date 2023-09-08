# README for HNLclassifier Repository

This repository is dedicated to the study of Heavy Neutral Leptons (HNLs) decaying to three leptons at CMS. The study specifically targets events with at least one hadronically decayed tau(tau_h) among the three final-state leptons.

The project aims to contribute to the field of particle physics, particularly in the search for new physics "Beyond the Standard Model" (BSM). Observations such as neutrino oscillation and mass suggest that there are physical phenomena yet to be fully understood within the framework of the Standard Model. Explorations in this repository could help elucidate these phenomena and contribute to the development of theories that provide a more complete understanding of the universe's fundamental particles and forces.

## Repository Structure

This repository contains several subdirectories, each focusing on a specific aspect of the analysis:

1. **SignificancePlotting**: Contains scripts for plotting the significance of signal classification for different kinematic features or trained Feedforward Neural Network (FNN) models.
2. **FakeDatasetMaking**: Contains scripts for generating fake raw input variables such as eta, phi, and p_t. These values are then used to calculate kinematic features of this fake data.
3. **DNN**: Ctrains a Deep Neural Network (DNN) that takes in both raw input variables and calculated kinematic features. This is based off Nelson Glardons [work](https://github.com/Nelson-00/TP4b).
.
4. **utils**: Contains Python scripts that provide utility functions used across different subdirectories of the project.
5. **FeatureRegression**: DNN model that predicts the calculated kinematic variables of the particles using the raw input data.
6. **TransferLearning**: Uses transfer learning with the pretrained regression dnn to do HNL classification. 

Additionally, the repository includes the `data_extraction.ipynb` notebook, which is used for data extraction and preprocessing.

## Getting Started

To get started with this project, clone the repository and ensure that all the necessary Python packages are installed. Run the `data_extraction.ipynb` notebook to extract and preprocess the data. After that, you can explore the various subdirectories to generate fake data, train models, and evaluate their performance.

## Additional Information

For more detailed information about the project and the tasks performed in each subdirectory, refer to the README files in each subdirectory and the documentation within the Python scripts and Jupyter notebooks. This project provides a systematic and rigorous approach to the potential discovery of HNL particles, offering valuable insights into the performance of machine learning models across a variety of conditions.