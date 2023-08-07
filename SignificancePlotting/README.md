# README for the SignificancePlotting Folder

This subfolder, part of a larger project on the analysis of Heavy Neutral Lepton (HNL) particles, is focused on plotting the significance of signal classification for different kinematic features or trained Feedforward Neural Network (FNN) models. The plotted results allow for the comparison of different models in terms of their ability to identify HNL signals under various mass hypotheses.

## Folder Structure

The folder contains the following files:

- `SignificanceCalculation.ipynb`: A Jupyter notebook containing the main analysis pipeline.
- `Significance_func.py`: A Python script containing several functions used in the Jupyter notebook for data processing, histogram making, significance score calculations, and plotting.

## Analysis Pipeline

The `SignificanceCalculation.ipynb` notebook includes the following sections:

1. **Preprocessing**: This section prepares the data for analysis.
2. **Histogram Making**: This section generates scores using a previously trained Deep Neural Network (DNN) model. The scores are then used to create histograms.
3. **Histogram and Plots**: The histograms represent the distribution of scores predicted by the model for each channel. These scores are essentially the model's confidence that a given event is a signal (HNL) rather than a background. The bins of the histogram are made such that the signal predicted amount of events (each event multiplied by its weight ‘genWeight’ and summed) is the same for each bin. This allows a visual assessment of how well the model is distinguishing between signal and background events.
4. **Significance Scores**: The significance scores, calculated based on the model output scores, the 'signal_label', and the 'weightOriginal', serve as a measure of the separation power of the model. A higher significance score indicates that the model is better able to differentiate HNL signals from background noise.
5. **Plotting Significance Scores**: This section visualizes the significance of the model's predictions. For each channel and mass hypothesis or 'Mt_tot' value, the events are divided into bins based on their model output scores or 'Mt_tot' values, and the significance is calculated for each bin. This results in a plot of maximum significance for each mass hypothesis or 'Mt_tot' value, providing a visual representation of the model's performance across a range of conditions.

## Getting Started

To run the analysis, open and run the `SignificanceCalculation.ipynb` notebook. Ensure that the necessary data is available at the defined path and that all the required Python packages are installed.

## Additional Information

For more detailed information about the analysis and the functions used, refer to the documentation within the `SignificanceCalculation.ipynb` notebook and the `Significance_func.py` script. The ability to calculate and plot significance scores for different mass hypotheses or 'Mt_tot' values adds another layer to our analysis, providing valuable insight into the model's performance across a variety of conditions.