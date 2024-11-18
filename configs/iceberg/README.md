This folder contains a set of sample config files for each of the tools. These configurations are used to work on the Iceberg dataset obtainable from https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data . The number in the config indicates the order in which the corresponding tool can be run with each config, in order to have a full run of all of the tools, and get results.

For each configuration, the first word in the file name is the name of the tool to be called, the rest is just description. 

For this dataset and these configs, the steps are:

1) Train the specified model with the training set. It also trains a time-series model that will be used for the metric calculation.
2) Label the test data using the trained model.
3) Merge the train and labelled test data to have a bigger data set.
4) (OPTIONAL) Train the model(s) again with the combined data set. 
5) (OPTIONAL) Evaluate the resulting trained model.
6) Generate drift for the combined data set.
7) (OPTIONAL) Run the Predictor in predict mode to only get the results of applying the model and time-series model to the drifted data set.
8) Run the Predictor in analyze mode to get the results and metrics of the drifted data set.
