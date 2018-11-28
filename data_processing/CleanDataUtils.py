from data_processing import dataUtils

# Drop different sets of features here that can be used for any particular experiment

# Retrieve the processed data set
feature_set = dataUtils.retreiveDataSet("../feature_sets/jonstest7.csv")

#Drop the label and ID column, since we dont want to include these in the clustering algorithm.
feature_set_copy = feature_set
feature_set_copy.drop(columns=['label'])
feature_set_copy.drop(columns=['userID'])