import pandas as pd
import numpy as np
import seaborn as sb
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.markers as markers
import os


# Provides utilities for processing data, including.
# Authors: Jon Andrew and Claudia Trafton
# Authored: 11/17/2019

# ---- Important Globally Used Variables ----

global datasetRoot
global datasetFolders
global dataSetLabels

datasetRoot = '../age_anonymized/'
datasetFolders = [datasetRoot + '-15/', datasetRoot + '16-19/', datasetRoot + '20-29/', \
                  datasetRoot + '30-39/', datasetRoot + '40-49/', datasetRoot + '50+/']


# The labels (i.e. which age group the datapoint belongs to)
#Key: the folder name it belongs to
#Value: the integer indicating the group number
# Group 1: 0-15
# Group 2: 16-19
# Group 3: 20-29
# Group 4: 30-39
# Group 5: 40-49
# Group 6: 50+
dataSetLabels = {datasetFolders[0]: 0, datasetFolders[1]: 1, datasetFolders[2]: 2,
                 datasetFolders[3]: 3, datasetFolders[4]: 4, datasetFolders[5]: 5}

# ---- End Important Globally Used Variables ----




# Writes the features to a CSV file.
# takes in a list of lists, i.e. [[1,2,3],[4,5,6]] and writes to a file with each feature vector in its own line
# Parameters  -
# featureVectorsArray: array of arrays with each second level array representing a feature vectors
# fileName: the name given to the file, this does not include the extension or directory, since that is assigned
# labels: the label for each column, in order from left to right. Taken in as a comma seperate list of strings:
#   ie. ['apples', 'orange', 'bananas']
# saves the file under the featureSets directory
def generateFeatureCsv(featureVectorsArray, fileName, labels):
    fileName = "../feature_sets/" + fileName + ".csv"
    featureVectorsCsv = open(fileName, "w+")
    featureVectorsCsv.write(cleanListString(labels) + '\n')
    for vector in featureVectorsArray:
        stringifiedVector = cleanListString(vector)
        featureVectorsCsv.write(stringifiedVector + '\n')

# Loads a CSV file as a Pandas dataframe
def retreiveDataSet(path):
    data = pd.read_csv(path)
    return pd.DataFrame(data)

# Turns a list into a comma seperated string
# Parameters:
# list: a list type
# returns: a string of comma seperated values
def cleanListString(list):
    return str(list).replace("[", "").replace("]", "").replace("\n", "")

# Reads in the entire dataset.
# Parameters:
# keepLabel: A boolean that toggles whether or not the original label will be kept (which age group it belongs to)
# returns: A list of lists. The sublists each represent an individual file
def readAgeDataSet(keepLabel):
    print("Reading files in: ", datasetFolders)
    print("Found the following raw dataset files:")
    dataSet = []
    for folder in datasetFolders:
        files = os.listdir(folder)
        for file in files:
            # This is here to be able to extract data by file, so that it can be read by file.
            csvFile = []
            fullName = os.path.join(folder, file)
            if os.path.isfile(fullName) and file.endswith('.csv'):
                with open(fullName, "r") as f:
                    for line in f:
                        row = cleanListString(line)
                        if keepLabel:
                            row = row + "," + str(dataSetLabels.get(str(folder)))
                        csvFile.append(row)
                dataSet.append(csvFile)
    return dataSet


# Generalized function to plot results. Iterates through ever cluster and
# puts the points on the scatterplot per cluster in order
# Parameters:
# X : the results of the kMeans clustering
# result:
# numberOfClusters: the number of clustered generated from KMeans
# labels = labels for this data. is defaulted to empty and only used when it is not empty to color data points
# this label indicates what age group it belonged to originally and can alll=ow for accuracy spotchecks
def plotClusteredResults(X, result, numberOfClusters, labels=[]):
    colors = ['lightgreen', 'orange', 'blue', 'yellow', 'orange']
    markers = ['s', 'p', '^', 'o', '*']
    if len(labels) > 0:
        color = colors[1] #TODO this color will have to come from the the label, perhaps iterate through all datapoints
    for cluster in range(1, numberOfClusters):
        plt.scatter(X[result == cluster-1, 1],
                    X[result == cluster-1, 2],
                    s=50, c=color,
                    marker=markers[cluster], edgecolor='black',
                    label='cluster ' + str(cluster))
    return plt




