import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os


# Provides utilities for processing data, including.
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
    return data

# Turns a list into a comma seperated string
# Parameters:
# list: a list type
# returns: a string of comma seperated values
def cleanListString(list):
    return str(list).replace("[", "").replace("]", "").replace("\n", "")



# Reads in the dataset from the
def readAgeDataSet(keepLabel):
    print("Reading files in: ", datasetFolders)
    print("Found the following raw dataset files:")
    dataSet = []
    for folder in datasetFolders:
        files = os.listdir(folder)
        for file in files:
            with open(folder + file, "r") as f:
                for line in f:
                    row = cleanListString(line)
                    if keepLabel:
                        row = row + "," + str(dataSetLabels.get(str(folder)))
                    dataSet.append(row)
        print(dataSet)
    return dataSet

# Plots data using seaborn and matplotlib. Wraps this utilitiy to a generalized method.
# def plotData(X):
#     plt.scatter(X[result == 0, 0],
#                 X[result == 0, 1],
#                 s=50, c='lightgreen',
#                 marker='s', edgecolor='black',
#                 label='cluster 1')
#     plt.scatter(X[result == 1, 0],
#                 X[result == 1, 1],
#                 s=50, c='orange',
#                 marker='o', edgecolor='black',
#                 label='cluster 2')
#     plt.scatter(X[result == 2, 0],
#                 X[result == 2, 1],
#                 s=50, c='blue',
#                 marker='v', edgecolor='black',
#                 label='cluster 3')
#     plt.scatter(X[result == 3, 0],
#                 X[result == 3, 1],
#                 s=50, c='yellow',
#                 marker='D', edgecolor='black',
#                 label='cluster 4')
#     plt.scatter(X[result == 4, 0],
#                 X[result == 4, 1],
#                 s=50, c='orange',
#                 marker='h', edgecolor='black',
#                 label='cluster 5')

readAgeDataSet(False)


