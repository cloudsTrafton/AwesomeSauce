import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# Provides utilities for processing data, including.
# Authored: 11/17/2019


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
    print(data)
    return data

# Turns a list into a comma seperated string
# Parameters:
# list: a list type
# returns: a string of comma seperated values
def cleanListString(list):
    return str(list).replace("[", "").replace("]", "")

# Plots data using seaborn and matplotlib. Wraps this utilitiy to a generalized method.
def plotData(xLabel, yLabel, res):
    sb.lmplot(xLabel, yLabel,
               data=res,
               fit_reg=False,
               hue="clusters")

     # Scatter plot
    plt.title("Clusters " + xLabel + " vs " + yLabel)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
