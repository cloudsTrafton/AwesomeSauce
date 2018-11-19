# Utility class that wraps KMeans and outputs the formatted experiment output

import time
from sklearn.metrics import f1_score

# Prints the experiment data as well as calculates the F-Scores and prints them to the file
#
# Parameters:
# kMeansResults: the results of running K-Means, which are the clusters each data point belongs to
#
def writeExperimentData(kMeansResults, initialLabels, featuresUsed, dataSetName):
    path = '../experiments/'
    fileName = "experiment_" + dataSetName + "_" + str(time.time()) + ".out"
    file = open(fileName, "w+")

    #TODO allow this to be passed in
    fScore = f1_score(kMeansResults, initialLabels, average='micro')

    file.writelines("Results from Experiment for : " + dataSetName)
    file.writelines("Features used: ")
    file.writelines(str(featuresUsed))
    file.writelines("F Score: " + str(fScore))
