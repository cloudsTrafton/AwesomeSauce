# Utility class that wraps KMeans and outputs the formatted experiment output

import time
from sklearn.metrics import f1_score


#Normalizes all data except the label column, since we want to leave that intact.
def normalizeLabeledData(data):
    result = data.copy()
    for feature in data.columns:
        if(feature != 'label' and feature != 'userId'):
            max_value = data[feature].max()
            min_value = data[feature].min()
            result[feature] = (data[feature] - min_value) / (max_value - min_value)
    return result


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


def getClusterBucketsForMultiClustering(feature_set, result):
    # These buckets will contain the feature vectors that were put in the corresponding cluster
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    cluster5 = []
    cluster6 = []

    # Since feature_set is a DataFrame, we get the values to treat it as an array
    feature_set_vals = feature_set.values

    # Iterate through all of the results and the features. Since the results are in order
    # of the data points, and is an array of the cluster that the datapoint in the same position
    # in the features, we can leverage this to fill the buckets.
    for i in range(0, len(result)):
        if result[i] == 1:
            cluster1.append(feature_set_vals[i])
        elif result[i] == 2:
            cluster2.append(feature_set_vals[i])
        elif result[i] == 3:
            cluster3.append(feature_set_vals[i])
        elif result[i] == 4:
            cluster4.append(feature_set_vals[i])
        elif result[i] == 5:
            cluster5.append(feature_set_vals[i])
        elif result[i] == 6:
            cluster6.append(feature_set_vals[i])

    return cluster1, cluster2, cluster3, cluster4, cluster5, cluster6



def getAveragesPerLabelForMultiClustering(clusterValues, clusterNumber, outputFileName):

    # use these to hold the percentages of each label per cluster
    percent_15_below = 0.0
    percent_16_19 = 0.0
    percent20_29 = 0.0
    percent30_39 = 0.0
    percent40_49 = 0.0
    percent50_plus = 0.0

    num_15_below = 0.0
    num_16_19 = 0.0
    num20_29 = 0.0
    num30_39 = 0.0
    num40_49 = 0.0
    num50_plus = 0.0

    #Number of features that we are looking at in this cluster
    numFeatureInCluster = len(clusterValues)
    for vector in clusterValues:
        label = vector[len(vector) - 1]
        if (label == 0):
            num_15_below += 1
        elif (label == 1):
            num_16_19 += 1
        elif (label == 2):
            num20_29 += 1
        elif (label == 3):
            num30_39 += 1
        elif (label == 4):
            num40_49 += 1
        elif (label == 5):
            num50_plus += 1

    numFeatureInCluster = float(numFeatureInCluster)
    print(numFeatureInCluster)

    if (numFeatureInCluster != 0.0):
        percent_15_below = num_15_below/numFeatureInCluster
    if(numFeatureInCluster != 0.0):
        percent_16_19 = num_16_19/numFeatureInCluster
    if(numFeatureInCluster != 0.0):
        percent20_29 = num20_29/numFeatureInCluster
    if(numFeatureInCluster != 0.0):
        percent30_39 = num30_39/numFeatureInCluster
    if(numFeatureInCluster != 0.0):
        percent40_49 = num40_49/numFeatureInCluster
    if(numFeatureInCluster != 0.0):
        percent50_plus = num50_plus/numFeatureInCluster

    fileName = "../experiments/experiment_" + outputFileName + "_" + ".out"

    with open(fileName, "a") as file:
        file.write("\n")
        file.write("Results for " + clusterNumber + "\n")
        file.write("Number of features in this cluster: " + str(numFeatureInCluster) + "\n")
        file.write("Percent and amount of each label in cluster " + clusterNumber + "\n")
        file.write("15 below: " + "\n")
        file.write("percent_15_below: " + str(percent_15_below) + "\n")
        file.write("num_15_below: " + str(num_15_below) + "\n")

        file.write("16-19: " + "\n")
        file.write("percent_16_19: " + str(percent_16_19) + "\n")
        file.write("num_16_19: " + str(num_16_19) + "\n")

        file.write("20-29: " + "\n")
        file.write("percent20_29: " + str(percent20_29) + "\n")
        file.write("num20_29: " + str(num20_29) + "\n")

        file.write("30-39: " + "\n")
        file.write("percent30_39: " + str(percent30_39) + "\n")
        file.write("num30_39: " + str(num30_39) + "\n")

        file.write("40-49: " + "\n")
        file.write("percent40_49: " + str(percent40_49) + "\n")
        file.write("num40_49: " + str(num40_49) + "\n")

        file.write("50+: " + "\n")
        file.write("percent50_plus: " + str(percent50_plus) + "\n")
        file.write("num50_plus: " + str(num50_plus) + "\n")

        file.write("Highest percentage in this cluster: ")
        file.write(str(max(percent_15_below,percent_16_19,percent20_29,percent30_39,percent40_49,percent50_plus)))
        file.write("\n")


def getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, outputFileName):
    getAveragesPerLabelForMultiClustering(cluster1, "Cluster 1", outputFileName)
    getAveragesPerLabelForMultiClustering(cluster2, "Cluster 2", outputFileName)
    getAveragesPerLabelForMultiClustering(cluster3, "Cluster 3", outputFileName)
    getAveragesPerLabelForMultiClustering(cluster4, "Cluster 4", outputFileName)
    getAveragesPerLabelForMultiClustering(cluster5, "Cluster 5", outputFileName)
    getAveragesPerLabelForMultiClustering(cluster6, "Cluster 6", outputFileName)
