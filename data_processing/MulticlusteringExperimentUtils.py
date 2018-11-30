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
        if result[i] == 0:
            cluster1.append(feature_set_vals[i])
        elif result[i] == 1:
            cluster2.append(feature_set_vals[i])
        elif result[i] == 2:
            cluster3.append(feature_set_vals[i])
        elif result[i] == 3:
            cluster4.append(feature_set_vals[i])
        elif result[i] == 4:
            cluster5.append(feature_set_vals[i])
        elif result[i] == 5:
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
    numDataPoints = len(clusterValues)
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

    numDataPoints = float(numDataPoints)
    print(numDataPoints)

    if (numDataPoints != 0.0):
        percent_15_below = num_15_below/numDataPoints
    if(numDataPoints != 0.0):
        percent_16_19 = num_16_19/numDataPoints
    if(numDataPoints != 0.0):
        percent20_29 = num20_29/numDataPoints
    if(numDataPoints != 0.0):
        percent30_39 = num30_39/numDataPoints
    if(numDataPoints != 0.0):
        percent40_49 = num40_49/numDataPoints
    if(numDataPoints != 0.0):
        percent50_plus = num50_plus/numDataPoints
        
        
    # calculate the percentage of points from each group of the dataset in the cluster
        
    percent_15_below_ds, percent_16_19_ds, percent20_29_ds, percent30_39_ds, percent40_49_ds, percent50_plus_ds = \
        calculatePercentOfDataPointsInCluster(num_15_below, num_16_19, num20_29, num30_39, num40_49, num50_plus)
    

    fileName = "../experiments/experiment_" + outputFileName + "_" + ".out"

    with open(fileName, "a") as file:
        file.write("\n")
        file.write("Results for " + clusterNumber + "\n")
        file.write("Number of data points in this cluster: " + str(numDataPoints) + "\n")
        file.write("Percent and amount of each label in cluster " + clusterNumber + "\n")
        file.write("15 below: " + "\n")
        file.write("percent in cluster 15_below: " + str(percent_15_below) + "\n")
        file.write("num_15_below: " + str(num_15_below) + "\n")
        file.write("percent of all 15_below data points in this cluster: " + str(percent_15_below_ds) + "\n")

        file.write("16-19: " + "\n")
        file.write("percent in cluster 16_19: " + str(percent_16_19) + "\n")
        file.write("num_16_19: " + str(num_16_19) + "\n")
        file.write("percent of all 16_19 data points in this cluster: " + str(percent_16_19_ds) + "\n")

        file.write("20-29: " + "\n")
        file.write("percent in cluster 20_29: " + str(percent20_29) + "\n")
        file.write("num20_29: " + str(num20_29) + "\n")
        file.write("percent of all 20_19 data points in this cluster: " + str(percent20_29_ds) + "\n")

        file.write("30-39: " + "\n")
        file.write("percent in cluster 30_39: " + str(percent30_39) + "\n")
        file.write("num30_39: " + str(num30_39) + "\n")
        file.write("percent of all 30_39 data points in this cluster: " + str(percent30_39_ds) + "\n")

        file.write("40-49: " + "\n")
        file.write("percent in cluster 40_49: " + str(percent40_49) + "\n")
        file.write("num40_49: " + str(num40_49) + "\n")
        file.write("percent of all 40_49 data points in this cluster: " + str(percent40_49_ds) + "\n")

        file.write("50+: " + "\n")
        file.write("percent in cluster 50_plus: " + str(percent50_plus) + "\n")
        file.write("num50_plus: " + str(num50_plus) + "\n")
        file.write("percent of all 50_plus data points in this cluster: " + str(percent50_plus_ds) + "\n")

        file.write("Highest percentage in this cluster: ")
        file.write(str(max(percent_15_below,percent_16_19,percent20_29,percent30_39,percent40_49,percent50_plus)))
        file.write("\n")


def getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, outputFileName, results):
    getAveragesPerLabelForMultiClustering(cluster1, "Cluster 1", outputFileName)
    getAveragesPerLabelForMultiClustering(cluster2, "Cluster 2", outputFileName)
    getAveragesPerLabelForMultiClustering(cluster3, "Cluster 3", outputFileName)
    getAveragesPerLabelForMultiClustering(cluster4, "Cluster 4", outputFileName)
    getAveragesPerLabelForMultiClustering(cluster5, "Cluster 5", outputFileName)
    getAveragesPerLabelForMultiClustering(cluster6, "Cluster 6", outputFileName)

    #save the clustering results for this experiment
    fileName = "../experiments/experiment_" + outputFileName + "_" + "clusters.out"
    with open(fileName, "w") as file:
        for cluster in results:
            file.write(str(cluster) + '\n')


# calculate the percent of  particular age group within the cluster.
# Returns the percent of each type from the dataset that is present in the given cluster.
def calculatePercentOfDataPointsInCluster(num_15_below, num_16_19, num20_29, num30_39, num40_49, num50_plus):

    #These were extracted from Pentel's paper and confirmed via the DataSet
    num_15_below_dataset = 376
    num_16_19_dataset = 258
    num_20_29_dataset = 357
    num_30_39_dataset = 1327
    num_40_49_dataset = 3213
    num_50_plus_dataset = 1588

    # use these to hold the percentages of each label per dataset
    percent_15_below_ds = num_15_below / num_15_below_dataset
    percent_16_19_ds = num_16_19 / num_16_19_dataset
    percent20_29_ds = num20_29 / num_20_29_dataset
    percent30_39_ds = num30_39 / num_30_39_dataset
    percent40_49_ds = num40_49 / num_40_49_dataset
    percent50_plus_ds = num50_plus / num_50_plus_dataset

    return percent_15_below_ds, percent_16_19_ds, percent20_29_ds, percent30_39_ds, percent40_49_ds, percent50_plus_ds



