# Utility class that wraps KMeans and outputs the formatted experiment output

import time
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np


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

# labels: the original labels of the datapoints
# the resulting clusters
def getClusterBucketsForMultiClustering(labels, result):
    # These buckets will contain the feature vectors that were put in the corresponding cluster
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    cluster5 = []
    cluster6 = []

    # Since feature_set is a DataFrame, we get the values to treat it as an array
    labels_vals = labels.values

    # Iterate through all of the results and the features. Since the results are in order
    # of the data points, and is an array of the cluster that the datapoint in the same position
    # in the features, we can leverage this to fill the buckets.
    for i in range(0, len(result)):
        if result[i] == 0:
            cluster1.append(labels_vals[i])
        elif result[i] == 1:
            cluster2.append(labels_vals[i])
        elif result[i] == 2:
            cluster3.append(labels_vals[i])
        elif result[i] == 3:
            cluster4.append(labels_vals[i])
        elif result[i] == 4:
            cluster5.append(labels_vals[i])
        elif result[i] == 5:
            cluster6.append(labels_vals[i])

    return cluster1, cluster2, cluster3, cluster4, cluster5, cluster6

def getF1Score(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, experimentName):

    if(len(cluster1) != 0):
        cluster1_mostFreq = max(cluster1, key=cluster1.count)
        exp_cluster1 = []
        for item in cluster1:
            exp_cluster1.append(cluster1_mostFreq)
        fscore_cluster1 = f1_score(exp_cluster1, cluster1, average='weighted')
    else:
        fscore_cluster1 = 0

    if(len(cluster2) != 0):
        cluster2_mostFreq = max(cluster2, key=cluster2.count)
        exp_cluster2 = []
        for item in cluster2:
            exp_cluster2.append(cluster2_mostFreq)
        fscore_cluster2 = f1_score(exp_cluster2, cluster2, average='weighted')
    else:
        fscore_cluster2 = 0

    if (len(cluster3) != 0):
        cluster3_mostFreq = max(cluster3, key=cluster3.count)
        exp_cluster3 = []
        for item in cluster3:
            exp_cluster3.append(cluster3_mostFreq)
        fscore_cluster3 = f1_score(exp_cluster3, cluster3, average='weighted')
    else:
        fscore_cluster3 = 0

    if (len(cluster4) != 0):
        cluster4_mostFreq = max(cluster4, key=cluster4.count)
        exp_cluster4 = []
        for item in cluster4:
            exp_cluster4.append(cluster4_mostFreq)
        fscore_cluster4 = f1_score(exp_cluster4, cluster4, average='weighted')
    else:
        fscore_cluster4 = 0

    if (len(cluster5) != 0):
        cluster5_mostFreq = max(cluster5, key=cluster5.count)
        exp_cluster5 = []
        for item in cluster5:
            exp_cluster5.append(cluster5_mostFreq)
        fscore_cluster5 = f1_score(exp_cluster5, cluster5, average='weighted')
    else:
        fscore_cluster5 = 0

    if (len(cluster6) != 0):
        cluster6_mostFreq = max(cluster6, key=cluster6.count)
        exp_cluster6 = []
        for item in cluster6:
            exp_cluster6.append(cluster6_mostFreq)
        fscore_cluster6 = f1_score(exp_cluster6, cluster6, average='weighted')
    else:
        fscore_cluster6 = 0


    clusters = 6

    avgFScore = (fscore_cluster1 + fscore_cluster2 + fscore_cluster3 +
                 fscore_cluster4 + fscore_cluster5 + fscore_cluster6) / float(clusters)


    output = open("../PrecisionRecallScores/F1_Scores.csv", "a")

    # write the header
    output.write("\n" + str(experimentName) + "," + str(fscore_cluster1)+  "," +str(fscore_cluster2)+  "," +str(fscore_cluster3)+  "," +str(fscore_cluster4)+  "," +str(fscore_cluster5)+  "," +str(fscore_cluster6)+  "," +str(avgFScore))




def getAveragesPerLabelForMultiClustering(clusterValues, clusterNumber, outputFileName, labels):

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

    labels = labels.values

    #Number of features that we are looking at in this cluster
    numDataPoints = len(clusterValues)
    for label in clusterValues:
        label = int(label)
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
        calculatePercentOfDataPointsInCluster(labels, num_15_below, num_16_19, num20_29, num30_39, num40_49,
                                              num50_plus)

    output_str1 = getOutputStr(clusterNumber,"15 below", numDataPoints, percent_15_below,
                   num_15_below,percent_15_below_ds)
    output_str2 = getOutputStr(clusterNumber,"16-19", numDataPoints, percent_16_19,
                               num_16_19,percent_16_19_ds)
    output_str3 = getOutputStr(clusterNumber,"20-29", numDataPoints, percent20_29,
                               num20_29,percent20_29_ds)
    output_str4 = getOutputStr(clusterNumber,"30-39", numDataPoints, percent30_39,
                               num30_39,percent30_39_ds)
    output_str5 = getOutputStr(clusterNumber,"40-49", numDataPoints, percent40_49,
                               num40_49,percent40_49_ds)
    output_str6 = getOutputStr(clusterNumber,"50+", numDataPoints, percent50_plus,
                               num50_plus,percent50_plus_ds)
    output = output_str1 + output_str2 + output_str3 + output_str4 + output_str5 + output_str6

    return output



def getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, outputFileName, labels):
    output_str1 = getAveragesPerLabelForMultiClustering(cluster1, "Cluster 1", outputFileName, labels)
    output_str2 = getAveragesPerLabelForMultiClustering(cluster2, "Cluster 2", outputFileName, labels)
    output_str3 = getAveragesPerLabelForMultiClustering(cluster3, "Cluster 3", outputFileName, labels)
    output_str4 = getAveragesPerLabelForMultiClustering(cluster4, "Cluster 4", outputFileName, labels)
    output_str5 = getAveragesPerLabelForMultiClustering(cluster5, "Cluster 5", outputFileName, labels)
    output_str6 = getAveragesPerLabelForMultiClustering(cluster6, "Cluster 6", outputFileName, labels)

    fileName = "../experiments/experiment_" + outputFileName + "_" + ".csv"

    with open(fileName, "w+") as file:
        file.write("Cluster,label,totalDataPoints,PercentLabelInCluster, "
                   "numberOfDataPointsPerLabelInCluster,percentOfLabelTypeInCluster" + "\n")
        file.write(output_str1)
        file.write("\n")
        file.write(output_str2)
        file.write("\n")
        file.write(output_str3)
        file.write("\n")
        file.write(output_str4)
        file.write("\n")
        file.write(output_str5)
        file.write("\n")
        file.write(output_str6)
        file.write("\n")



def getOutputStr(clusterNumber, label, numDataPoints, percent_cluster, num_data, percent_data_set):
    output = str(clusterNumber) + "," + str(label) + "," + str(numDataPoints) + "," + str(percent_cluster) + "," + str(num_data)+ "," + str(percent_data_set) + '\n'
    return output



# calculate the percent of  particular age group within the cluster.
# Returns the percent of each type from the dataset that is present in the given cluster.
def calculatePercentOfDataPointsInCluster(labels, num_15_below, num_16_19, num20_29, num30_39,
                                          num40_49, num50_plus):

    num_15_below_dataset = 0.0
    num_16_19_dataset = 0.0
    num_20_29_dataset = 0.0
    num_30_39_dataset = 0.0
    num_40_49_dataset = 0.0
    num_50_plus_dataset = 0.0

    percent_15_below_ds = 0
    percent_16_19_ds = 0
    percent20_29_ds = 0
    percent30_39_ds = 0
    percent40_49_ds = 0
    percent50_plus_ds = 0


    for label in labels:
        label = int(label)
        if (label == 0):
            num_15_below_dataset += 1
        elif (label == 1):
            num_16_19_dataset += 1
        elif (label == 2):
            num_20_29_dataset += 1
        elif (label == 3):
            num_30_39_dataset += 1
        elif (label == 4):
            num_40_49_dataset += 1
        elif (label == 5):
            num_50_plus_dataset += 1

    # use these to hold the percentages of each label per dataset
    if(num_15_below_dataset != 0):
        percent_15_below_ds = num_15_below / num_15_below_dataset
    if(num_16_19_dataset != 0):
        percent_16_19_ds = num_16_19 / num_16_19_dataset
    if(num_20_29_dataset != 0):
        percent20_29_ds = num20_29 / num_20_29_dataset
    if(num_30_39_dataset != 0):
        percent30_39_ds = num30_39 / num_30_39_dataset
    if (num_40_49_dataset != 0):
        percent40_49_ds = num40_49 / num_40_49_dataset
    if (num_50_plus_dataset != 0):
        percent50_plus_ds = num50_plus / num_50_plus_dataset

    return percent_15_below_ds, percent_16_19_ds, percent20_29_ds, percent30_39_ds, percent40_49_ds, percent50_plus_ds



