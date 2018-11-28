# Multi-clustering experiments utilities.

from data_processing import dataUtils
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy import stats
import pandas as pd
import data_processing.MulticlusteringExperimentUtils as expUtils
import numpy as np

from data_processing.dataUtils import getColumnZScores, removeOutliersByZScore

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# # Multi-clustering with labeled feature vectors
#
# # Retrieve the processed data set
feature_set = dataUtils.retreiveDataSet("../feature_sets/jonstest7.csv")

#Drop the label and ID column, since we dont want to include these in the clustering algorithm.
feature_set_copy = feature_set
feature_set_copy.drop(columns=['label'])
feature_set_copy.drop(columns=['userID'])


#Normalize the data using minMax scalers
feature_set_copy = feature_set_copy.astype(float)
#
# min_max_scaler = preprocessing.MinMaxScaler()
# feature_set_copy = min_max_scaler.fit_transform(feature_set_copy.values)
#
# # Cluster for K-Means
kmeans = KMeans(init='k-means++', n_clusters=7, n_init=120, max_iter=500, tol=1e-04, random_state=1)
result = kmeans.fit_predict(feature_set_copy)


# --- EXPERIMENT 2 --- #
# Removing outliers based on Z-Score, using jonstest7 data set.

feature_set_copy1 = feature_set
feature_set_copy1 = feature_set_copy1.drop(columns=['userID']).drop(columns=['num'])

normalizedLabeledData = expUtils.normalizeLabeledData(pd.DataFrame(feature_set_copy1))
normalizedLabeledData = normalizedLabeledData.astype(float)



# average hold time
feature = 'avgSeekTime'
z_scored = getColumnZScores(pd.DataFrame(normalizedLabeledData), feature)
outliers_removed = removeOutliersByZScore(z_scored, feature, 3)

#run experiment with this dataset
kmeans_2 = KMeans(init='k-means++', n_clusters=6, n_init=120, max_iter=800, tol=1e-04, random_state=1)
result_outliers_removed = kmeans.fit_predict(z_scored)

cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_outliers_removed)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          "jonstest7_2_outliers_removed_2" + feature, result_outliers_removed)

#---------------------------------------------------------------------------
# average hold time
feature = 'avgHoldTime'
z_scored = getColumnZScores(pd.DataFrame(normalizedLabeledData), feature)
outliers_removed = removeOutliersByZScore(z_scored, feature, 3)

#run experiment with this dataset
kmeans_2 = KMeans(init='k-means++', n_clusters=6, n_init=120, max_iter=800, tol=1e-04, random_state=1)
result_outliers_removed = kmeans.fit_predict(z_scored)

cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_outliers_removed)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          "jonstest7_2_outliers_removed_2" + feature, result_outliers_removed)

#---------------------------------------------------------------------------
# average hold time
feature = 'avgHoldTime'
z_scored = getColumnZScores(pd.DataFrame(normalizedLabeledData), feature)
outliers_removed = removeOutliersByZScore(z_scored, feature, 3)

#run experiment with this dataset
kmeans_2 = KMeans(init='k-means++', n_clusters=6, n_init=120, max_iter=800, tol=1e-04, random_state=1)
result_outliers_removed = kmeans.fit_predict(z_scored)

cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_outliers_removed)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          "jonstest7_2_outliers_removed_2" + feature, result_outliers_removed)

#---------------------------------------------------------------------------
# average hold time
feature = 'averageNgramTime'
z_scored = getColumnZScores(pd.DataFrame(normalizedLabeledData), feature)
outliers_removed = removeOutliersByZScore(z_scored, feature, 3)

#run experiment with this dataset
kmeans_2 = KMeans(init='k-means++', n_clusters=6, n_init=120, max_iter=800, tol=1e-04, random_state=1)
result_outliers_removed = kmeans.fit_predict(z_scored)

cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_outliers_removed)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          "jonstest7_2_outliers_removed_2" + feature, result_outliers_removed)

#---------------------------------------------------------------------------
# average hold time
feature = 'LA'
z_scored = getColumnZScores(pd.DataFrame(normalizedLabeledData), feature)
outliers_removed = removeOutliersByZScore(z_scored, feature, 3)

#run experiment with this dataset
kmeans_2 = KMeans(init='k-means++', n_clusters=6, n_init=120, max_iter=800, tol=1e-04, random_state=1)
result_outliers_removed = kmeans.fit_predict(z_scored)

cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_outliers_removed)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          "jonstest7_2_outliers_removed_2" + feature, result_outliers_removed)

# --- Remove all of the outliers for the big features ----
# average hold time
feature1 = 'avgSeekTime'
feature2 = 'avgHoldTime'
feature3 = 'averageNgramTime'
z_scored = getColumnZScores(pd.DataFrame(normalizedLabeledData), feature1)
z_scored = getColumnZScores(pd.DataFrame(z_scored), feature2)
z_scored = getColumnZScores(pd.DataFrame(z_scored), feature3)
z_scored_outliers_removed = removeOutliersByZScore(z_scored, feature1, 3)
z_scored_outliers_removed = removeOutliersByZScore(z_scored_outliers_removed, feature2, 3)
z_scored_outliers_removed = removeOutliersByZScore(z_scored_outliers_removed, feature3, 3)
print(len(z_scored_outliers_removed))

result_removed_outliers = kmeans.fit_predict(z_scored_outliers_removed)

cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_removed_outliers)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          "jonstest7_2_outliers_removed_large_features", z_scored_outliers_removed)

kmeans_2_clusters = KMeans(init='k-means++', n_clusters=3, n_init=120, max_iter=800, tol=1e-04, random_state=1)
result_removed_outliers_2_clusters = kmeans_2_clusters.fit_predict(z_scored_outliers_removed)


cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_removed_outliers_2_clusters)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          "jonstest7_2_outliers_removed_large_features_2_clusters", z_scored_outliers_removed)







