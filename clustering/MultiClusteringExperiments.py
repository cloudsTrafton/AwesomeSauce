# Multi-clustering experiments

from data_processing import dataUtils
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy import stats
import data_processing.experimentUtils as expUtils
import numpy as np


# Multi-clustering with labeled feature vectors

# Retrieve the processed data set
feature_set = dataUtils.retreiveDataSet("../feature_sets/jonstest7.csv")
print(feature_set.keys())

keys = ['\'userID\'', ' \'avgHoldTime\'', ' \'avgSeekTime\'', ' \'averageNgramTime\'',
       ' \'I \'', ' \'AL\'', ' \'S \'', ' \'KS\'', ' \'EI\'', ' \'D \'', ' \'AK\'', ' \'L \'',
       ' \' O\'', ' \'LE\'', ' \'MA\'', ' \'IN\'', ' \'SI\'', ' \'EL\'', ' \'E \'', ' \'JA\'',
       ' \'LA\'', ' \'label\'']

#reduce using z-scores
feature_set_copy = feature_set
feature_set_copy.drop(columns=['label'])
feature_set_copy.drop(columns=['userID'])


#Normalize the data using minMax scalers
feature_set_copy = feature_set_copy.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()
feature_set_copy = min_max_scaler.fit_transform(feature_set_copy.values)

# Cluster for K-Means
kmeans = KMeans(init='k-means++', n_clusters=6, n_init=120, max_iter=500, tol=1e-04, random_state=0)
result = kmeans.fit_predict(feature_set_copy)
print(result)


cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, "jonstest7")