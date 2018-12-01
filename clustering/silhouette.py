import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from data_processing import dataUtils
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy import stats
import pandas as pd
import data_processing.MulticlusteringExperimentUtils as expUtils
import numpy as np
from data_processing.dataUtils import getColumnZScores, removeOutliersByZScore

def makeSilhouettePlot(featureSet, kmeansOutput, experimentName = ''):
    from datetime import datetime, date, time

    cluster_labels = np.unique(kmeansOutput)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(featureSet, kmeansOutput, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[kmeansOutput == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper),
                 c_silhouette_vals,
                 height=1.0,
                 edgecolor='none',
                 color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg,
                color="red",
                linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette Coefficient: ' + str(silhouette_avg))

    if experimentName == '':
        experimentName = datetime.now().strftime("%m_%d_%y_%H_%M_%S_%f")

    plt.savefig("../silhouettes/" + experimentName + ".png", dpi=600)
    # plt.show() TODO put back in
    return plt

# run with test data

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Multi-clustering with labeled feature vectors

# Retrieve the processed data set
feature_set = dataUtils.retreiveDataSet("../feature_sets/jonstest7.csv")

#Drop the label and ID column, since we dont want to include these in the clustering algorithm.
feature_set_copy = feature_set
feature_set_copy.drop(columns=['label'])
feature_set_copy.drop(columns=['userID'])

#Normalize the data using minMax scalers
feature_set_copy = feature_set_copy.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()
feature_set_copy = min_max_scaler.fit_transform(feature_set_copy.values)

# Cluster for K-Means
kmeans = KMeans(init='k-means++', n_clusters=6, n_init=120, max_iter=500, tol=1e-04, random_state=1)
result = kmeans.fit_predict(feature_set_copy)
for res in result:
    print(res)