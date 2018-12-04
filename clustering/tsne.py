from sklearn.manifold import TSNE
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
import time
#from ggplot import *

def makeTSNEPlot(featureSet, numClusters=6, experimentName = '', tsne_perplexity=30, tsne_iter=1000):
    from datetime import datetime, date, time

    # Run TSNE, dropping label and cluster columns.
    # use color as cluster, marker as age group
    tsne = TSNE(n_components=2, verbose=1, perplexity=tsne_perplexity, n_iter=tsne_iter)
    featureSetStripped = featureSet.drop(columns=['label']).drop(columns=['cluster']).copy()
    tsne_result = tsne.fit_transform(featureSetStripped.values)

    df_tsne = featureSet.copy()
    df_tsne['x_tsne'] = tsne_result[:, 0]
    df_tsne['y_tsne'] = tsne_result[:, 1]
    df_tsne['label'] = featureSet['label'].astype(int)
    df_tsne['cluster'] = featureSet['cluster'].astype(int)

    print(df_tsne)

    markers = {0 : '+', 1 : 's', 2 : '.', 3 : '*', 4 : 'v', 5 : 'D'}
    #markers = {0: '+', 1: 's', 2: '.'}
    clustercolors = {0 : 'red', 1 : 'green', 2 : 'blue', 3 : 'pink', 4 : 'purple', 5 : 'grey'}
    #clustercolors = {0 : 'red', 1 : 'green', 2 : 'blue'}
    plots = [[[] for i in range(6)] for i in range(6)]
    #plots = [[[] for i in range(3)] for i in range(3)]

    for age_group in markers:
        print(" scattering age group ", age_group, " with marker: ", markers[age_group])
        for cluster_num in clustercolors:
            print("  scattering cluster ", cluster_num, " with color: ", clustercolors[cluster_num])
            d = df_tsne[(df_tsne.label == age_group) & (df_tsne.cluster == cluster_num)]
            plots[age_group][cluster_num] = plt.scatter(d.x_tsne, d.y_tsne,
                                s = 30,
                                c = clustercolors[cluster_num],
                                marker=markers[age_group], edgecolors="black")

    #legend1 = plt.legend((plots[0][0], plots[1][0], plots[2][0]), ('<19', '20-39', '40+'), loc=1)
    legend1 = plt.legend((plots[0][0], plots[1][0], plots[2][0], plots[3][0], plots[4][0], plots[5][0]), ('<15', '15-19', '20-29', '30-39', '40-49', '50+'), loc=1)
    #plt.legend((plots[1][0], plots[1][1], plots[1][2]), ('Cluster 1', 'Cluster 2', 'Cluster 3'), loc=4)
    plt.legend((plots[1][0], plots[1][1], plots[1][2], plots[1][3], plots[1][4], plots[1][5]), ('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6'), loc=4)
    plt.gca().add_artist(legend1)
    #plt.legend()

    if experimentName == '':
        experimentName = datetime.now().strftime("%m_%d_%y_%H_%M_%S_%f")

    plt.savefig("./tsne/" + experimentName + ".png", dpi=600)
    plt.close()

    # plt.show()

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Multi-clustering with labeled feature vectors

# Retrieve the processed data set
feature_set = dataUtils.retreiveDataSet("./feature_sets/jonstest9-3classes-allsamples.csv")

#Drop the label and ID column, since we dont want to include these in the clustering algorithm.
feature_set_stripped = feature_set.copy()
#make a copy of the original label to add back in after kmeans
#feature_set_tsne = feature_set.copy()

feature_set_stripped = feature_set_stripped.drop(columns=['num']).drop(columns=['userID']).drop(columns=['label'])

normalizedUnlabeledData = expUtils.normalizeLabeledData(pd.DataFrame(feature_set_stripped))
normalizedUnlabeledData = normalizedUnlabeledData.astype(float)

print(normalizedUnlabeledData)

# Cluster for K-Means
kmeans = KMeans(init='random', n_clusters=6, n_init=120, max_iter=500, tol=1e-04, random_state=1)
cluster_result = kmeans.fit_predict(normalizedUnlabeledData)

# cluster_out = feature_set_stripped.copy()
# cluster_out['label'] = feature_set['label']
# cluster_out['cluster'] = cluster_result
# cluster_out.to_csv('kmeans_out.csv')
normalizedUnlabeledData['label'] = feature_set['label']
normalizedUnlabeledData['cluster'] = cluster_result.copy()
makeTSNEPlot(normalizedUnlabeledData.copy(), '')
