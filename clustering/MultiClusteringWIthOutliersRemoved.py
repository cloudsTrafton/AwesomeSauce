# Keep the clustering experiments that involve outliers here
from clustering.KMeansVariations import kMeans_baseline, kMeans_baseline_high_iteration, kMeans_baseline_random_init, \
    kMeans_baseline_4_clusters, kMeans_baseline_3_clusters
from clustering.MultiClusteringExperiments import normalizedLabeledData
from data_processing import MulticlusteringExperimentUtils as expUtils
from data_processing import CleanDataUtils
from data_processing import dataUtils
from clustering import silhouette as sil
import pandas as pd



# --- Remove all of the outliers for the big features ----
# average hold time
from data_processing.CleanDataUtils import feature_set
from data_processing.dataUtils import getColumnZScores, removeOutliersByZScore

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


#-- Run experiments and then generate the silhouettes ----

# Base K-Means Run
experiment_name = "jonstest7_Kmeans_baseline_outliers_removed"

result_kmeans_baseline = kMeans_baseline.fit_predict(z_scored_outliers_removed)

cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_kmeans_baseline)

sil.makeSilhouettePlot(z_scored_outliers_removed, result_kmeans_baseline, experiment_name)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          experiment_name, z_scored_outliers_removed)

#---------------------------------------------------------------------------------------

# K-Means with high number of iterations

experiment_name =  "jonstest7_Kmeans_high_iters_outliers_removed"

result_kmeans_high_iters = kMeans_baseline_high_iteration.fit_predict(z_scored_outliers_removed)

cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_kmeans_high_iters)

sil.makeSilhouettePlot(z_scored_outliers_removed, result_kmeans_high_iters, experiment_name)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          experiment_name, z_scored_outliers_removed)

#---------------------------------------------------------------------------------------

# K-Means with random initialization

experiment_name = "jonstest7_Kmeans_random_init_outliers_removed"

result_kmeans_random_init = kMeans_baseline_random_init.fit_predict(z_scored_outliers_removed)

cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_kmeans_random_init)

sil.makeSilhouettePlot(z_scored_outliers_removed, result_kmeans_random_init, experiment_name)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          experiment_name, z_scored_outliers_removed)

#---------------------------------------------------------------------------------------

# K-Means with random initialization - again

experiment_name = "jonstest7_Kmeans_random_init_outliers_removed_2"

result_kmeans_random_init = kMeans_baseline_random_init.fit_predict(z_scored_outliers_removed)

cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_kmeans_random_init)

sil.makeSilhouettePlot(z_scored_outliers_removed, result_kmeans_random_init, experiment_name)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          experiment_name, z_scored_outliers_removed)


#---------------------------------------------------------------------------------------

# K-Means with the same as the baseline but with 4 clusters

experiment_name = "jonstest7_Kmeans_baseline_4_clusters_outliers_removed"

result_kmeans_4_clusters = kMeans_baseline_4_clusters.fit_predict(z_scored_outliers_removed)

cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_kmeans_4_clusters)

sil.makeSilhouettePlot(z_scored_outliers_removed, result_kmeans_4_clusters, experiment_name)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          experiment_name, z_scored_outliers_removed)


#---------------------------------------------------------------------------------------

# K-Means with the same as the baseline but with 3 clusters

experiment_name = "jonstest7_Kmeans_baseline_3_clusters_outliers_removed"

result_kmeans_3_clusters = kMeans_baseline_3_clusters.fit_predict(z_scored_outliers_removed)

cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
    expUtils.getClusterBucketsForMultiClustering(feature_set, result_kmeans_3_clusters)

sil.makeSilhouettePlot(z_scored_outliers_removed, result_kmeans_3_clusters, experiment_name)

expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                          experiment_name, z_scored_outliers_removed)

