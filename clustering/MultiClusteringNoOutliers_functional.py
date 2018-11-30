from clustering import silhouette as sil
from data_processing import MulticlusteringExperimentUtils as expUtils
# Keep the clustering experiments that involve outliers here
from clustering.KMeansVariations import kMeans_baseline, kMeans_baseline_high_iteration, kMeans_baseline_random_init, \
    kMeans_baseline_4_clusters, kMeans_baseline_3_clusters, kMeans_baseline_2_clusters, kMeans_baseline_2_clusters_low_iter,\
    kMeans_baseline_2_clusters_high_iter


import pandas as pd

# --- Remove all of the outliers for the big features ----
# average hold time
from data_processing.CleanDataUtils import feature_set
from data_processing.dataUtils import getColumnZScores, removeOutliersByZScore
feature_set_copy1 = feature_set
normalizedLabeledData = expUtils.normalizeLabeledData(pd.DataFrame(feature_set_copy1))
normalizedLabeledData = normalizedLabeledData.astype(float)

feature1 = 'avgSeekTime'
feature2 = 'avgHoldTime'
feature3 = 'averageNgramTime'
z_scored = getColumnZScores(pd.DataFrame(normalizedLabeledData), feature1)
z_scored = getColumnZScores(pd.DataFrame(z_scored), feature2)
z_scored = getColumnZScores(pd.DataFrame(z_scored), feature3)
z_scored_outliers_removed = removeOutliersByZScore(z_scored, feature1, 3)
z_scored_outliers_removed = removeOutliersByZScore(z_scored_outliers_removed, feature2, 3)
z_scored_outliers_removed = removeOutliersByZScore(z_scored_outliers_removed, feature3, 3)



def runExperiment(expName, kmeans):
    kmeans_res = kmeans.fit_predict(z_scored_outliers_removed)

    cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
        expUtils.getClusterBucketsForMultiClustering(feature_set, kmeans_res)

    sil.makeSilhouettePlot(z_scored_outliers_removed, kmeans_res, expName)

    expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                              expName, z_scored_outliers_removed)

#automated runs with function - WILL RUN
runExperiment("jonstest7_Kmeans_baseline_outliers_removed_2", kMeans_baseline)
runExperiment("jonstest7_Kmeans_high_iters_outliers_removed_2", kMeans_baseline_high_iteration)
runExperiment("jonstest7_Kmeans_baseline_4_clusters_outliers_removed_2", kMeans_baseline_4_clusters)
runExperiment("jonstest7_Kmeans_baseline_3_clusters_outliers_removed_2", kMeans_baseline_3_clusters)
runExperiment("jonstest7_Kmeans_baseline_2_clusters_outliers_removed_2", kMeans_baseline_2_clusters)