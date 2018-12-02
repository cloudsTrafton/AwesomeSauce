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



def runExperiment(expName, kmeans, labels, feature_set):
    kmeans_res = kmeans.fit_predict(feature_set)

    cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
        expUtils.getClusterBucketsForMultiClustering(labels, kmeans_res)

    sil.makeSilhouettePlot(feature_set, kmeans_res, expName)

    expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                              expName, labels)




outliersRemovedNotNormalized = feature_set
feature1 = 'avgSeekTime'
feature2 = 'avgHoldTime'
feature3 = 'averageNgramTime'
outliersRemovedNotNormalized = getColumnZScores(pd.DataFrame(outliersRemovedNotNormalized), feature1)
outliersRemovedNotNormalized = getColumnZScores(pd.DataFrame(outliersRemovedNotNormalized), feature2)
outliersRemovedNotNormalized = getColumnZScores(pd.DataFrame(outliersRemovedNotNormalized), feature3)
outliersRemovedNotNormalized = removeOutliersByZScore(outliersRemovedNotNormalized, feature1, 3)
outliersRemovedNotNormalized = removeOutliersByZScore(outliersRemovedNotNormalized, feature2, 3)
outliersRemovedNotNormalized = removeOutliersByZScore(outliersRemovedNotNormalized, feature3, 3)
outliersRemoved_prior2stdvs = expUtils.normalizeLabeledData(pd.DataFrame(outliersRemovedNotNormalized))
outliersRemoved_prior2stdvs = outliersRemoved_prior2stdvs.astype(float)
outliersRemoved2Stdvs_noNormal_labels = outliersRemoved_prior2stdvs.get(['label'])

outliersRemoved_prior2stdvs.drop(columns=['label', 'num', 'userID'], inplace=True)
print(outliersRemoved_prior2stdvs)
runExperiment("jonstest7_Kmeans_baseline_OutliersRemovedPrev3Stdvs", kMeans_baseline, outliersRemoved2Stdvs_noNormal_labels,  outliersRemoved_prior2stdvs)




# feature_set_copy1 = feature_set
# normalizedLabeledData = expUtils.normalizeLabeledData(pd.DataFrame(feature_set_copy1))
# normalizedLabeledData = normalizedLabeledData.astype(float)
#
# normalizedLabeledDataLabels = normalizedLabeledData.get(['label'])
# normalizedLabeledDataDroppedCols = normalizedLabeledData.drop(columns=['label', 'num', 'userID'])
#
# # normalized labeled data without the averages
# normalizedLabeledData_noAvgs = normalizedLabeledDataDroppedCols.\
#     drop(columns=['avgSeekTime', 'avgHoldTime', 'averageNgramTime'])
#
# #Get outlier removed data with only the Avg Data
# normalizedLabeledData_avgsOnly = normalizedLabeledData.get(['avgSeekTime', 'avgHoldTime', 'averageNgramTime', 'label'])
# normalizedLabeledData_avgsOnly_labels = normalizedLabeledData_avgsOnly.get(['label'])
#
# normalizedLabeledData_avgsOnly_zScored = normalizedLabeledData_avgsOnly
# feature1 = 'avgSeekTime'
# feature2 = 'avgHoldTime'
# feature3 = 'averageNgramTime'
# normalizedLabeledData_avgsOnly_zScored = getColumnZScores(pd.DataFrame(normalizedLabeledData_avgsOnly_zScored), feature1)
# normalizedLabeledData_avgsOnly_zScored = getColumnZScores(pd.DataFrame(normalizedLabeledData_avgsOnly_zScored), feature2)
# normalizedLabeledData_avgsOnly_zScored = getColumnZScores(pd.DataFrame(normalizedLabeledData_avgsOnly_zScored), feature3)
# normalizedLabeledData_avgsOnly_zScored = removeOutliersByZScore(normalizedLabeledData_avgsOnly_zScored, feature1, 3)
# normalizedLabeledData_avgsOnly_zScored = removeOutliersByZScore(normalizedLabeledData_avgsOnly_zScored, feature2, 3)
# normalizedLabeledData_avgsOnly_zScored = removeOutliersByZScore(normalizedLabeledData_avgsOnly_zScored, feature3, 3)
# normalizedLabeledData_avgsOnly_zScored_labels = normalizedLabeledData_avgsOnly_zScored.get(['label'])
# normalizedLabeledData_avgsOnly_zScored = normalizedLabeledData_avgsOnly_zScored.drop(columns=['label'])
#
#
# # Get Z-Scores using all features
# feature1 = 'avgSeekTime'
# feature2 = 'avgHoldTime'
# feature3 = 'averageNgramTime'
# normalizedLabeledData_allFeatures = normalizedLabeledData
# normalizedLabeledData_allFeatures = getColumnZScores(pd.DataFrame(normalizedLabeledData_allFeatures), feature1)
# normalizedLabeledData_allFeatures = getColumnZScores(pd.DataFrame(normalizedLabeledData_allFeatures), feature2)
# normalizedLabeledData_allFeatures = getColumnZScores(pd.DataFrame(normalizedLabeledData_allFeatures), feature3)
# normalizedLabeledData_allFeatures = removeOutliersByZScore(normalizedLabeledData_allFeatures, feature1, 3)
# normalizedLabeledData_allFeatures = removeOutliersByZScore(normalizedLabeledData_allFeatures, feature2, 3)
# normalizedLabeledData_allFeatures = removeOutliersByZScore(normalizedLabeledData_allFeatures, feature3, 3)
# labels_z_score = normalizedLabeledData_allFeatures.get(['label'])
# normalizedLabeledData_allFeatures = normalizedLabeledData_allFeatures.drop(columns=['label', 'num', 'userID'])
#
#
# # Remove the Seek time and see if its high correlation with Grams makes a difference
#
# feature2 = 'avgHoldTime'
# feature3 = 'averageNgramTime'
# z_scored_no_seekTime = getColumnZScores(pd.DataFrame(normalizedLabeledData), feature2)
# z_scored_no_seekTime = getColumnZScores(pd.DataFrame(z_scored_no_seekTime), feature3)
# z_scored_no_seekTime = removeOutliersByZScore(z_scored_no_seekTime, feature2, 3)
# z_scored_no_seekTime = removeOutliersByZScore(z_scored_no_seekTime, feature3, 3)
# labels_z_score_no_seekTime = z_scored_no_seekTime.get(['label'])
# z_scored_no_seekTime = z_scored_no_seekTime.drop(columns=['label', 'num', 'userID', 'avgSeekTime', 'avgSeekTime_zscore'])




#automated runs with function - WILL RUN


#Experiment with the seek time removed
# runExperiment("jonstest7_Kmeans_baseline_NoSeekTime", kMeans_baseline, labels_z_score_no_seekTime,  z_scored_no_seekTime)

#Experiments with only the averages

# normalizedLabeledData_avgsOnly.drop(columns=['avgSeekTime_zscore', 'label'], inplace=True)
# runExperiment("jonstest7_Kmeans_baseline_AvgsOnly", kMeans_baseline, normalizedLabeledData_avgsOnly_labels,  normalizedLabeledData_avgsOnly)
# runExperiment("jonstest7_Kmeans_baseline_AvgsOnly_outliers_removed", kMeans_baseline, normalizedLabeledData_avgsOnly_zScored_labels,  normalizedLabeledData_avgsOnly_zScored)

#with all avgs except for seek time
# normalizedLabeledData_avgsOnly.drop(columns=['avgSeekTime'], inplace=True)
# print(normalizedLabeledData_avgsOnly)
# runExperiment("jonstest7_Kmeans_baseline_AvgsOnly_noSeekTime", kMeans_baseline, normalizedLabeledData_avgsOnly_labels,  normalizedLabeledData_avgsOnly)
# runExperiment("jonstest7_Kmeans_baseline_AvgsOnly_noSeekTime_2Clusters", kMeans_baseline_2_clusters, normalizedLabeledData_avgsOnly_labels,  normalizedLabeledData_avgsOnly)


#Experiment without the averages
# runExperiment("jonstest7_Kmeans_baseline_noAvgs", kMeans_baseline, normalizedLabeledDataLabels, normalizedLabeledData_noAvgs)

# Removing outliers from the dataset in the most powerful features
print()
# runExperiment("jonstest7_Kmeans_baseline", kMeans_baseline, normalizedLabeledDataLabels, normalizedLabeledDataDroppedCols)
# runExperiment("jonstest7_Kmeans_baseline_outliers_removed", kMeans_baseline,labels_z_score, normalizedLabeledData_allFeatures)
# runExperiment("jonstest7_Kmeans_baseline_outliers_removed_random_init", kMeans_baseline_random_init,labels_z_score, normalizedLabeledData_allFeatures)
# runExperiment("jonstest7_Kmeans_baseline_outliers_removed_random_init_2", kMeans_baseline_random_init,labels_z_score, normalizedLabeledData_allFeatures)
# runExperiment("jonstest7_Kmeans_high_iters_outliers_removed", kMeans_baseline_high_iteration, labels_z_score, normalizedLabeledData_allFeatures)

# Try different numbers of clusters and different iteration limits
# runExperiment("jonstest7_Kmeans_baseline_4_clusters_outliers_removed", kMeans_baseline_4_clusters, labels_z_score, normalizedLabeledData_allFeatures)
# runExperiment("jonstest7_Kmeans_baseline_3_clusters_outliers_removed", kMeans_baseline_3_clusters, labels_z_score, normalizedLabeledData_allFeatures)
# runExperiment("jonstest7_Kmeans_baseline_2_clusters_outliers_removed", kMeans_baseline_2_clusters, labels_z_score, normalizedLabeledData_allFeatures)