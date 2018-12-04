from sklearn.manifold import TSNE

from clustering import silhouette as sil
from data_processing import MulticlusteringExperimentUtils as expUtils
# Keep the clustering experiments that involve outliers here
from clustering.KMeansVariations import kMeans_baseline, kMeans_baseline_high_iteration, kMeans_baseline_random_init, \
    kMeans_baseline_4_clusters, kMeans_baseline_3_clusters, kMeans_baseline_2_clusters, kMeans_baseline_2_clusters_low_iter,\
    kMeans_baseline_2_clusters_high_iter, kMeans_baseline_highest_iteration, kMeans_baseline_highest_iteration_2_clusters,\
    kMeans_baseline_5_clusters, kMeans_baseline_3_clusters_random_high_iter, kMeans_baseline_3_clusters_random_med_iter

from clustering.tsne import makeTSNEPlot
import pandas as pd

# --- Remove all of the outliers for the big features ----
# average hold time
from data_processing.CleanDataUtils import feature_set, feature_set_complete_vectors_only,feature_set_more_even_vectors, feature_set_3_labels_completeSamplesOnly,feature_set_3_labels_AllSamples, feature_set_4049_reduced
from data_processing.dataUtils import getColumnZScores, removeOutliersByZScore


def removeOutliersAndNormalizeData(feature_set_input, threshold):
    feature1 = 'avgSeekTime'
    feature2 = 'avgHoldTime'
    feature3 = 'averageNgramTime'
    feature_set_outliers_removed = feature_set_input
    feature_set_outliers_removed = getColumnZScores(pd.DataFrame(feature_set_outliers_removed), feature1)
    feature_set_outliers_removed = getColumnZScores(pd.DataFrame(feature_set_outliers_removed), feature2)
    feature_set_outliers_removed = getColumnZScores(pd.DataFrame(feature_set_outliers_removed), feature3)
    feature_set_outliers_removed = removeOutliersByZScore(feature_set_outliers_removed, feature1, threshold)
    feature_set_outliers_removed = removeOutliersByZScore(feature_set_outliers_removed, feature2, threshold)
    feature_set_outliers_removed = removeOutliersByZScore(feature_set_outliers_removed, feature3, threshold)
    feature_set_outliers_removed = expUtils.normalizeLabeledData(pd.DataFrame(feature_set_outliers_removed))
    feature_set_outliers_removed = feature_set_outliers_removed.astype(float)
    feature_set_outliers_removed_labels = pd.DataFrame(feature_set_outliers_removed).get(['label'])
    feature_set_outliers_removed.drop(columns=['label', 'userID'], inplace=True)
    return feature_set_outliers_removed, feature_set_outliers_removed_labels



def runExperiment(expName, kmeans, labels, feature_set):
    kmeans_res = kmeans.fit_predict(feature_set)

    cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
        expUtils.getClusterBucketsForMultiClustering(labels, kmeans_res)

    expUtils.getF1Score(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, expName)

    feature_set_labeled = pd.DataFrame(feature_set).join(labels)
    feature_set_labeled['cluster'] = kmeans_res.copy()

    sil.makeSilhouettePlot(feature_set, kmeans_res, expName)

    makeTSNEPlot(feature_set_labeled, experimentName=expName)

    expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                              expName, labels)
feature1 = 'avgSeekTime'
feature2 = 'avgHoldTime'
feature3 = 'averageNgramTime'

set_combined_labels_completeOnly = feature_set_3_labels_completeSamplesOnly
normalized_completeOnly_combined, normalized_completeOnly_combined_labels = \
    removeOutliersAndNormalizeData(set_combined_labels_completeOnly, 3)
runExperiment("jonstest10_Kmeans_baseline_completeVectors_3Clusters", kMeans_baseline_3_clusters, normalized_completeOnly_combined_labels,  normalized_completeOnly_combined)


set_combined_labels_allSamples = feature_set_3_labels_AllSamples
normalized_set_combined_labels_allSamples, set_combined_labels_allSamples_labels = \
    removeOutliersAndNormalizeData(set_combined_labels_allSamples, 3)
runExperiment("jonstest10_Kmeans_baseline_AllSamples_3Clusters", kMeans_baseline_3_clusters, set_combined_labels_allSamples_labels,  normalized_set_combined_labels_allSamples)

runExperiment("jonstest10_Kmeans_baseline_completeVectors_3Clusters_randomInitMedIter", kMeans_baseline_3_clusters_random_med_iter, normalized_completeOnly_combined_labels,  normalized_completeOnly_combined)

runExperiment("jonstest10_Kmeans_baseline_completeVectors_3Clusters_randomInitHighIter", kMeans_baseline_3_clusters_random_high_iter, normalized_completeOnly_combined_labels,  normalized_completeOnly_combined)

runExperiment("jonstest9_Kmeans_baseline_AllSamples_3Clusters_randomInitMedIter", kMeans_baseline_3_clusters_random_med_iter, set_combined_labels_allSamples_labels,  normalized_set_combined_labels_allSamples)

runExperiment("jonstest9_Kmeans_baseline_AllSamples_3Clusters_randomInitHighIter", kMeans_baseline_3_clusters_random_high_iter, set_combined_labels_allSamples_labels,  normalized_set_combined_labels_allSamples)

middleage_reduced = feature_set_4049_reduced
normalized_middleage_reduced, middleage_reduced_labels = \
    removeOutliersAndNormalizeData(middleage_reduced, 3)

runExperiment("middleage_reduced_baseline", kMeans_baseline, middleage_reduced_labels,  normalized_middleage_reduced)

runExperiment("middleage_reduced_baseline", kMeans_baseline, middleage_reduced_labels,  normalized_middleage_reduced)


set = feature_set_complete_vectors_only

outliersRemovedNotNormalized = getColumnZScores(pd.DataFrame(set), feature1)
outliersRemovedNotNormalized = getColumnZScores(pd.DataFrame(outliersRemovedNotNormalized), feature2)
outliersRemovedNotNormalized = getColumnZScores(pd.DataFrame(outliersRemovedNotNormalized), feature3)

outliersRemovedNotNormalized = removeOutliersByZScore(outliersRemovedNotNormalized, feature1, 3)
outliersRemovedNotNormalized = removeOutliersByZScore(outliersRemovedNotNormalized, feature2, 3)
outliersRemovedNotNormalized = removeOutliersByZScore(outliersRemovedNotNormalized, feature3, 3)
set_normalized = expUtils.normalizeLabeledData(pd.DataFrame(set))
set_normalized = set_normalized.astype(float)
print(set_normalized)
set_normalized_labels = pd.DataFrame(set_normalized).get(['label'])
print(set_normalized_labels)
set_normalized.drop(columns=['label', 'userID'], inplace=True)
runExperiment("jonstest8_Kmeans_baseline_completeVectors", kMeans_baseline, set_normalized_labels,  set_normalized)


#with outliers removed
set_complete_vectors = set
feature1 = 'avgSeekTime'
feature2 = 'avgHoldTime'
feature3 = 'averageNgramTime'
#
set_complete_vectors = getColumnZScores(pd.DataFrame(set_complete_vectors), feature1)
set_complete_vectors = getColumnZScores(pd.DataFrame(set_complete_vectors), feature2)
set_complete_vectors = getColumnZScores(pd.DataFrame(set_complete_vectors), feature3)
set_complete_vectors = removeOutliersByZScore(set_complete_vectors, feature1, 3)
set_complete_vectors = removeOutliersByZScore(set_complete_vectors, feature2, 3)
set_complete_vectors = removeOutliersByZScore(set_complete_vectors, feature3, 3)
set_complete_vectors = expUtils.normalizeLabeledData(pd.DataFrame(set_complete_vectors))
set_complete_vectors = set_complete_vectors.astype(float)
set_complete_vectors_COPY = set_complete_vectors
set_complete_vectors_labels = pd.DataFrame(set_complete_vectors).get(['label'])
set_complete_vectors.drop(columns=['label', 'userID'], inplace=True)
runExperiment("jonstest8_Kmeans_baseline_completeVectors_outliersRemoved_3clusters", kMeans_baseline_3_clusters, set_complete_vectors_labels,  set_complete_vectors)

runExperiment("jonstest8_Kmeans_baseline_completeVectors_outliersRemoved_highest_iters", kMeans_baseline_highest_iteration, set_complete_vectors_labels,  set_complete_vectors)


#EXPERIMENT RUNS WITH ONLY COMPLETE VECTORS

even_vectors = feature_set_more_even_vectors
even_vectors = getColumnZScores(pd.DataFrame(even_vectors), feature1)
even_vectors = getColumnZScores(pd.DataFrame(even_vectors), feature2)
even_vectors = getColumnZScores(pd.DataFrame(even_vectors), feature3)
even_vectors = removeOutliersByZScore(even_vectors, feature1, 3)
even_vectors = removeOutliersByZScore(even_vectors, feature2, 3)
even_vectors = removeOutliersByZScore(even_vectors, feature3, 3)
even_vectors = expUtils.normalizeLabeledData(pd.DataFrame(even_vectors))
even_vectors = even_vectors.astype(float)
even_vectors_labels = pd.DataFrame(even_vectors).get(['label'])
even_vectors.drop(columns=['label', 'userID'], inplace=True)


runExperiment("evenVectors_Kmeans_baseline_completeVectors_outliersRemoved_3_clusters_high_iter_random", kMeans_baseline_3_clusters_random_high_iter, even_vectors_labels,  even_vectors)



# run it with T-SNE
tsne = TSNE(n_components=2, verbose=1)
tsne_result = tsne.fit_transform(set_complete_vectors_COPY.values)
print(len(tsne_result))

# runExperiment("jonstest8_Kmeans_baseline_completeVectors_tsne", kMeans_baseline, set_complete_vectors_labels,  tsne_result)


runExperiment("jonstest8_Kmeans_baseline_completeVectors_outliersRemoved_highest_iters_2_clusters", kMeans_baseline_highest_iteration_2_clusters, set_complete_vectors_labels,  set_complete_vectors)


runExperiment("jonstest8_Kmeans_baseline_completeVectors_outliersRemoved_2_clusters", kMeans_baseline_2_clusters, set_complete_vectors_labels,  set_complete_vectors)

runExperiment("jonstest8_Kmeans_baseline_completeVectors_outliersRemoved_4_clusters", kMeans_baseline_4_clusters, set_complete_vectors_labels,  set_complete_vectors)

runExperiment("jonstest8_Kmeans_baseline_completeVectors_outliersRemoved_3_clusters", kMeans_baseline_3_clusters, set_complete_vectors_labels,  set_complete_vectors)

avgs_only = set_complete_vectors.get(['avgSeekTime', 'avgHoldTime', 'averageNgramTime'])
runExperiment("jonstest8_Kmeans_baseline_completeVectors_outliersRemoved_avgs_only_2stdsvs", kMeans_baseline, set_complete_vectors_labels,  avgs_only)
#
noAvgs = set_complete_vectors.drop(columns=['avgSeekTime', 'avgHoldTime', 'averageNgramTime'])
runExperiment("jonstest8_Kmeans_baseline_completeVectors_outliersRemoved_no_avgs", kMeans_baseline, set_complete_vectors_labels,  noAvgs)




# -------------- TESTS WITHOUT COMPLETE VECTORS -----------------


feature_set_copy1 = feature_set
normalizedLabeledData = expUtils.normalizeLabeledData(pd.DataFrame(feature_set_copy1))
normalizedLabeledData = normalizedLabeledData.astype(float)

normalizedLabeledDataLabels = normalizedLabeledData.get(['label'])
normalizedLabeledDataDroppedCols = normalizedLabeledData.drop(columns=['label', 'num', 'userID'])

# normalized labeled data without the averages
normalizedLabeledData_noAvgs = normalizedLabeledDataDroppedCols.\
    drop(columns=['avgSeekTime', 'avgHoldTime', 'averageNgramTime'])
#
# #Get outlier removed data with only the Avg Data
normalizedLabeledData_avgsOnly = normalizedLabeledData.get(['avgSeekTime', 'avgHoldTime', 'averageNgramTime', 'label'])
normalizedLabeledData_avgsOnly_labels = normalizedLabeledData_avgsOnly.get(['label'])

normalizedLabeledData_avgsOnly_zScored = normalizedLabeledData_avgsOnly
feature1 = 'avgSeekTime'
feature2 = 'avgHoldTime'
feature3 = 'averageNgramTime'
normalizedLabeledData_avgsOnly_zScored = getColumnZScores(pd.DataFrame(normalizedLabeledData_avgsOnly_zScored), feature1)
normalizedLabeledData_avgsOnly_zScored = getColumnZScores(pd.DataFrame(normalizedLabeledData_avgsOnly_zScored), feature2)
normalizedLabeledData_avgsOnly_zScored = getColumnZScores(pd.DataFrame(normalizedLabeledData_avgsOnly_zScored), feature3)
normalizedLabeledData_avgsOnly_zScored = removeOutliersByZScore(normalizedLabeledData_avgsOnly_zScored, feature1, 3)
normalizedLabeledData_avgsOnly_zScored = removeOutliersByZScore(normalizedLabeledData_avgsOnly_zScored, feature2, 3)
normalizedLabeledData_avgsOnly_zScored = removeOutliersByZScore(normalizedLabeledData_avgsOnly_zScored, feature3, 3)
normalizedLabeledData_avgsOnly_zScored_labels = normalizedLabeledData_avgsOnly_zScored.get(['label'])
normalizedLabeledData_avgsOnly_zScored = normalizedLabeledData_avgsOnly_zScored.drop(columns=['label'])
#
#
# # Get Z-Scores using all features
feature1 = 'avgSeekTime'
feature2 = 'avgHoldTime'
feature3 = 'averageNgramTime'
normalizedLabeledData_allFeatures = normalizedLabeledData
normalizedLabeledData_allFeatures = getColumnZScores(pd.DataFrame(normalizedLabeledData_allFeatures), feature1)
normalizedLabeledData_allFeatures = getColumnZScores(pd.DataFrame(normalizedLabeledData_allFeatures), feature2)
normalizedLabeledData_allFeatures = getColumnZScores(pd.DataFrame(normalizedLabeledData_allFeatures), feature3)
normalizedLabeledData_allFeatures = removeOutliersByZScore(normalizedLabeledData_allFeatures, feature1, 3)
normalizedLabeledData_allFeatures = removeOutliersByZScore(normalizedLabeledData_allFeatures, feature2, 3)
normalizedLabeledData_allFeatures = removeOutliersByZScore(normalizedLabeledData_allFeatures, feature3, 3)
labels_z_score = normalizedLabeledData_allFeatures.get(['label'])
normalizedLabeledData_allFeatures = normalizedLabeledData_allFeatures.drop(columns=['label', 'num', 'userID'])
#
#
# # Remove the Seek time and see if its high correlation with Grams makes a difference
#
feature2 = 'avgHoldTime'
feature3 = 'averageNgramTime'
z_scored_no_seekTime = getColumnZScores(pd.DataFrame(normalizedLabeledData), feature2)
z_scored_no_seekTime = getColumnZScores(pd.DataFrame(z_scored_no_seekTime), feature3)
z_scored_no_seekTime = removeOutliersByZScore(z_scored_no_seekTime, feature2, 3)
z_scored_no_seekTime = removeOutliersByZScore(z_scored_no_seekTime, feature3, 3)
labels_z_score_no_seekTime = z_scored_no_seekTime.get(['label'])

z_scored_no_seekTime = z_scored_no_seekTime.drop(columns=['label', 'num', 'userID', 'avgSeekTime', 'avgSeekTime_zscore'])




#EXPERIMENT RUNS WITH INCOMPLETE DATA REPLACED WITH THE AVERAGES FOR THE FEATURE


# Experiment with the seek time removed
runExperiment("jonstest7_Kmeans_baseline_NoSeekTime", kMeans_baseline, labels_z_score_no_seekTime,  z_scored_no_seekTime)

# Experiments with only the averages

normalizedLabeledData_avgsOnly.drop(columns=['avgSeekTime_zscore', 'label'], inplace=True)
runExperiment("jonstest7_Kmeans_baseline_AvgsOnly", kMeans_baseline, normalizedLabeledData_avgsOnly_labels,  normalizedLabeledData_avgsOnly)
runExperiment("jonstest7_Kmeans_baseline_AvgsOnly_outliers_removed", kMeans_baseline, normalizedLabeledData_avgsOnly_zScored_labels,  normalizedLabeledData_avgsOnly_zScored)

# with all avgs except for seek time
normalizedLabeledData_avgsOnly.drop(columns=['avgSeekTime'], inplace=True)
print(normalizedLabeledData_avgsOnly)
runExperiment("jonstest7_Kmeans_baseline_AvgsOnly_noSeekTime", kMeans_baseline, normalizedLabeledData_avgsOnly_labels,  normalizedLabeledData_avgsOnly)
runExperiment("jonstest7_Kmeans_baseline_AvgsOnly_noSeekTime_2Clusters", kMeans_baseline_2_clusters, normalizedLabeledData_avgsOnly_labels,  normalizedLabeledData_avgsOnly)


# Experiment without the averages
runExperiment("jonstest7_Kmeans_baseline_noAvgs", kMeans_baseline, normalizedLabeledDataLabels, normalizedLabeledData_noAvgs)

# Removing outliers from the dataset in the most powerful features
runExperiment("jonstest7_Kmeans_baseline", kMeans_baseline, normalizedLabeledDataLabels, normalizedLabeledDataDroppedCols)
runExperiment("jonstest7_Kmeans_baseline_outliers_removed", kMeans_baseline,labels_z_score, normalizedLabeledData_allFeatures)
runExperiment("jonstest7_Kmeans_baseline_outliers_removed_random_init", kMeans_baseline_random_init,labels_z_score, normalizedLabeledData_allFeatures)
runExperiment("jonstest7_Kmeans_baseline_outliers_removed_random_init_2", kMeans_baseline_random_init,labels_z_score, normalizedLabeledData_allFeatures)
runExperiment("jonstest7_Kmeans_high_iters_outliers_removed", kMeans_baseline_high_iteration, labels_z_score, normalizedLabeledData_allFeatures)

# Try different numbers of clusters and different iteration limits
runExperiment("jonstest7_Kmeans_baseline_4_clusters_outliers_removed", kMeans_baseline_4_clusters, labels_z_score, normalizedLabeledData_allFeatures)
runExperiment("jonstest7_Kmeans_baseline_3_clusters_outliers_removed", kMeans_baseline_3_clusters, labels_z_score, normalizedLabeledData_allFeatures)
runExperiment("jonstest7_Kmeans_baseline_2_clusters_outliers_removed", kMeans_baseline_2_clusters, labels_z_score, normalizedLabeledData_allFeatures)