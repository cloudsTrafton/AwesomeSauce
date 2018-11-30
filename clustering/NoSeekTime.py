# Keep the clustering experiments that involve outliers here
from clustering.KMeansVariations import kMeans_baseline, kMeans_baseline_high_iteration, kMeans_baseline_random_init, \
    kMeans_baseline_4_clusters, kMeans_baseline_3_clusters, kMeans_baseline_2_clusters, kMeans_baseline_2_clusters_low_iter,\
    kMeans_baseline_2_clusters_high_iter
from clustering.MultiClusteringExperiments import normalizedLabeledData
from data_processing import MulticlusteringExperimentUtils as expUtils
from clustering import silhouette as sil
import pandas as pd

# Run some experiments with the Avg Seek time removed, since Pentel reported a higher correlation between seek time
# and some of the features in his paper.

# --- Remove all of the outliers for the big features ----
# average hold time
from data_processing.CleanDataUtils import feature_set
from data_processing.dataUtils import getColumnZScores, removeOutliersByZScore

avgSeekTime = 'avgSeekTime'
df_no_seek_time = pd.DataFrame(normalizedLabeledData).drop(avgSeekTime)



def runExperiment(dataset, kmeans, experiment_name):
    result = kmeans.fit_predict(dataset)

    cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = \
        expUtils.getClusterBucketsForMultiClustering(feature_set, result)

    sil.makeSilhouettePlot(dataset, result, experiment_name)

    expUtils.getAverageForAll(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6,
                              experiment_name, dataset)
