import scipy
from sklearn.cluster import KMeans
import data_processing.dataUtils as dataUtils
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Binary Clustering for age.
# Reads in the dataset and performs binary clustering, i.e. creates two clusters:
# First cluster is all data that is calculated to belong in the specified age range
# Second cluster is all those data points determined not to belong in the specified age range
# These clusters are built using the K-Means algorithm, which is implemented by the sklearn library

#NOTE: thank you to StackExchange for guidance on how to use SKLearn KMeans:
#https://datascience.stackexchange.com/questions/16700/confused-about-how-to-apply-kmeans-on-my-a-dataset-with-features-extracted




#Runs the K-Means clustering algorithm for binary clustering using SKLearn's optimized centroid placement

# -- Preliminary Trial for Clustering data based on the three features extracted as N-Grams -- #

X = dataUtils.retreiveDataSet('../feature_sets/jonstest.csv').values
nGramDataCopy = X

# Parameter:
# init: k-means++ -
kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10, max_iter=300, tol=1e-04, random_state=0)
result = kmeans.fit_predict(X)
print(result)
plt = dataUtils.plotClusteredResults(X, result, 5, False)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

#for label in labels:
#    print(label)

#dataUtils.plotData(labels)

# in row_dict we store actual meanings of rows, in my case it's russian words
# clusters = {}
# n = 0
# for item in labels:
#     if item in clusters:
#         clusters[item].append(row_dict[n])
#     else:
#         clusters[item] = [row_dict[n]]
#         n +=1
#
# for item in clusters:
#     print ("Cluster ", item)
#     for i in clusters[item]:
#         print(i)