import scipy
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn import preprocessing
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

df = dataUtils.retreiveDataSet('../feature_sets/jonstest.csv')
# df1_labels = df['\'label\'']
# df1_labels = df.loc(['\'label\''])
df1 = df.drop(columns=['\'userID\''])
# print(df1_labels)

#normalize the values
df = df1.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()
df_norm = min_max_scaler.fit_transform(df.values)

X = df_norm

# Parameter:
# init: k-means++ -
kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10, max_iter=300, tol=1e-04, random_state=0)
result = kmeans.fit_predict(X)

# plt = dataUtils.plotClusteredResults(X, result, 5, False)
print(result)
# plt.scatter(X[result == 0, 0],
#             X[result == 0, 1],
#             s=50, c='lightgreen',
#             marker='s', edgecolor='black',
#             label='cluster 1')
# plt.scatter(X[result == 1, 0],
#             X[result == 1, 1],
#             s=50, c='orange',
#             marker='o', edgecolor='black',
#             label='cluster 2')
# plt.scatter(X[result == 2, 0],
#             X[result == 2, 1],
#             s=50, c='blue',
#             marker='v', edgecolor='black',
#             label='cluster 3')
# plt.scatter(X[result == 3, 0],
#             X[result == 3, 1],
#             s=50, c='yellow',
#             marker='D', edgecolor='black',
#             label='cluster 4')
# plt.scatter(X[result == 4, 0],
#             X[result == 4, 1],
#             s=50, c='orange',
#             marker='h', edgecolor='black',
#             label='cluster 5')

# #result = kmeans.fit_predict(X[result == 0, 0])

# plt.legend(scatterpoints=1)
# plt.grid()
# plt.show()

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


# --- Trial experimentation with Labels and F-Scores ----
df_labeled = dataUtils.retreiveDataSet('../feature_sets/feature_vectors_labeled_ctrafton.csv')
print(df_labeled)
print(df_labeled.axes)
df_labeled_labels = df_labeled.get([' \'label\''])
df_labeled = df_labeled.drop(columns=['\'userID\''])
df_labeled = df_labeled.drop(columns=[' \'label\''])
# print(df1_labels)

#normalize the values
df_labeled = df_labeled.astype(float)

df_norm_labeled = min_max_scaler.fit_transform(df_labeled.values)

X2 = df_norm_labeled

# Parameter:
# init: k-means++ -
result_labeled = kmeans.fit_predict(X2)
#TODO get the labels
print(df_labeled_labels.values)
# Calculate the F Score for this run
fScore = f1_score(df_labeled_labels, result_labeled, average='micro')
print("F-Score is: " + str(fScore))