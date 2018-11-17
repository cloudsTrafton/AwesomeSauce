# Provides code to perform multiclustering
#TODO was just messing around, nothing crazy here

from sklearn.cluster import KMeans
import data_processing.dataUtils as dataUtils
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

nGramData = dataUtils.retreiveDataSet('../feature_sets/n_grams.csv')

#Only use hold and seek time features
print(nGramData.drop([0, 3]))
holdSeekTimeData = nGramData.drop('userCount', axis=1, inplace=True)
df.drop('column_name', axis=1, inplace=True)
columns = ['averageHoldTime', 'averageSeekTime']
print(holdSeekTimeData)

#Copy used for plotting
holdSeekTimeData_Copy = holdSeekTimeData

#Six clusters for each of the six age groups
kmeans = KMeans(init='k-means++', n_clusters=6, n_init=10)
kmeans.fit(holdSeekTimeData)
labels = kmeans.predict(holdSeekTimeData)
centroids = kmeans.cluster_centers_

results = pd.DataFrame(labels)
holdSeekTimeData_Copy['clusters'] = labels
columns.extend(['clusters'])

#Lets analyze the clusters
print(holdSeekTimeData_Copy.groupby(['clusters']).mean())
#holdSeekTimeData_Copy.insert((holdSeekTimeData_Copy.shape[1], 'Results', results))