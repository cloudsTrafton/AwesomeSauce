from sklearn.cluster import KMeans

# Some variations of K-Means with some preset parameters.
# We can keep track of the ones we use here and then just import them into our experiments

# Base Kmeans, not too much variation
kMeans_baseline = KMeans(init='k-means++', n_clusters=6, n_init=120, max_iter=500, tol=1e-04, random_state=1)

# KMeans baseline measure with high number of iterations
kMeans_baseline_high_iteration = KMeans(init='k-means++', n_clusters=6, n_init=120, max_iter=900, tol=1e-04, random_state=1)

# KMeans baseline measure with high number of iterations
kMeans_baseline_highest_iteration = KMeans(init='k-means++', n_clusters=6, n_init=120, max_iter=1500, tol=1e-04, random_state=1)

# KMeans baseline measure with high number of iterations and only 2 lusters
kMeans_baseline_highest_iteration_2_clusters = KMeans(init='k-means++', n_clusters=2, n_init=120, max_iter=1500, tol=1e-04, random_state=1)

# KMeans baseline with random centroid initialization
kMeans_baseline_random_init = KMeans(init='random', n_clusters=6, n_init=120, max_iter=500, tol=1e-04, random_state=1)

# KMeans clustering with only four clusters
kMeans_baseline_4_clusters = KMeans(init='k-means++', n_clusters=4, n_init=120, max_iter=500, tol=1e-04, random_state=1)

# KMeans clustering with only 3 clusters
kMeans_baseline_3_clusters = KMeans(init='k-means++', n_clusters=3, n_init=120, max_iter=500, tol=1e-04, random_state=1)

# Base Kmeans with only two clusters
kMeans_baseline_2_clusters = KMeans(init='k-means++', n_clusters=2, n_init=120, max_iter=500, tol=1e-04, random_state=1)

# Base Kmeans, not too much variation
kMeans_baseline_2_clusters_low_iter = KMeans(init='k-means++', n_clusters=2, n_init=120, max_iter=200, tol=1e-04, random_state=1)

# Base Kmeans, not too much variation
kMeans_baseline_2_clusters_high_iter = KMeans(init='k-means++', n_clusters=2, n_init=120, max_iter=1000, tol=1e-04, random_state=1)