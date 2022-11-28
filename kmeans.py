import math
import random

from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import urllib.request
import io
from scipy.io import arff
import pandas as pd


def load_data(url: str):
    ftp_stream = urllib.request.urlopen(url)
    data, meta = arff.loadarff(io.StringIO(ftp_stream.read().decode('utf-8')))
    data_frame = pd.DataFrame(data)
    return data_frame


def normalize_data(array):
    new_array = array.copy()
    _, num_cols = array.shape

    def normalize(value):
        return (value - cur_col_min) / float(cur_col_max - cur_col_min)

    for i in range(0, num_cols):
        cur_col = array[:, i]
        cur_col_max = np.max(cur_col)
        cur_col_min = np.min(cur_col)
        normalize_vectorized = np.vectorize(normalize)
        new_array[:, i] = normalize_vectorized(cur_col)
    return new_array


class KMeansCluster:
    def __init__(self, point):
        self.points = []
        self.centroid = point
        self.previous_centroid = None

    def add_point(self, point):
        self.points.append(point)

    def clear_points(self):
        self.points = []

    def get_centroid(self, num_decimals=None):
        avg_point = [0] * len(self.points[0].data)
        for point in self.points:
            for j in range(0, len(self.points[0].data)):
                avg_point[j] += point.data[j]
        for i in range(0, len(avg_point)):
            avg_point[i] /= len(self.points)
            if num_decimals is not None:
                avg_point[i] = round(avg_point[i], num_decimals)
        return avg_point

    def set_centroid(self):
        centroid = self.get_centroid()
        self.previous_centroid = Point(self.centroid.data)
        self.centroid = Point(centroid)
        return centroid

    def check_changed(self):
        if self.previous_centroid is None:
            return True

        for i in range(0, len(self.centroid.data)):
            if self.centroid.data[i] != self.previous_centroid.data[i]:
                return True
        return False


class Point:
    def __init__(self, data):
        self.data = data

    def get_distance_to(self, point2):
        dist_squared_sum = 0
        for i in range(0, len(self.data)):
            dist_squared_sum += (pow(self.data[i] - point2.data[i], 2))
        return math.sqrt(dist_squared_sum)


class KMEANSClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, debug=False):  ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug
        self.clusters = []
        self.data = []
        self.raw_data = []

    def fit(self, X):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.initialize_points(X)
        self.initialize_clusters()
        converged = False
        while not converged:
            self.group_cluster_points()
            converged = self.check_converged()
            if not converged:
                self.reset_cluster_points()
        return self

    def initialize_points(self, X):
        self.data = []
        for row in X:
            self.data.append(Point(row))

    def initialize_clusters(self):
        self.clusters = []
        if self.debug:
            for i in range(0, self.k):
                self.clusters.append(KMeansCluster(self.data[i]))
        else:
            selection_data = self.data.copy()
            for i in range(0, self.k):
                point_index = random.randint(0, len(selection_data) - 1)
                self.clusters.append(KMeansCluster(selection_data[point_index]))
                selection_data = np.delete(selection_data, point_index, 0)

    def reset_cluster_points(self):
        for cluster in self.clusters:
            cluster.set_centroid()
            cluster.clear_points()

    def group_cluster_points(self):
        for point_index, data_point in enumerate(self.data):
            min_distance = np.inf
            min_cluster_index = -1
            for i, cluster in enumerate(self.clusters):
                distance_to_centroid = data_point.get_distance_to(cluster.centroid)
                if distance_to_centroid < min_distance:
                    min_distance = distance_to_centroid
                    min_cluster_index = i
            self.clusters[min_cluster_index].add_point(data_point)

    def check_converged(self):
        for cluster in self.clusters:
            cluster_changed = cluster.check_changed()
            if cluster_changed:
                return False
        return True

    def print_clusters(self, num_decimals=None):
        """
            Used for grading.
            print("Num clusters: {:d}\n".format(k))
            print("Silhouette score: {:.4f}\n\n".format(silhouette_score))
            for each cluster and centroid:
                print(np.array2string(centroid,precision=4,separator=","))
                print("{:d}\n".format(size of cluster))
        """
        print("Num clusters: {:d}\n".format(self.k))
        print("Silhouette score: {:.4f}\n\n".format(self.get_silhouette_score()))
        for cluster in self.clusters:
            print(cluster.get_centroid(num_decimals=num_decimals))
            print(len(cluster.points))

    def get_silhouette_score(self):
        classified_data = []
        classifications = []
        for i, cluster in enumerate(self.clusters):
            for point in cluster.points:
                classified_data.append(point.data)
            cluster_classifications = [i] * len(cluster.points)
            classifications.extend(cluster_classifications)
        return silhouette_score(X=classified_data, labels=classifications)

def run_debug():
    raw_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/abalone.arff"))
    normalized_data = normalize_data(raw_data)
    kmeans_clustering = KMEANSClustering(k=5, debug=True)
    kmeans_clustering.fit(normalized_data)
    kmeans_clustering.print_clusters(num_decimals=4)

def run_evaluation():
    raw_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/seismic-bumps_train.arff"))

    def decode(value):
        if type(value) == bytes:
            return value.decode()
        return value
    decode_vectorized = np.vectorize(decode)
    decoded_data = decode_vectorized(raw_data)
    normalized_data = normalize_data(decoded_data)
    kmeans_clustering = KMEANSClustering(k=5, debug=True)
    kmeans_clustering.fit(normalized_data)
    kmeans_clustering.print_clusters(num_decimals=4)


def run_iris():
    raw_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/iris.arff"))
    data = raw_data[:, :-1]
    k_array = [2, 3, 4, 5, 6, 7]
    silhouette_scores = []
    for i in k_array:
        print(f"KMeans Clustering with k = {i}")
        kmeans_clustering = KMEANSClustering(k=i)
        kmeans_clustering.fit(data)
        kmeans_clustering.print_clusters(num_decimals=4)
        silhouette_scores.append(kmeans_clustering.get_silhouette_score())
        print()
        print()
    plt.plot(k_array, silhouette_scores)
    plt.title("Silhouette Score by K-value")
    plt.xlabel("K-value")
    plt.ylabel("Silhouette Score")
    plt.show()


def run_iris_centroid_experiment():
    raw_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/iris.arff"))
    data = raw_data[:, :-1]
    for i in range(0, 5):
        print(f"KMeans Clustering with k = 5, iteration {i}")
        kmeans_clustering = KMEANSClustering(k=4)
        kmeans_clustering.fit(data)
        kmeans_clustering.print_clusters(num_decimals=4)
        print()
        print()


def run_sklearn_iris_hac():
    raw_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/iris.arff"))
    data = raw_data[:, :-1]

    for i in range(2, 8):
        print(f"Complete link - n-clusters={i}")
        sklearn_hac = AgglomerativeClustering(n_clusters=i, linkage="complete")
        labels = sklearn_hac.fit_predict(data)
        print(silhouette_score(data, labels))
    print()

    print("Ward link - n-clusters=2")
    sklearn_hac = AgglomerativeClustering(n_clusters=2, linkage="ward")
    labels = sklearn_hac.fit_predict(data)
    print(silhouette_score(data, labels))
    print()

    print("Ward link - n-clusters=3")
    sklearn_hac = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = sklearn_hac.fit_predict(data)
    print(silhouette_score(data, labels))
    print()

    print("Average link - n-clusters=3")
    sklearn_hac = AgglomerativeClustering(n_clusters=3, linkage="average")
    labels = sklearn_hac.fit_predict(data)
    print(silhouette_score(data, labels))
    print()

    print("L1 affinitiy, n-clusters=3, average link")
    sklearn_hac = AgglomerativeClustering(n_clusters=3, affinity='l1', linkage="average")
    labels = sklearn_hac.fit_predict(data)
    print(silhouette_score(data, labels))
    print()

    print("L2 affinitiy, n-clusters=3, average link")
    sklearn_hac = AgglomerativeClustering(n_clusters=3, affinity='l2', linkage="average")
    labels = sklearn_hac.fit_predict(data)
    print(silhouette_score(data, labels))


def run_sklearn_iris_kmeans():
    raw_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/iris.arff"))
    data = raw_data[:, :-1]

    print("KMeans")
    for i in range(2,8):
        print(f"n-clusters={i}")
        sklearn_kmeans = KMeans(n_clusters=i, n_init=1, init="random")
        labels = sklearn_kmeans.fit_predict(data)
        print(silhouette_score(data, labels))
    print()

    print("n-clusters=3, number of different trial seeds=10, kmeans++ centroid initialization")
    sklearn_kmeans = KMeans(n_clusters=3)
    labels = sklearn_kmeans.fit_predict(data)
    print(silhouette_score(data, labels))
    print()


def main():
    run_sklearn_iris_kmeans()


if __name__ == "__main__":
    main()
# Initialize centroids,
# Point grouping has changed
# While point grouping has changed
    # For each point
        # Get distance to all cluster centroids
        # Group into cluster with the closest centroid (earliest in list)
    # Evaluate new centroids
    # If point grouping is the same, set point_grouping_changed to false
