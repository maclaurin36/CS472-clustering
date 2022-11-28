import math

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


# Initialize clusters
# Repeat until k clusters remain
# Find the clusters that are closest together using single/complete link
# Merge the two clusters

# Merge clusters

# Single link distance - Distance between two clusters is distance from their closest points
# For each point in cluster 1
# For each point in cluster 2
# Calculate the distance between the two points
# Set min_distance to this distance if distance is less than min_distance

# Complete link - Distance between two clusters is distance between their farthest points
# For each point in cluster 1
# For each point in cluster 2
# Calculate the distance between the two points
# Set max_distance to this distance if distance is greater than max_distance


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


class Cluster:
    def __init__(self, point):
        self.points = [point]

    def get_complete_link_distance(self, cluster2):
        max_distance = 0
        for i in range(0, len(self.points)):
            cluster1_point = self.points[i]
            for j in range(0, len(cluster2.points)):
                cluster2_point = cluster2.points[j]
                distance = cluster1_point.get_distance_to(cluster2_point)
                if distance > max_distance:
                    max_distance = distance
        return max_distance

    def get_single_link_distance(self, cluster2):
        min_distance = np.inf
        for i in range(0, len(self.points)):
            cluster1_point = self.points[i]
            for j in range(0, len(cluster2.points)):
                cluster2_point = cluster2.points[j]
                distance = cluster1_point.get_distance_to(cluster2_point)
                if distance < min_distance:
                    min_distance = distance
        return min_distance

    def merge(self, cluster2):
        self.points.extend(cluster2.points)

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


class Point:
    def __init__(self, data):
        self.data = data

    def get_distance_to(self, point2):
        dist_squared_sum = 0
        for i in range(0, len(self.data)):
            dist_squared_sum += (pow(self.data[i] - point2.data[i], 2))
        return math.sqrt(dist_squared_sum)


class HACClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, link_type='single'):  ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
        self.clusters = []
        self.data = []

    def fit(self, data):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.initialize_clusters(data)
        while len(self.clusters) > self.k:
            cluster_index1, cluster_index2 = self.find_nearest_clusters()
            self.merge_clusters(cluster_index1, cluster_index2)
        return self

    def print_clusters(self, num_decimals=None):
        """
            Used for grading.
            print("Num clusters: {:d}\n".format(k))
            print("Silhouette score: {:.4f}\n\n".format(silhouette_score))
            for each cluster and centroid:
                print(np.array2string(centroid,precision=4,separator=","))
                print("{:d}\n".format(size of cluster))
        """
        print("Num clusters: {:d}".format(self.k))
        print("Silhouette score: {:.4f}\n".format(self.get_silhouette_score()))
        for cluster in self.clusters:
            print(cluster.get_centroid(num_decimals=num_decimals))
            print(len(cluster.points))

    def initialize_clusters(self, data):
        self.data = data
        self.clusters = []
        for row in data:
            point = Point(row)
            cluster = Cluster(point)
            self.clusters.append(cluster)

    def find_nearest_clusters(self):
        min_distance = np.inf
        cluster_indices = (-1, -1)
        for cluster1_index in range(0, len(self.clusters)):
            for cluster2_index in range(cluster1_index + 1, len(self.clusters)):
                cluster1 = self.clusters[cluster1_index]
                cluster2 = self.clusters[cluster2_index]
                distance_between = self.get_distance(cluster1, cluster2)
                if distance_between < min_distance:
                    min_distance = distance_between
                    cluster_indices = (cluster1_index, cluster2_index)
        return cluster_indices

    def get_distance(self, cluster1, cluster2):
        if self.link_type == 'single':
            return cluster1.get_single_link_distance(cluster2)
        else:
            return cluster1.get_complete_link_distance(cluster2)

    def merge_clusters(self, cluster1_index, cluster2_index):
        cluster1 = self.clusters[cluster1_index]
        cluster2 = self.clusters[cluster2_index]
        cluster1.merge(cluster2)
        del self.clusters[cluster2_index]

    def get_silhouette_score(self):
        classified_data = []
        classifications = []
        for i, cluster in enumerate(self.clusters):
            for point in cluster.points:
                classified_data.append(point.data)
            cluster_classifications = [i] * len(cluster.points)
            classifications.extend(cluster_classifications)
        return silhouette_score(X=classified_data, labels=classifications)


def run_single_debug():
    raw_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/abalone.arff"))
    normalized_data = normalize_data(raw_data)
    hac_clustering = HACClustering(k=5, link_type="single")
    hac_clustering.fit(normalized_data)
    hac_clustering.print_clusters(num_decimals=4)


def run_complete_debug():
    raw_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/abalone.arff"))
    normalized_data = normalize_data(raw_data)
    hac_clustering = HACClustering(k=5, link_type="complete")
    hac_clustering.fit(normalized_data)
    hac_clustering.print_clusters(num_decimals=4)


def run_single_eval():
    raw_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/seismic-bumps_train.arff"))

    def decode(value):
        if type(value) == bytes:
            return value.decode()
        return value
    decode_vectorized = np.vectorize(decode)
    decoded_data = decode_vectorized(raw_data)
    normalized_data = normalize_data(decoded_data)
    hac_clustering = HACClustering(k=5, link_type="single")
    hac_clustering.fit(normalized_data)
    hac_clustering.print_clusters(num_decimals=4)


def run_complete_eval():
    raw_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/seismic-bumps_train.arff"))

    def decode(value):
        if type(value) == bytes:
            return value.decode()
        return value
    decode_vectorized = np.vectorize(decode)
    decoded_data = decode_vectorized(raw_data)
    normalized_data = normalize_data(decoded_data)
    hac_clustering = HACClustering(k=5, link_type="complete")
    hac_clustering.fit(normalized_data)
    hac_clustering.print_clusters(num_decimals=4)


def run_single_iris():
    raw_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/iris.arff"))
    data = raw_data[:,:-1]
    k_array = [2, 3, 4, 5, 6, 7]
    silhouette_scores = []
    for i in k_array:
        print(f"Single-link HAC Clustering with k = {i}")
        hac_clustering = HACClustering(k=i, link_type="single")
        hac_clustering.fit(data)
        hac_clustering.print_clusters(num_decimals=4)
        silhouette_scores.append(hac_clustering.get_silhouette_score())
        print()
        print()
    plt.plot(k_array, silhouette_scores)
    plt.title("Silhouette Score by K-value")
    plt.xlabel("K-value")
    plt.ylabel("Silhouette Score")
    plt.show()


def run_complete_iris():
    raw_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/iris.arff"))
    data = raw_data[:,:-1]
    k_array = [2, 3, 4, 5, 6, 7]
    silhouette_scores = []
    for i in k_array:
        print(f"Complete-link HAC Clustering with k = {i}")
        hac_clustering = HACClustering(k=i, link_type="complete")
        hac_clustering.fit(data)
        hac_clustering.print_clusters(num_decimals=4)
        silhouette_scores.append(hac_clustering.get_silhouette_score())
        print()
        print()
    plt.plot(k_array, silhouette_scores)
    plt.title("Silhouette Score by K-value")
    plt.xlabel("K-value")
    plt.ylabel("Silhouette Score")
    plt.show()



def main():
    run_complete_iris()


if __name__ == "__main__":
    main()
