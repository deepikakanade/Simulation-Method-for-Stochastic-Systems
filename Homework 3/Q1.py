"""
Created on Tue Mar 13 22:46:24 2018

@author: deepikakanade
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

trainFile = '/Users/deepikakanade/Desktop/oldfaithful.csv'
Data = np.genfromtxt(trainFile, delimiter = ',')

plt.figure()
plt.scatter(Data[:,1],Data[:,2])
plt.title('Scatter plot of original data')

#K-means algorithm for K=2
kmeans = KMeans(n_clusters=2)
kmeans.fit(Data[:,[1,2]])

plt.figure()
plt.title('Scatter plot for K=2 clusters')
plt.scatter(Data[:,1],Data[:,2],c=kmeans.labels_, cmap='jet')

#plotting the center of the two clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
