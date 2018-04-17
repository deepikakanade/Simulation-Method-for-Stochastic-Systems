"""
Created on Tue Mar 13 23:59:08 2018

@author: deepikakanade
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

trainFile = '/Users/deepikakanade/Desktop/nips-87-92.csv'
Data = np.genfromtxt(trainFile, delimiter = ',')

trial2 = Data[1:,2:np.shape(Data)[1]]

# k means determine k
distortions = []
index= []
#K = 15
K_dict={}
K_dict=[10,50,100,300,500,600,650]
silhoutte_score=[]
ad = {}
for k in sorted(K_dict):
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(trial2)
    distortions.append(sum(np.min(cdist(trial2, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / trial2.shape[0])
    ad[k] = kmeanModel
    silhoutte_score.append(silhouette_score(trial2, kmeanModel.labels_))


bestk = 100
best_model = ad[bestk]
for i in range(0,bestk):
    print('Cluster of: ',i)
    ad1 = Data[1:,:]
    print(ad1[best_model.labels_ == i][:,1])

  
# Plot the elbow
    '''
plt.figure()
plt.plot(sorted(K_dict), distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of sqaured distances')
plt.title('The Elbow Method showing the optimal k')
plt.show()
'''
plt.figure()
plt.plot(sorted(K_dict), silhoutte_score, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhoutte score')
plt.title('The Silhoutte showing the optimal k')
plt.show()
