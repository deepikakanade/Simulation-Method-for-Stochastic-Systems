#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 22:35:49 2018

@author: deepikakanade
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets.samples_generator import make_blobs
from numpy.core.umath_tests import inner1d
from sklearn import preprocessing

class GMMClustering:
    
    def __init__(self, n_clusters, max_iterations, initializeMethod):
        self.name = "GMMClustering"
        self.n_clusters = n_clusters
        #self.n_features = n_features
        #self.n_data = n_data
        self.max_iterations = max_iterations
        self.initializeMethod = initializeMethod
        self.labels_ = []
    
    def fit(self, X):
        
        self.n_data, self.n_features  = X.shape
        
        if self.initializeMethod == "Random":
            self.mean_vectors = np.random.uniform(0,1, size=[self.n_clusters,self.n_features])
            self.covariance_matrices = np.random.uniform(0,1, size=[self.n_features,self.n_features,self.n_clusters])
            self.alpha_vectors = np.array([0.5,0.5])
        
        self.log_likelihood_values = []
        for iteration in range(0,self.max_iterations):
            ### Expectation Step
            for i in range(0, self.n_clusters):
                temp_likelihood = inner1d(((X - self.mean_vectors[i]).dot(np.linalg.inv(self.covariance_matrices[i]))), ((X - self.mean_vectors[i])))
                likelihood = np.exp(-temp_likelihood/2.0)
                likelihood = likelihood / ((2* math.pi * abs(np.linalg.det(self.covariance_matrices[i])))**0.5)
                e_step = self.alpha_vectors[i] * likelihood
                if i==0:
                    expectation = e_step
                else:
                    expectation = np.vstack([expectation,e_step])
            expectation = expectation /  np.sum(expectation, axis=0)
               
            ### Maximization step (Updation)
            for i in range(0, self.n_clusters):
                ### update mean vector
                mean_temp = np.multiply(X.T, expectation[i]).T
                self.mean_vectors[i,:] = np.sum(mean_temp, axis=0) / np.sum(expectation[i])
            
                covariance_temp1 = np.einsum('ij,kj->jik',(X - self.mean_vectors[i]).T,(X - self.mean_vectors[i]).T)
                covariance_temp2 = np.multiply(covariance_temp1.T, expectation[i]).T
                self.covariance_matrices[i,:,:] = np.sum(covariance_temp2, axis=0) / np.sum(expectation[i])
                ### update alpha vector
                self.alpha_vectors[i] = np.sum(expectation[i]) /  self.n_data
             
            ### evaluate log likelihood
            for i in range(0, self.n_clusters):
                temp_likelihood2 = inner1d(((X - self.mean_vectors[i]).dot(np.linalg.inv(self.covariance_matrices[i]))), ((X - self.mean_vectors[i])))
                likelihood2 = np.exp(-temp_likelihood2/2.0)
                likelihood2 = likelihood2 / ((2* math.pi * abs(np.linalg.det(self.covariance_matrices[i])))**0.5)
                e_step2 = self.alpha_vectors[i] * likelihood2
                if i==0:
                    expectation2 = e_step2
                else:
                    expectation2 = np.vstack([expectation2,e_step2])
            log_likelihood = np.sum(expectation2, axis=0)
            log_likelihood = np.sum(np.log(log_likelihood))
            self.log_likelihood_values.append(log_likelihood)
            if iteration > 0:
                if self.log_likelihood_values[iteration] == self.log_likelihood_values[iteration-1]:
                    print("Converged at iteration ", iteration)
                    break
            ### assign cluster values
            self.labels_ = np.argmax(expectation2,axis=0)
            
        
            
n_features=2
n_clusters = 2
n_data = 300

### Generate data Spherical structure
centers = [[-5, 0], [0, 1.5]]
X1, y1 = make_blobs(n_samples=n_data, centers=centers, random_state=40)
X1_scaled = preprocessing.scale(X1)

plt.figure()
plt.scatter(X1[:,0],X1[:,1])
plt.title('Generated Data - Spherical')

gmm_model1 = GMMClustering(n_clusters, 100, "Random")    
print('For spherical data: ') 
gmm_model1.fit(X1_scaled)
plt.figure()
plt.scatter(X1[:,0], X1[:,1], c=gmm_model1.labels_, cmap='jet')
plt.title('Generated Data - Spherical - GMM clustered')

### Generate data Ellipsoid structure
centers = [[-5, 0], [0, 1.5]]
X2, y2 = make_blobs(n_samples=n_data, centers=centers, random_state=40)
transformation = [[0.4, 0.2], [-0.4, 1.2]]
X2 = np.dot(X2, transformation)
X2_scaled = preprocessing.scale(X2)

plt.figure()
plt.scatter(X2[:,0],X2[:,1])
plt.title('Generated Data - Ellipsoid')

gmm_model2 = GMMClustering(n_clusters, 100, "Random")  
print('For Ellipsoidal data: ')  
gmm_model2.fit(X2_scaled)
plt.figure()
plt.scatter(X2[:,0], X2[:,1], c=gmm_model2.labels_, cmap='jet')
plt.title('Generated Data - Ellipsoid - GMM clustered')

### Generate poorly seperated subpopulations
centers = [[-5, 0], [-4, 0]]
X3, y3 = make_blobs(n_samples=n_data, centers=centers, random_state=40)
X3_scaled = preprocessing.scale(X3)

plt.figure()
plt.scatter(X3[:,0],X3[:,1])
plt.title('Generated Data - Poorly separated data')

gmm_model3 = GMMClustering(n_clusters, 1000, "Random")    
print('For poorly seperated data: ') 
gmm_model3.fit(X3_scaled)
plt.figure()
plt.scatter(X3[:,0], X3[:,1], c=gmm_model3.labels_, cmap='jet')
plt.title('Generated Data - Poorly separated data  - GMM clustered')

### old-faithful data
X4 = np.genfromtxt("/Users/deepikakanade/Desktop/oldfaithful.csv",delimiter = ',')
X4_scaled = preprocessing.scale(X4)

plt.figure()
plt.scatter(X4[:,1], X4[:,2])
plt.title("Scatter plot of old-faithful data")
plt.xlabel("old faithful geyser eruptions")
plt.ylabel("waiting time between eruptions")

gmm_model4 = GMMClustering(n_clusters, 50, "Random")    
print('For old faithful dataset: ') 
gmm_model4.fit(X4_scaled[:,1:])
plt.figure()
plt.scatter(X4[:,1], X4[:,2], c=gmm_model4.labels_, cmap='jet')
plt.title("Scatter plot of old-faithful data - GMM clustered")
plt.xlabel("old faithful geyser eruptions")
plt.ylabel("waiting time between eruptions")
