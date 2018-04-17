#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 23:46:55 2018

@author: deepikakanade
"""
import numpy as np
import matplotlib.pyplot as plt

###################################Question 1#########################################
#pi-Estimation for 1 iteration
x_array = (np.random.uniform(0,1,100) **2) 
y_array = (np.random.uniform(0,1,100) **2)
summation = x_array + y_array
print('Area of the inscribed quarter circle:', sum(summation<=1))


pi = []
for k in range (0,50):
    x_array = (np.random.uniform(0,1,100) **2) 
    y_array = (np.random.uniform(0,1,100) **2)
    summation = x_array + y_array
    pi_value = sum(summation<=1)*4/100
    pi.append(pi_value)
variance=np.var(pi)
pi_value = sum(summation<=1)*4/100
print('Estimated value of pi: ', pi_value)
plt.figure()
plt.hist(pi)
plt.title('Samples(n)=100 and K=50')
plt.xlabel('Estimated Pi Value')
plt.ylabel('Count')

#pi-Estimation for k=50 iterations
n=[10, 50, 100, 500, 1000, 5000, 10000]
variance = []
mean=[]
for i in n:
    pi_1 = []
    for k in range (0,50):
        x_array = (np.random.uniform(0,1,i) **2) 
        y_array = (np.random.uniform(0,1,i) **2)
        summation = x_array + y_array
        pi_value = sum(summation<=1)*4/i
        pi_1.append(pi_value)
    variance.append(np.var(pi_1))
    mean.append(np.mean(pi_1))

plt.figure()
plt.plot(n,variance)
plt.xlabel('n')
plt.ylabel('Variance')
plt.title('Plot of variance values for different n')


print('\nMean for all different values of n: ')
print(mean)

print('\nVariance for all different values of n: ')
print(variance)
