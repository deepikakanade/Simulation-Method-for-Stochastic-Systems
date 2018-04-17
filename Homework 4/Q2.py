#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 00:33:47 2018

@author: deepikakanade
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, uniform
from scipy.stats import multivariate_normal

########## STRATIFICATION
fcn=[]
stratified_function=[]
## Function 1
for i in range (0,50):
    x = np.random.uniform(0.8,3,1000)
    function=pow((1+(np.sinh(2*x) * np.log(x))),-1)
    fcn.append(((np.sum(function))/1000) * (3-0.8))
    
var1=np.var(fcn)
mean1=np.mean(fcn)
print('\nMean value for function 1 with simple Monte Carlo: ',mean1)
print('Variance value for function 1 with simple Monte Carlo',var1)

n1=750
n2=250
for i in range(0,50):
    x_1 = np.random.uniform(0.8,1.5,n1)
    function_1=pow((1+(np.sinh(2*x_1) * np.log(x_1))),-1)
    fcn_1=((np.sum(function_1))/n1 *(1.5 - 0.8))

    x_2 = np.random.uniform(1.5,3,n2)
    function_2=pow((1+(np.sinh(2*x_2) * np.log(x_2))),-1)
    fcn_2=((np.sum(function_2))/n2 * (3 - 1.5))

    stratified_function.append(fcn_1+fcn_2)

var2=np.var(stratified_function)
mean2=np.mean(stratified_function)
print('\nMean value for function 1 with stratification: ',mean2)
print('Variance value for function 1 with stratification',var2)

fcn2=[]
stratified_function2=[]
## Function 2
for i in range (0,50):
    x2= np.random.uniform(-3.142,3.142,(1000, 2))
    function2=np.exp(-pow(x2[:,0],4) - pow(x2[:,1],4))
    fcn2.append((np.sum(function2))/1000 * (3.142 -(-3.142)) * (3.142 -(-3.142)))

var1_f2=np.var(fcn2)
mean1_f2=np.mean(fcn2)


for i in range (0,50):
    x2_2= np.random.uniform(-1.7,1.7,(1000, 2))
    function2_2=np.exp(-pow(x2_2[:,0],4) - pow(x2_2[:,1],4))
    fcn2_2=(np.sum(function2_2))/1000 * 4 * 1.7 * 1.7

    stratified_function2.append(fcn2_2)
    
var2_f2=np.var(stratified_function2)
mean2_f2=np.mean(stratified_function2)


# Importance Sampling for Function 1
Sampling_Fnc1 = []
for i in range (0,50):
    mu = 0.45
    std = 0.045
    sigma = pow(std,2)
    X1 = np.random.normal(loc=mu, scale=sigma, size=10000)
    
    Fnc1 = pow((1 + (np.sinh(2*X1)*np.log(X1))),-1)*2.2
    
    Fnc1_dummy = norm.pdf(X1, loc=mu, scale=sigma)
    W1_x = uniform.pdf(X1, loc=0.8, scale=2.2)/(Fnc1_dummy)
    
    mul = Fnc1*W1_x
    mul = mul[~np.isnan(mul)]
    Sampling_Fnc1.append(np.mean(mul))
    
variance_impsampling_Fnc1=np.var(Sampling_Fnc1)
Mean_impsampling_Fnc1=np.mean(Sampling_Fnc1)

print('\nMean value for function 1 with important sampling', Mean_impsampling_Fnc1) 
print('Variance value for function 1 with  important sampling: ', variance_impsampling_Fnc1)


print('\nMean value for function 2 with simple Monte Carlo: ',mean1_f2)
print('Variance value for function 2 with simple Monte Carlo',var1_f2)

print('\nMean value for function 2 with stratification: ',mean2_f2)
print('Variance value for function 2 with stratification',var2_f2)

# Importance Sampling for Function 2
def function_f(x,y):
    z = math.pow((math.pi*2),2) * (np.exp((-1 * math.pow(x,4)) + (-1 * math.pow(y,4))))
    return z

def function_p(x):
    z1 = uniform.pdf(x[0], loc=-math.pi, scale=math.pi)
    z2 = uniform.pdf(x[1], loc=-math.pi, scale=math.pi)
    return z1*z2

def function_g(x, mu, std):
    z = multivariate_normal.pdf(x, mean=mu, cov=std) 
    return z

importance_sampling_estimates = []
mu = 0
std = 0.5

mu_mat = np.array([mu, mu])

for i in range(0,50):
    test_x3 = np.random.normal(loc=mu,size=1000)
    test_y3 = np.random.normal(loc=mu,size=1000)

    xx, yy = np.meshgrid(test_x3, test_y3)
    test_xy = np.hstack((xx.reshape(xx.shape[0]*xx.shape[1], 1, order='F'), yy.reshape(yy.shape[0]*yy.shape[1], 1, order='F')))
    
    fx3 = math.pow((math.pi*2),2) * (np.exp((-1 * pow(test_xy[:,0],4)) + (-1 * pow(test_xy[:,1],4))))
    z1 = uniform.pdf(test_xy[:,0], loc=-math.pi, scale=math.pi+math.pi)
    z2 = uniform.pdf(test_xy[:,1], loc=-math.pi, scale=math.pi+math.pi)
    px3 = z1*z2
    gx3 = multivariate_normal.pdf(test_xy, mean=mu_mat)
    
    
    z3 = (np.array(fx3)) * (np.array(px3) / np.array(gx3))
    z3 = z3[~np.isnan(z3)]
    importance_sampling_estimates.append(np.mean(z3))

print('\nMean value for function 2 with important sampling', np.mean(importance_sampling_estimates))    
print('Variance value for function 2 with  important sampling: ', np.var(importance_sampling_estimates))


last_fnc=[]
stratified_lastfunction=[]
## Function 3
for i in range (0,50):
    x3= np.random.uniform(-5,5,(1000,2))
    function3=20 + pow(x3[:,0],2) + pow(x3[:,1],2) - 10 * (np.cos(2 * math.pi * x3[:,0] ) + np.cos(2 * math.pi * x3[:,1] ))
    last_fnc.append((np.sum(function3))/1000 * (5 -(-5)) * (5 -(-5)))

var_lastFcn=np.var(last_fnc)
mean_lastFcn=np.mean(last_fnc)

print('\nMean value for function 3 with simple Monte Carlo: ',mean_lastFcn)
print('Variance value for function 3 with simple Monte Carlo',var_lastFcn)

n1=50
n2=50
n3=800
n4=50
n5=50
for i in range (0,50):
    
    x3_1= np.random.uniform(-5,5,n1)
    y3_1= np.random.uniform(-5,-2.5,n1)
    last_fnc_1=20 + pow(x3_1,2) + pow(y3_1,2) - 10 * (np.cos(2 * math.pi * x3_1 ) + np.cos(2 * math.pi * y3_1 ))
    fcn3_1=(np.sum(last_fnc_1))/n1 * 10 * 2.5
    
    x3_2= np.random.uniform(-5,5,n2)
    y3_2= np.random.uniform(-2.5,2.5,n2)
    last_fnc_2=20 + pow(x3_2,2) + pow(y3_2,2) - 10 * (np.cos(2 * math.pi * x3_2 ) + np.cos(2 * math.pi * y3_2 ))
    fcn3_2=(np.sum(last_fnc_2))/n2 * 10 * 5
    
    x3_3= np.random.uniform(-5,5,n3)
    y3_3= np.random.uniform(2.5,5,n3)
    last_fnc_3=20 + pow(x3_3,2) + pow(y3_3,2) - 10 * (np.cos(2 * math.pi * x3_3 ) + np.cos(2 * math.pi * y3_3 ))
    fcn3_3=(np.sum(last_fnc_3))/n3 * 10 * 2.5
    
    y3_4= np.random.uniform(-5,5,n4)
    x3_4= np.random.uniform(-5,-2.5,n4)
    last_fnc_4=20 + pow(x3_4,2) + pow(y3_4,2) - 10 * (np.cos(2 * math.pi * x3_4 ) + np.cos(2 * math.pi * y3_4 ))
    fcn3_4=(np.sum(last_fnc_4))/n4 * 10 * 2.5
    
    y3_5= np.random.uniform(-5,5,n5)
    x3_5= np.random.uniform(2.5,5,n5)
    last_fnc_5=20 + pow(x3_5,2) + pow(y3_5,2) - 10 * (np.cos(2 * math.pi * x3_5 ) + np.cos(2 * math.pi * y3_5 ))
    fcn3_5=(np.sum(last_fnc_5))/n5 * 10 * 2.5
    
    stratified_lastfunction.append(fcn3_1+fcn3_2+fcn3_3+fcn3_4+fcn3_5)
    
var_stratlastfunc=np.var(stratified_lastfunction)
mean2_stratlastfunc=np.mean(stratified_lastfunction)


print('\nMean value for function 3 with stratification: ',mean2_stratlastfunc)
print('Variance value for function 3 with stratification',var_stratlastfunc)

