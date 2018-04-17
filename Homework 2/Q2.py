##Generation of exponential random variable from uniform trials using inverse cdf method
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

N=1000
u=np.zeros(N)
x=np.zeros(N)
theta=0.2   #Value of average waiting time

#Generation of exponential random variable from uniform trials using inverse cdf method
uniform_trials=np.array(np.random.uniform(size=1000))
expo_observed_values=-theta * np.log(1-uniform_trials)

#Generation of exponential randon variable using in-built function
expo_expected_values=np.array(np.random.exponential(theta,size=1000))

#Taking the x axis values of expected and observed random variables
expected=plt.hist(expo_expected_values,bins=30)[0]
observed=plt.hist(expo_observed_values,bins=30)[0]

#Running the chi-square goodness of fit test
[chi_square,p]=scipy.stats.chisquare(observed[0:15], expected[0:15])
print('Value of chi-square')
print (chi_square)
print('Value of p')
print (p)
#Running the ks goodness of fit test
[KS,p]=scipy.stats.kstest(expo_observed_values, 'expon')
print('Value of KS')
print (KS)
print('Value of p')
print (p)

#plt.figure()
#plt.hist(expo_expected_values)

#To calculate the number of exponential time intervals in 1 unit time
sum=0
count=0
count_array=[]
for i in range(0,1000):
    sum=sum+expo_observed_values[i]
    count=count+1
    if sum>1:
        count_array.append(count)
        count=0
        sum = 0
        
#Plotting the histogram of the number of exponential time intervals in 1 unit time
plt.figure()
plt.suptitle('Number of exponentially distributed time intervals in 1 unit time')
plt.xlabel('Number of exponential time intervals in 1 unit time')
plt.ylabel('Count of exponential time intervals in 1 unit time')
plt.hist(count_array)
