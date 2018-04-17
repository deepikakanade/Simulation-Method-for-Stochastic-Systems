import numpy as np
import matplotlib.pyplot as plt

k = 5000
a = 0
b = 6

### Generate Uniform samples
uniform_trials= np.random.uniform(low=a, high=b, size=k)

### Get the maximum value of f(x) for Accept/Reject
cmax = 1.46

### Accept/Reject Routine
accept_reject = []
uniform_trials_double_rejection = np.random.uniform(low = a, high=cmax, size=k)

reject_count = 0
tot_count = 0
number_of_rejections = []
x_list = []
function_of_x_list = []

while (np.sum(accept_reject) < 1000): 
    x = np.random.uniform(low=a, high=b, size=1)
    #function_of_x = returnFunctionOfx(x)
    if ((x > 0) & (x <= 1)):
        function_of_x = 0.5 * np.random.beta(8,5)
    elif ((x > 4) & (x <= 5)):
        function_of_x =  0.5 * (x - 4)
    elif ((x > 5) & (x <= 6)):
        function_of_x = -0.5 * (x - 6)
    else:
        function_of_x = 0
    c = np.random.uniform(low = a, high=cmax, size=1)
    x_list.append(x)
    function_of_x_list.append(function_of_x)
    #print ("f(x): " , function_of_x)
    if (c <= function_of_x):
        tot_count += 1
        accept_reject.append(1)
        number_of_rejections.append(reject_count)
        reject_count = 0
        tot_count = 0
        #print (" -- Accept")
    else:
        accept_reject.append(0)
        reject_count += 1
        tot_count += 1
       # print (" -- Reject")

print ("Total Number of Iterations: ", len(accept_reject))
print ("Average number of samples rejected",np.mean(number_of_rejections))
print ("Rejection rate", float(np.sum(number_of_rejections)) / float(len(accept_reject)))


plt.figure()
plt.hist(function_of_x_list, bins=100)
plt.xlabel("f(x)")
plt.ylabel("Frequency")
plt.title("Histogram of f(x) - Bimodal")
plt.ylim([0,150])

