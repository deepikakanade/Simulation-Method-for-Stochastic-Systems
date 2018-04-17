clc;
close all;

%Initialization of variables
size = 10;
x = rand(size);   %To generate 100 uniform random numbers
A=2;
count = zeros(size*size,1);
count_final = zeros(A,1);
count1=0;
count0=0;

%Generate a fair Bernoulli trial for 100 samples
for i=1:100
    if(x(i)>=0.5)
        count(i)=1;
        count1=count1+1;
    else
        count(i)=0;
        count0=count0+1;
    end
end

%count successes for all samples
for j = 1:size*size
    for i = 1:A
       if count(j) == i-1
           count_final(i,1) = count_final(i,1)+1;
       end
    end
end

disp("No. of Successes are: ");
disp(count1)
disp("No. of Failures are: ");
disp(count0);

%Plotting Histogram of the data
bar(0:1,count_final);
xlabel("Success=1 and Failure=0");
ylabel("Count for each Sucess/Failure");
title("Routine for one fair Bernoulli Trial for 100 samples");


