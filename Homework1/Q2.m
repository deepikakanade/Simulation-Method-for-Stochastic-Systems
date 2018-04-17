clc;

%Initialization of variables
k = 50;    %k = 5,10,30,50
prob = 0.5;
samp = 300;
count1 = zeros(samp);
count_total = zeros(k+1,1);

%Generate random numbers and count number of successes in all the trials
for j = 1:samp
    count = 0;
    for i = 1:k
        A(i) = rand(1);
        if A(i) > prob
        count = count + 1;
        end
    end
    count1(j) = count; 
end

%Count of Successes in all the samples
for j = 1:k+1
    for i = 1:samp
       if count1(i) == j-1
           count_total(j,1) = count_total(j,1)+1;
       end
    end
end

%Plotting histogram
figure(1);
bar(0:k,count_total);
xlim([0,k]);
xlabel('Value of k');
ylabel('Count');
title('Counting Successes for Bernoulli Random Variable');


