clc;

%Initialization of variables
size = 500;
prob = 0.5;
samp = 100;
sample = zeros(samp,1);
count = zeros(size,1);
count_final = zeros(samp,1);
max = 0;
temp= 0;

%Generate Random samples and find longest run of heads in all the trials
for i = 1:size
    max = 0;
    temp = 0;
    for j = 1:samp
      A(j) = rand(1);
        if A(j)>prob
            A(j) = 1;
        else
            A(j) = 0;
        end
    end
    for j = 1:samp-1
        if A(j) == 1 && A(j+1) == 1
            temp = temp+ 1;
        else
            temp = 0;
        end
        if max < temp
            max = temp;
        end
    end
    count(i) = max;
end

%Count the number of longest run of Heads for all samples
for i = 1:samp+1
    for j= 1:size
        if count(j) == i-1
            count_final(i) = count_final(i) + 1;
        end
    end
end

%Plotting Histogram
figure(1);
bar(1:samp,count_final');
xlim([1,20]);
xlabel('Longest run');
ylabel('Count');
title('Longest Run of Heads in 100 Bernoulli samples');


