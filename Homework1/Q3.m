clc;
close all;

%Initialization of variables
n=20;
size = nchoosek(20,2);   %To choose 190 combinations
prob = 0.05;
samp = 1000;  %100,500,1000 samples
temp1=zeros(samp);
count_final=zeros(size+1,1);

%Generate Random Numbers and count successes for all trials
for j = 1:samp
    count=0;
    for i = 1:size
        S(i) = rand(1);
        if S(i)<=prob
        count = count + 1;
        end
    end
    temp1(j) = count; 
end

%count successes for all samples
for j = 1:size+1
    for i = 1:samp
       if temp1(i) == j-1
           count_final(j,1) = count_final(j,1)+1;
       end
    end
end

%Potting Histogram
figure(1);
bar(0:size,count_final);
xlim([0,25]);
ylim([0,150]);
xlabel('Number of edges selected');
ylabel('Count of the occurence of the edges');


