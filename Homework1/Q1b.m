clc;

%Initialization of variables
bern = 7;
prob = 0.5;
samp = 100;
temp1=zeros(samp);
temp2=zeros(bern+1,1);

%Generate Random Numbers and count successes in 7 trials for 100 samples
for j = 1:samp
    count=0;
    for i = 1:bern
        S(i) = rand(1);
        if S(i)>prob
        count = count + 1;
        end
    end
    temp1(j) = count; 
end

%count successes for all samples
for j = 1:bern+1
    for i = 1:samp
       if temp1(i) == j-1
           temp2(j,1) = temp2(j,1)+1;
       end
    end
end

%Potting Histogram
figure(1);
bar(0:bern,temp2);
xlabel('Number of successes');
ylabel('Count for each success');
