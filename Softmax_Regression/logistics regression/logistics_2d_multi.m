%%  
% * Logistic方法用于回归分析与分�?
% 
%% 
clc
clear
close all
%% Load data
% * 数据预处�?这里因为只分两类情况，所以挑选出1,2类数据，
% 并将标签重新设置丿丿，方便sigmod函数应用 
data = importdata('seed_data.txt');
%选择训练样本个数
num_train = 120;
%随机选择序列
choose = randperm(length(data));
train_data = data(choose(1:num_train),:);
label_train = train_data(:,end);
test_data = data(choose(num_train+1:end),:);
label_test = test_data(:,end);
data_D = size(train_data,2) - 1;
% initial 'weights' para
% label number
cluster_num = 3;
% n*(n-1)/2
weights = ones(cluster_num*(cluster_num-1)/2,data_D);
%% training data weights
% * 随机梯度上升算法-在线学习
t = 1;
for index1 = 1:cluster_num-1
    for index2 = index1+1:cluster_num
        index_c = find((train_data(:,8)==index1)|(train_data(:,8)==index2));
        train_data_temp = train_data(index_c,:);
        train_data_temp(find((train_data_temp(:,8)==index1)),8) = 0;
        train_data_temp(find((train_data_temp(:,8)==index2)),8) = 1;
        %----------------------------
        for j = 1:200
            alpha = 0.1/j;
            for i = 1:length(train_data_temp)
                data = train_data_temp(i,1:end-1);
                h = 1.0/(1+exp(-(data*weights(t,:)')));
                error = train_data_temp(i,8) - h;
                weights(t,:) = weights(t,:) + (alpha * error * data);
            end
        end
        t = t + 1;
    end
end
%
%% test itself (the training data)
%
predict = zeros(length(train_data),cluster_num);
for i = 1:length(train_data)
    t = 1;
    data = train_data(i,1:end-1);
    for index1 = 1:cluster_num-1
        for index2 = index1+1:cluster_num
            h = 1.0/(1+exp(-(data*weights(t,:)')));
            if h < 0.5
                predict(i,index1) = predict(i,index1) + 1;
            else
                predict(i,index2) = predict(i,index2) + 1;
            end
            t = t + 1;
        end
    end
end 
[~,predict] = max((predict'));
%% show the result
figure;
plot(train_data(:,8),'+')
hold on
plot(predict,'or');
hold on 
plot(abs(predict'-train_data(:,8)));
axis([0,length(train_data),0,5])
accuracy = length(find(predict'==train_data(:,8)))/length(train_data);
title(['predict Training Data and the accuracy is :',num2str(accuracy)]);
%
%% predict the testing data
%
predict = zeros(length(test_data),cluster_num);
for i = 1:length(test_data)
    t = 1;
    data = test_data(i,1:end-1);
    for index1 = 1:cluster_num-1
        for index2 = index1+1:cluster_num
            h = 1.0/(1+exp(-(data*weights(t,:)')));
            if h < 0.5
                predict(i,index1) = predict(i,index1) + 1;
            else
                predict(i,index2) = predict(i,index2) + 1;
            end
            t = t + 1;
        end
    end
end 
[~,predict] = max((predict'));
%% show the result
figure;
plot(test_data(:,8),'+')
hold on
plot(predict,'or');
hold on 
plot(abs(predict'-test_data(:,8)));
axis([0,length(test_data),0,5])
accuracy = length(find(predict'==test_data(:,8)))/length(test_data);
title(['predict the testing data and the accuracy is :',num2str(accuracy)]);