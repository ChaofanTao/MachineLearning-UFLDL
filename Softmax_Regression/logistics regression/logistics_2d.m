%%  种子分类，201样本，7特征，分成3种
%先进行二分类，将里面1,2类或者2,3类或者1,3类的数据提取出来，单单对这两个类分类。
% * Logistic方法用于回归分析与分类设计
% * 简单0-1两类分类
% 
%% 
clc
clear
close all
%% Load data
% * 数据预处理--这里因为只分两类情况，所以挑选出1,2类数据，
% 并将标签重新设置为0与1，方便sigmod函数应用 
data = importdata('seed_data.txt');
index = find((data(:,8)==1)|(data(:,8)==2));
data = data(index,:);
data(:,8) = data(:,8) - 1;
num_train = 70;

choose = randperm(length(data));
train_data = data(choose(1:num_train),:);
label_train = train_data(:,end);
test_data = data(choose(num_train+1:end),:);
label_test = test_data(:,end);
data_D = size(train_data,2) - 1;
% initial 'weights' para,初始化权重用1，不是0
weights = ones(1,data_D);
%% training data weights
% * 随机梯度下降算法/在线学习
for j = 1:500
    alpha = 0.1/j;
    for i = 1:length(train_data)
        data = train_data(i,1:end-1);
        h = 1.0/(1+exp(-(data*weights')));
        error = h - label_train(i);
        weights = weights - (alpha * error * data);
    end
end
%  * 整体梯度算法-批量/离线学习
% for j = 1:2000
%     alpha = 0.1/j;
% %     alpha = 0.001;
%     data = train_data(:,1:end-1);
%     h = 1./(1+exp(-(data*weights')));
%     error = label_train - h;
%     weights = weights + (alpha * data' * error)';
% end
%% test itself (the training data)
diff = zeros(2,length(train_data));
for i = 1:length(train_data)
    data = train_data(i,1:end-1);
    h = 1.0/(1+exp(-(data*weights')));
    %compare to every label
    for j = 1:2 
        diff(j,i) = abs((j-1)-h);
    end
end
[~,predict] = min(diff);
%show the result
figure;
plot(label_train+1,'+')
hold on
plot(predict,'or');
hold on 
plot(abs(predict'-(label_train+1)));
axis([0,length(train_data),0,3])
accuracy_train = length(find(predict'==(label_train+1)))/length(train_data);
title(['predict Training Data and the accuracy is :',num2str(accuracy_train)]);
%% predict the testing data

predict=zeros(1,length(test_data));
for i = 1:length(test_data)
    data = test_data(i,1:end-1);
    h = 1.0/(1+exp(-(data*weights')));
    predict(i)=sign(h-0.5)
    %compare to every label
end

%注意必须先把1转化为2，再把-1转化为1。否则所有类别都变为2
predict(find(predict==1))=2;  
predict(find(predict==-1))=1;

% show the result
figure;
plot(label_test+1,'+')
hold on
plot(predict,'or');
hold on 
plot(abs(predict'-(label_test+1)));
axis([0,length(test_data),0,3])
accuracy = length(find(predict'==(label_test+1)))/length(test_data);
title(['predict the testing data and the accuracy is :',num2str(accuracy)]);