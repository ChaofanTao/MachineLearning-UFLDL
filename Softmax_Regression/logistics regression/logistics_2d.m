%%  ���ӷ��࣬201������7�������ֳ�3��
%�Ƚ��ж����࣬������1,2�����2,3�����1,3���������ȡ����������������������ࡣ
% * Logistic�������ڻع������������
% * ��0-1�������
% 
%% 
clc
clear
close all
%% Load data
% * ����Ԥ����--������Ϊֻ�����������������ѡ��1,2�����ݣ�
% ������ǩ��������Ϊ0��1������sigmod����Ӧ�� 
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
% initial 'weights' para,��ʼ��Ȩ����1������0
weights = ones(1,data_D);
%% training data weights
% * ����ݶ��½��㷨/����ѧϰ
for j = 1:500
    alpha = 0.1/j;
    for i = 1:length(train_data)
        data = train_data(i,1:end-1);
        h = 1.0/(1+exp(-(data*weights')));
        error = h - label_train(i);
        weights = weights - (alpha * error * data);
    end
end
%  * �����ݶ��㷨-����/����ѧϰ
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

%ע������Ȱ�1ת��Ϊ2���ٰ�-1ת��Ϊ1������������𶼱�Ϊ2
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