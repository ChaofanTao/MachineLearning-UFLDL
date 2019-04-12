clc
clear

%svm验证logistics模型
data=importdata('seed_data.txt');
index=find( data(:,8)==1 | data(:,8)==2 );
data=data(index,:);
order=randperm(length(data));
num_train=50;
training0=data(order(1:num_train),:);
test0=data(order(num_train+1:end),:);
group_train=training0(:,end);
group_test=test0(:,end);
training0(:,end)=[];
test0(:,end)=[];
test=test0';
training=training0';
%mapstd按行进行归一化，我们需要按列（某个feature的所有样本数据）归一，故转置
[train_data, ps]=mapstd(training);
test_data = mapstd('apply' , test ,ps);
%比较而言，rbf的核函数对应精确度最高
s=svmtrain(train_data',group_train,'Method','SMO','Kernel_Function','rbf');

figure;
check_train=svmclassify(s,train_data');
err1=abs(check_train-group_train);
accuracy1=sum(check_train==group_train)/num_train
hold on
plot(check_train,'or')
plot(group_train,'b+')
plot(err1)
axis([0,num_train,0,3]);
title(['predict the training data and the accuracy is :',num2str(accuracy1)]);

figure;
check_test=svmclassify(s,test_data');
err2=abs(check_test-group_test);
accuracy2=sum(check_test==group_test)/(length(data)-num_train)
hold on
plot(check_test,'or')
plot(group_test,'b+')
plot(err2)
axis([0,length(data)-num_train,0,3]);
title(['predict the testing data and the accuracy is :',num2str(accuracy2)]);
