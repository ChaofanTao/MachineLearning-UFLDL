%目前了解到的MATLAB中分类器有：K近邻分类器，随机森林分类器，朴素贝叶斯，集成学习方法，鉴别分析分类器，支持向量机。现将其主要函数使用方法总结如下，更多细节需参考MATLAB 帮助文件。
%设
%　训练样本：train_data             % 矩阵，每行一个样本，每列一个特征
%　　训练样本标签：train_label       % 列向量
%　　测试样本：test_data
%　　测试样本标签：test_label
 
%K近邻分类器 （KNN）
mdl = ClassificationKNN.fit(train_data,train_label,'NumNeighbors',1);
predict_label = predict(mdl, test_data);
accuracy = length(find(predict_label == test_label))/length(test_label)*100
               
 
%随机森林分类器（Random Forest）
B = TreeBagger(nTree,train_data,train_label);
predict_label = predict(B,test_data);
 
 
%朴素贝叶斯 （Naive Bayes）
nb = NaiveBayes.fit(train_data, train_label);
predict_label   =       predict(nb, test_data);
accuracy         =       length(find(predict_label == test_label))/length(test_label)*100;
 
 
%集成学习方法（Ensembles for Boosting, Bagging, or Random Subspace）
ens = fitensemble(train_data,train_label,'AdaBoostM1' ,100,'tree','type','classification');
predict_label   =       predict(ens, test_data);
 
 
%鉴别分析分类器（discriminant analysis classifier）
obj = ClassificationDiscriminant.fit(train_data, train_label);
predict_label   =       predict(obj, test_data);
 
 
%支持向量机（Support Vector Machine, SVM）
SVMStruct = svmtrain(train_data, train_label);
predict_label  = svmclassify(SVMStruct, test_data)

%我自己代码如下：

[python] view plain copy
clc  
clear all   
 load('wdtFeature');  
   
%  　　训练样本：train_data             % 矩阵，每行一个样本，每列一个特征  
% 　　训练样本标签：train_label       % 列向量  
% 　　测试样本：test_data  
% 　　测试样本标签：test_label  
 train_data = traindata'  
 train_label = trainlabel'  
 test_data = testdata'  
 test_label = testlabel'  
%  K近邻分类器 （KNN）  
% mdl = ClassificationKNN.fit(train_data,train_label,'NumNeighbors',1);  
% predict_label   =       predict(mdl, test_data);  
% accuracy         =       length(find(predict_label == test_label))/length(test_label)*100  
%                  
%  94%  
% 随机森林分类器（Random Forest）  
% nTree = 5  
% B = TreeBagger(nTree,train_data,train_label);  
% predict_label = predict(B,test_data);  
%    
% m=0;  
% n=0;  
% for i=1:50  
%     if predict_label{i,1}>0  
%         m=m+1;  
%     end  
%     if predict_label{i+50,1}<0  
%         n=n+1;  
%     end  
% end  
%   
% s=m+n  
% r=s/100  
  
%  result 50%  
  
% **********************************************************************  
% 朴素贝叶斯 （Na?ve Bayes）  
% nb = NaiveBayes.fit(train_data, train_label);  
% predict_label   =       predict(nb, test_data);  
% accuracy         =       length(find(predict_label == test_label))/length(test_label)*100;  
%   
%   
% % 结果 81%  
% % **********************************************************************  
% % 集成学习方法（Ensembles for Boosting, Bagging, or Random Subspace）  
% ens = fitensemble(train_data,train_label,'AdaBoostM1' ,100,'tree','type','classification');  
% predict_label   =       predict(ens, test_data);  
%   
% m=0;  
% n=0;  
% for i=1:50  
%     if predict_label(i,1)>0  
%         m=m+1;  
%     end  
%     if predict_label(i+50,1)<0  
%         n=n+1;  
%     end  
% end  
%   
% s=m+n  
% r=s/100  
  
% 结果 97%  
% **********************************************************************  
% 鉴别分析分类器（discriminant analysis classifier）  
% obj = ClassificationDiscriminant.fit(train_data, train_label);  
% predict_label   =       predict(obj, test_data);  
%    
% m=0;  
% n=0;  
% for i=1:50  
%     if predict_label(i,1)>0  
%         m=m+1;  
%     end  
%     if predict_label(i+50,1)<0  
%         n=n+1;  
%     end  
% end  
%   
% s=m+n  
% r=s/100  
%  result 86%  
% **********************************************************************  
% 支持向量机（Support Vector Machine, SVM）  
SVMStruct = svmtrain(train_data, train_label);  
predict_label  = svmclassify(SVMStruct, test_data)  
m=0;  
n=0;  
for i=1:50  
    if predict_label(i,1)>0  
        m=m+1;  
    end  
    if predict_label(i+50,1)<0  
        n=n+1;  
    end  
end  
  
s=m+n  
r=s/100  
  
%  result 86%