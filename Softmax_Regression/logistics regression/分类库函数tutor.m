%Ŀǰ�˽⵽��MATLAB�з������У�K���ڷ����������ɭ�ַ����������ر�Ҷ˹������ѧϰ���������������������֧�����������ֽ�����Ҫ����ʹ�÷����ܽ����£�����ϸ����ο�MATLAB �����ļ���
%��
%��ѵ��������train_data             % ����ÿ��һ��������ÿ��һ������
%����ѵ��������ǩ��train_label       % ������
%��������������test_data
%��������������ǩ��test_label
 
%K���ڷ����� ��KNN��
mdl = ClassificationKNN.fit(train_data,train_label,'NumNeighbors',1);
predict_label = predict(mdl, test_data);
accuracy = length(find(predict_label == test_label))/length(test_label)*100
               
 
%���ɭ�ַ�������Random Forest��
B = TreeBagger(nTree,train_data,train_label);
predict_label = predict(B,test_data);
 
 
%���ر�Ҷ˹ ��Naive Bayes��
nb = NaiveBayes.fit(train_data, train_label);
predict_label   =       predict(nb, test_data);
accuracy         =       length(find(predict_label == test_label))/length(test_label)*100;
 
 
%����ѧϰ������Ensembles for Boosting, Bagging, or Random Subspace��
ens = fitensemble(train_data,train_label,'AdaBoostM1' ,100,'tree','type','classification');
predict_label   =       predict(ens, test_data);
 
 
%���������������discriminant analysis classifier��
obj = ClassificationDiscriminant.fit(train_data, train_label);
predict_label   =       predict(obj, test_data);
 
 
%֧����������Support Vector Machine, SVM��
SVMStruct = svmtrain(train_data, train_label);
predict_label  = svmclassify(SVMStruct, test_data)

%���Լ��������£�

[python] view plain copy
clc  
clear all   
 load('wdtFeature');  
   
%  ����ѵ��������train_data             % ����ÿ��һ��������ÿ��һ������  
% ����ѵ��������ǩ��train_label       % ������  
% ��������������test_data  
% ��������������ǩ��test_label  
 train_data = traindata'  
 train_label = trainlabel'  
 test_data = testdata'  
 test_label = testlabel'  
%  K���ڷ����� ��KNN��  
% mdl = ClassificationKNN.fit(train_data,train_label,'NumNeighbors',1);  
% predict_label   =       predict(mdl, test_data);  
% accuracy         =       length(find(predict_label == test_label))/length(test_label)*100  
%                  
%  94%  
% ���ɭ�ַ�������Random Forest��  
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
% ���ر�Ҷ˹ ��Na?ve Bayes��  
% nb = NaiveBayes.fit(train_data, train_label);  
% predict_label   =       predict(nb, test_data);  
% accuracy         =       length(find(predict_label == test_label))/length(test_label)*100;  
%   
%   
% % ��� 81%  
% % **********************************************************************  
% % ����ѧϰ������Ensembles for Boosting, Bagging, or Random Subspace��  
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
  
% ��� 97%  
% **********************************************************************  
% ���������������discriminant analysis classifier��  
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
% ֧����������Support Vector Machine, SVM��  
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