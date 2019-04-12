function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);
%注意labels：m*1,由于sparse是取labels的value为横坐标，所以value才决定groundtruth的大小（一个k*m的矩阵）
%与概率矩阵相对应点乘
groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
%minus a constant value to avoid overflow
%																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																														z
[N,M]=size(data);  
eta=bsxfun(@minus,theta*data,max(theta*data,[],1));  
eta=exp(eta);  
pij=bsxfun(@rdivide,eta,sum(eta));  
  
cost=-1./M*sum(sum(groundTruth.*log(pij)))+lambda/2*sum(sum(theta.^2));  
  
thetagrad=-1/M.*(groundTruth-pij)*data'+lambda.*thetagrad;  


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

