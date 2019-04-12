visibleSize = 8*8;   % number of input units 
hiddenSize = 25;     % number of hidden units 
sparsityParam = 0.01;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Implement sampleIMAGES
%
%  After implementing sampleIMAGES, the display_network command should
%  display a random sample of 200 patches from the dataset

patches = sample();
display_network2(patches(:,randi(size(patches,2),200,1)),8);

%  Obtain random parameters theta
theta = initializeParameters2(hiddenSize, visibleSize);

iter=300;
J=zeros(iter,1);
G=zeros(iter,size(theta,1));
for i=1:iter
[cost, grad] = vertorized_autocoder2(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, patches);
J(i)=cost;
G(i,:)=grad';
end
%%======================================================================
%% STEP 3: Gradient Checking
%
% Hint: If you are debugging your code, performing gradient checking on smaller models 
% and smaller training sets (e.g., using only 10 training examples and 1-2 hidden 
% units) may speed things up.

% First, lets make sure your numerical gradient computation is correct for a
% simple function.  After you have implemented computeNumericalGradient.m,
% run the following: 
checkNumericalGradient2();

% Now we can use it to check your cost function and derivative calculations
% for the sparse autoencoder.  
numgrad = computeNumericalGradient2( @(x) sparseAutoencoderCost(x, visibleSize, ...
                                                  hiddenSize, lambda, ...
                                                  sparsityParam, beta, ...
                                                  patches), theta);

% Use this to visually compare the gradients side by side
disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff) % Should be small. In our implementation, these values are usually less than 1e-9

%% STEP 5: Visualization 

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network2(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 


