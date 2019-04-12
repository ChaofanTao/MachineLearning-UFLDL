function patches = sampleIMAGES()
% sampleIMAGES
% Returns 10000 patches for training

load IMAGES

patchsize = 8;  % we'll use 8x8 patches 
numpatches = 10000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1
size_image=size(IMAGES);
i=randi(size_image(1)-patchsize+1,1,numpatches);
j=randi(size_image(2)-patchsize+1,1,numpatches);
k=randi(size_image(3),1,numpatches);
for num=1:numpatches
    patches(:,num)=reshape( IMAGES( i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,k(num)),1, patchsize*patchsize);
end
%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);
display_network(patches(:,randi(size(patches,2),200,1)),8);
end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;
%patches(:)是把这个矩阵变成一个向量。然后std，也就是
%求所有10000个掩模所有像素值的标准差，这里如果你不是很明白为什么要把所以样本一块求标准差，
%而不是一个样本组一个样本组的求标准差。并且还最后把所有像素的标准差*3。解释一下，以前求
%数据都是一个样本组一个样本组进行归一化，那是把数据进行归一化，也就是均值为0，方差为1，但是
%这样并不能保证所有的数据都在[-1,1]之间，只是方差为1。但是我们知道我们把所有数据减去均值
%的话，那么所有数据的99.7%都会落在[-3*标准差，3*标准差]之间，所以我们只需要把剩下的0.03%的数据都
%置成-3*标准差或3*标准差即可。这样所有数据都在[-3*标准差，3*标准差]之间，然后我们除以
%3*标准差，那么素有数据都会在[-1,1]之间了。注意这里是将所有数据一块处理，而不需要一组一组样本处理，
%因为这里并不需要处理后的每个样本满足标准正态分布:均值为0，方差为1那种关系，也不可能满足，所以
%数据一块处理即可。min(patches,pstd)把矩阵patches中大于pstd的数换成pstd,函数max同理（替换小值）
% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
