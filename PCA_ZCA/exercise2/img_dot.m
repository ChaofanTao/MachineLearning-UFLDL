
img = imread('C:\uestc.jpg');
figure('name','Raw images');

img=im2double(img);
img=rgb2gray(img)
imshow(img)
x_mean=mean(img,1);
x_mean_rep=repmat(x_mean,size(img,1),1);
img=img-x_mean_rep;
