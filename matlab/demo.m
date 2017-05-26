clear; clc;
addpath('../data');

Compile;

I1 = imread('nn_left.jpg');
I2 = imread('nn_right.jpg');

I1 = imresize(I1, [640 480]);
I2 = imresize(I2, [640 480]);

I1gray = rgb2gray(I1);
I2gray = rgb2gray(I2);

% parameter
numP = 10000; 
scale = 0;
rotate = 0;

[X1, X2] = gms_match(I1gray, I2gray, numP, scale, rotate);

showMatchedFeatures(I1,I2,X1',X2','montage');

