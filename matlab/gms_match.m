function [X1, X2] = gms_match(I1gray, I2gray, NumP, scale, rotate)
%   Image 1 2 must be gray scale uchar8
%   NumP: number of orb points (default 10,000 for 640x480 image)   
%   scale : 0 for not, 1 for multi-scale. (default no)
%   rotate: 0 for not, 1 for multi-direction. (default no)
if nargin < 5
    rotate = 0;
end

if nargin < 4
    scale = 0;
end

if nargin < 3
    NumP = 10000;
end
    
[X1, X2] = MexGMS(I1gray, I2gray, NumP, scale, rotate);


end