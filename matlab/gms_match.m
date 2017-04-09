function [X1, X2] = gms_match(I1gray, I2gray, NumP, Rotated)
%   Image 1 2 must be gray scale uchar8
%   NumP: number of orb points (default 10,000 for 640x480 image)   
%   Rotated: 0 for not, 1 for rotated pair. (default no)

if nargin < 4
    Rotated = 0;
end

if nargin < 3
    NumP = 10000;
end
    
[X1, X2] = MexGMS(I1gray, I2gray, NumP, Rotated);


end