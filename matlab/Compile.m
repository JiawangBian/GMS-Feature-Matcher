%%
% You should compile the code with opencv  
%

% opencv path
OpenCV = 'C:/SDK/OpenCV/'; %Input Your OpenCV Path
version = '320'; % Input Your OpenCV Version

IPath = ['-I' OpenCV 'include'];
LPath = ['-L' OpenCV 'x64/vc14/lib'];

lib1 = ['-lopencv_core' version '.lib'];
lib2 = ['-lopencv_features2d' version '.lib'];
lib3 = ['-lopencv_imgcodecs' version '.lib'];
lib4 = ['-lopencv_imgproc' version '.lib'];


mex -setup
mex ('MexGMS.cpp', '-I../include/', IPath, LPath, lib1, lib2, lib3, lib4); 

