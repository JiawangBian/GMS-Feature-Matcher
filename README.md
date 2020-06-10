# GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence

![alt tag](http://mmcheng.net/wp-content/uploads/2017/03/dog_ours.jpg)



## Publication:

[JiaWang Bian](http://jwbian.net), Wen-Yan Lin, Yasuyuki Matsushita, Sai-Kit Yeung, Tan Dat Nguyen, Ming-Ming Cheng, **GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence**, **CVPR 2017**, [[Project Page](http://jwbian.net/gms)] [[pdf](http://jwbian.net/Papers/GMS_CVPR17.pdf)] [[Bib](http://jwbian.net/Papers/bian2017gms.txt)] [[Code](https://github.com/JiawangBian/GMS-Feature-Matcher)] [[Youtube](https://youtu.be/3SlBqspLbxI)]

[JiaWang Bian](http://jwbian.net), Wen-Yan Lin, Yun Liu, Le Zhang, Sai-Kit Yeung, Ming-Ming Cheng, Ian Reid, **GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence**, **IJCV 2020**, [[pdf](https://link.springer.com/content/pdf/10.1007%2Fs11263-019-01280-3.pdf)] 


## Other Resouces

  The method has been integrated into OpenCV library (see [xfeatures2d.matchGMS](https://docs.opencv.org/master/db/dd9/group__xfeatures2d__match.html)).
  
  More experiments are shown in [FM-Bench](https://jwbian.net/fm-bench).

  The paper was selected and reviewed by [Computer Vision News](http://www.rsipvision.com/ComputerVisionNews-2017August/#48).


## If you find this work useful in your research, please consider citing our paper:
	
	@article{Bian2020gms,
  		title={{GMS}: Grid-based Motion Statistics for Fast, Ultra-Robust Feature Correspondence},
  		author={Bian, JiaWang and Lin, Wen-Yan and Liu, Yun and Zhang, Le and Yeung, Sai-Kit and Cheng, Ming-Ming and Reid, Ian},
  		journal={International Journal of Computer Vision (IJCV)},
  		year={2020}
	}


## Usage

Requirement:

	1.OpenCV 3.0 or later (for ORB features, necessary)

	2.cudafeatures2d module(for gpu nearest neighbor, optional)
	
	3.OpenCV xfeatures2D moudle (if using the opencv built-in GMS function) 

C++ Example:

	See src/demo.cpp


Python Example:
	
	Go to "python" folder. Run "python3 opencv_demo.py". 
	(You need install opencv_contrib by "pip install opencv-contrib-python")
	
	
Matlab Example:
	
	1. Go to "matlab" folder. Compile the code with OpenCV ('Compile.m'), and run 'demo.m'.

External Examples:

   [OpenCV C++ demo](https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/samples/gms_matcher.cpp) and [Mexopencv example](http://amroamroamro.github.io/mexopencv/opencv_contrib/gms_matcher_img_demo.html)


Tuning Parameters:

	In src/demo.cpp
		1.	#define USE_GPU" (need cudafeatures2d module) 
				using cpu mode by commenting it.
				
		2.	We suggest using SIFT features for accuracy, and using ORB features for speed.

	
	In gms_matcher.h
				
		2.	#define THRESH_FACTOR 6		
				Set it higher for more input matches, and lower for the fewer input matches.
				Often 6 for ORB all matches, and 4 or 3 for SIFT matches (after ratio test).
				
		3. 	int GetInlierMask(vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false)
				Set WithScale to be true for wide-baseline matching and false for video matching.
				Set WithRotation to be true if images have significant reative rotations.
				

## Related projects

 * [FM-Bench](https://github.com/JiawangBian/FM-Bench) (BMVC 2019, More evaluation details for GMS.)


