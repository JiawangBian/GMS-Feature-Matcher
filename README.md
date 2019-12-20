# GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence

![alt tag](http://mmcheng.net/wp-content/uploads/2017/03/dog_ours.jpg)



## Publication:

[JiaWang Bian](http://jwbian.net), Wen-Yan Lin, Yasuyuki Matsushita, Sai-Kit Yeung, Tan Dat Nguyen, Ming-Ming Cheng, **GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence**, **CVPR 2017**, [[Project Page](http://jwbian.net/gms)] [[pdf](http://jwbian.net/Papers/GMS_CVPR17.pdf)] [[Bib](http://jwbian.net/Papers/bian2017gms.txt)] [[Code](https://github.com/JiawangBian/GMS-Feature-Matcher)] [[Youtube](https://youtu.be/3SlBqspLbxI)]

[JiaWang Bian](http://jwbian.net), Wen-Yan Lin, Yun Liu, Le Zhang, Sai-Kit Yeung, Ming-Ming Cheng, Ian Reid, **GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence**, **IJCV 2019**, [[pdf](https://link.springer.com/content/pdf/10.1007%2Fs11263-019-01280-3.pdf)] 

## Other Resouces

  The method has been integrated into OpenCV library (see xfeatures2d in [opencv_contrib](https://github.com/opencv/opencv_contrib)).

  The paper was selected and reviewed by [Computer Vision News](http://www.rsipvision.com/ComputerVisionNews-2017August/#48).

  More experiments are shown in [FM-Bench](https://jwbian.net/fm-bench).
	
## Usage

Requirement:

	1.OpenCV 3.0 or later (for IO and ORB features, necessary)

	2.cudafeatures2d module(for gpu nearest neighbor, optional)

C++ Example:

	Image pair demo in demo.cpp.
	
Matlab Example
	
	You should compile the code with opencv library firstly(see the 'Compile.m').

Python Example:
	
	Use Python3 to run gms_matcher script.
	
Tune Parameters:

	In demo.cpp
		1.	#define USE_GPU" (need cudafeatures2d module) 
				using cpu mode by commenting it.
				
		2.	For high-resolution images, we suggest using 100K features with setFastThreshod(5);
		
		3.	For low-resolution (like VGA) images, we suggest using 10K features with setFastThreshod(0);
	
	In gms_matcher.h
				
		2.	#define THRESH_FACTOR 6			
				The higher, the less matches。
				
		3. 	int GetInlierMask(vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false)
				Set WithScale to be true for unordered image matching and false for video matching.
				

## If you use this work, please cite our paper
	@inproceedings{bian2017gms,
 	  title={GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence},
  	  author={JiaWang Bian and Wen-Yan Lin and Yasuyuki Matsushita and Sai-Kit Yeung and Tan Dat Nguyen and Ming-Ming Cheng},
  	  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  	  year={2017}
	}



