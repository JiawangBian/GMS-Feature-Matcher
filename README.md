# GMS: Grid-base Motion Statistics for Fast, Ultra-robust Feature Correspondence

Abstract

It has been demonstrated that incorporating smoothness constraints with feature matching enables ultra-robust matching. However, such formulations are both complex and slow, making them unsuitable for video applications. This paper proposes GMS (Grid-based Motion Statistics), a simple means of encapsulating motion smoothness as the statistical likelihood of a certain number of matches in a region. GMS enables translation of high match numbers into high match quality. When coupled with our grid accelerated score evaluation, it provides a real-time, ultra-robust correspondence system. Evaluation on videos, with low textures, blurs and wide-baselines show GMS consistently outperforms other real-time matchers and can achieve parity with more sophisticated, much slower techniques.

Citation

If you are using the code provided here in a publication, please cite our paper: 

@inproceedings{cvpr2017gms,

  title={ {GMS}: Grid-base Motion Statistics for Fast, Ultra-robust Feature Correspondence},
  
  author={JiaWang Bian and Daniel Lin and Yasuyuki Matsushita and Sai-Kit Yeung and Tan Dat Nguyen and Ming-Ming Cheng},
  
  booktitle={IEEE CVPR},
  
  year={2017},
  
}



Requirement:

	1.OpenCV 3.0 or later (for IO and ORB features, necessary)

	2.cudafeatures2d module(for gpu nearest neighbor, optional)

How to run:

	Image pari demo and video demo in demo.cpp.

	(Note:	Please use gpu match when you run video demo.)
	
Tune Parameters:

	In Main.cpp
		1.#define USE_GPU" will need gpu cudafeatures2d module for nearest neighbor match, 
			using cpu match by commenting it.
	
	In GMS.h
		2.	#define LocalTheshold 1			// Chosing threshold has two ways : global and local
				1 for local threshold, 0 for global threshold;
				
		3.	#define TreshFactor 6			// factor for calculating threshold
				The higher, the less matches, vice verse
				
		4. 	GMS::setParameter(int num1, int num2 )
				The grid number = num * num for left and right image.
				num = 20 is a normal option.
				
		5. 	GMS::getInlier(int withRS)
				0 means no ration and no scale version (default)
				1 means with rotatio and scale change. (may better but slower)


