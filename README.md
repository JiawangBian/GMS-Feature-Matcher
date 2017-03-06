# GMS-Feature-Matcher
Feature Matching System (ORB + GMS)

Publication 

GMS: Grid-base Motion Statistics for Fast, Ultra-robust Feature correspondence, JW Bian, W Lin, Y Matsushita, SK Yeung, TD Nguyen, MM Cheng IEEE CVPR, 2017. [Project Page]

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


