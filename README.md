# GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence

![alt tag](http://mmcheng.net/wp-content/uploads/2017/03/dog_ours.jpg)



## Publication:

[JiaWang Bian](http://jwbian.net), Wen-Yan Lin, [Yasuyuki Matsushita](http://www-infobiz.ist.osaka-u.ac.jp/user/matsushita/index.html), [Sai-Kit Yeung](http://people.sutd.edu.sg/~saikit/), Tan Dat Nguyen, [Ming-Ming Cheng](http://mmcheng.net)

**GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence**  **IEEE CVPR, 2017** 

[[Project Page](http://jwbian.net/gms)] [[pdf](http://jwbian.net/Papers/GMS_CVPR17.pdf)] [[Code](https://github.com/JiawangBian/GMS-Feature-Matcher)] [[Video Demo](http://jwbian.net/Demo/gms_matching_demo.mp4)]


	
## Video Matching Demo
	
[![IMAGE ALT TEXT HERE](http://jwbian.net/wp-content/uploads/2017/04/matching_demo_chair-e1492913756279.png)](https://youtu.be/3SlBqspLbxI)   [![IMAGE ALT TEXT HERE](http://jwbian.net/wp-content/uploads/2017/04/matching_demo_tum-e1492913770981.png)](https://youtu.be/tjMpgno6k5A)   [![IMAGE ALT TEXT HERE](http://jwbian.net/wp-content/uploads/2017/04/matching_demo_car-e1492913739458.png)](https://youtu.be/TIVWTTQTkeI)


	
## Usage

Environment:

	The code can run on Windows, Linux, and Mac.

Requirement:

	1.OpenCV 3.0 or later (for IO and ORB features, necessary)

	2.cudafeatures2d module(for gpu nearest neighbor, optional)

Run:

	Image pair demo and video demo in demo.cpp.

	(Note:	Please use gpu match when you run video demo.)
	
Tune Parameters:

	In Main.cpp
		1.#define USE_GPU" will need gpu cudafeatures2d module for nearest neighbor match, 
			using cpu match by commenting it.
	
	In gms_matcher.h
				
		2.	#define THRESH_FACTOR 6			// factor for calculating threshold
				The higher, the less matches, vice verse
				
		3. 	int GetInlierMask(vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false)
				You can open multi-scale and rotation if your image pair contains that. 
				

## If you think this work is helpful, please cite
	@article{bian2017gms,
  	title={GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence},
  	author={JiaWang Bian and Wen-Yan Lin and Yasuyuki Matsushita and Sai-Kit Yeung and Tan Dat Nguyen and Ming-Ming Cheng},
  	booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  	year={2017}
	}


