# GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence

![alt tag](http://mmcheng.net/wp-content/uploads/2017/03/dog_ours.jpg)



## Publication:

[JiaWang Bian](http://jwbian.net), Wen-Yan Lin, [Yasuyuki Matsushita](http://www-infobiz.ist.osaka-u.ac.jp/user/matsushita/index.html), [Sai-Kit Yeung](http://people.sutd.edu.sg/~saikit/), Tan Dat Nguyen, [Ming-Ming Cheng](http://mmcheng.net)

**GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence**  **IEEE CVPR, 2017** 

[[Project Page](http://jwbian.net/gms)] [[pdf](http://jwbian.net/Papers/GMS_CVPR17.pdf)] [[Bib](http://jwbian.net/Papers/bian2017gms.txt)] [[Code](https://github.com/JiawangBian/GMS-Feature-Matcher)] [[Youtube](https://youtu.be/3SlBqspLbxI)]


	
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
		1.#define USE_GPU" will need gpu cudafeatures2d module for nearest neighbor match, 
			using cpu match by commenting it.
	
	In gms_matcher.h
				
		2.	#define THRESH_FACTOR 6			// factor for calculating threshold
				The higher, the less matches, vice verse
				
		3. 	int GetInlierMask(vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false)
				You can open multi-scale and rotation if your image pair contains that. 
				

## If you like this work, please cite our paper
	@inproceedings{bian2017gms,
 	  title={GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence},
  	  author={JiaWang Bian and Wen-Yan Lin and Yasuyuki Matsushita and Sai-Kit Yeung and Tan Dat Nguyen and Ming-Ming Cheng},
  	  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  	  year={2017}
	}



