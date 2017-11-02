# Usage of gms_matcher.py

Cntains 2 classes, viz. GmsMatcher and GmsRobe. 

## GmsMatcher 
This class is the original work as presented in the paper. 
```
@inproceedings{bian2017gms,
  title={GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence},
  author={JiaWang Bian and Wen-Yan Lin and Yasuyuki Matsushita and Sai-Kit Yeung and Tan Dat Nguyen and Ming-Ming Cheng},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```

For demo usage see function, `demo_gms_original_implementation()` in `demo.py`


## GmsRobe
This class extents the functionality of GmsMatcher.
The GmsMatcher returns an array of DMatch. This might be sometimes convinent to use. 
As you need to make reference to the keypoints to get to the image co-ordinates in question. 

GmsRobe class defines the following functions
     - match2( imC, imP )<br/>
		Given current image(imC) and prev image(imP) returns GMS matches co-ordinates (x,y)_i
            Essentially this is a thin wrapper around the original GmsMatcher class


     - match3( imC, imP, imCm )<br/>
		To find 3way correspondences, ie. a set of point co-ordinates in imC which also occur in imP and imCm. a<-->b<-->c. Note: The order in which the images is given is critical. 


     - match2_guided( imC, pt1, imP ) <br/>
		Given current image (imC), with pts_C 2xN as input points on imC, the
        objective is to find these points in previous image (imP)

        The way we do this is to first compute all the matches between imC and imP.
        Then filter these matches to include only those pysically close to input points.
		

Following functions return 2xN matrix with (x,y) of matches.
