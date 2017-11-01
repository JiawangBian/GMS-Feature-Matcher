// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <ctime>
#include <string>
#include <boost/timer.hpp>
using namespace std;
using namespace cv;

#ifdef USE_GPU
	#include <opencv2/cudafeatures2d.hpp>
	using cuda::GpuMat;
#endif
