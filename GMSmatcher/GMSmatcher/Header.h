// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include <stdio.h>
#include <tchar.h>
#include <Windows.h>

// TODO: reference additional headers your program requires here
#include <opencv2/opencv.hpp>
using namespace cv;
#ifdef _DEBUG
#define lnkLIB(name) name "d"
#else
#define lnkLIB(name) name
#endif

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#define cvLIB(name) lnkLIB("opencv_" name CV_VERSION_ID)
#pragma comment( lib, cvLIB("core"))
#pragma comment( lib, cvLIB("imgproc"))
#pragma comment( lib, cvLIB("highgui"))
#pragma comment(lib,  cvLIB("imgcodecs"))
#pragma comment(lib, cvLIB("features2d"))
#pragma comment(lib,cvLIB("videoio"))

#ifdef USE_GPU
	#pragma comment( lib, cvLIB("cudafeatures2d"))
	#include <opencv2/cudafeatures2d.hpp>
	using cuda::GpuMat;
#endif

#include <vector>
#include <iostream>
#include <ctime>
#include <map>
using namespace std;




