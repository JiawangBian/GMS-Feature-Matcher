// GridMatch.cpp : Defines the entry point for the console application.

//#define USE_GPU 
#include "stdafx.h"
#include "GMS.h"
#include "VideoMatch.h"

void GridMatch(Mat &img1, Mat &img2);

void runImagePair(){
	Mat img1 = imread("./data/nn_left.jpg");
	Mat img2 = imread("./data/nn_right.jpg");

	imresize(img1, 480);
	imresize(img2, 480);

	GridMatch(img1, img2);
}

void runVideo(){
	string filename = "./data/chair.mp4";
	testVideo(filename);
}


int _tmain(int argc, _TCHAR* argv[])
{
#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0){ cuda::setDevice(0); }
#endif // USE_GPU

	runImagePair();
//	runVideo();

	return 0;
}


void GridMatch(Mat &img1, Mat &img2){
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_grid;

	Ptr<ORB> orb = ORB::create(10000);
	orb->setFastThreshold(0);
	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);

#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
#endif

	// GMS filter
	GMS gms;
	gms.init(img1.size(), img2.size(), kp1, kp2, matches_all);
	gms.setParameter(20, 20);
	matches_grid = gms.getInlier();

	cout << "Get total " << matches_grid.size() << " matches." << endl;

	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_grid, 1);
	imshow("show", show);
	waitKey();
}


