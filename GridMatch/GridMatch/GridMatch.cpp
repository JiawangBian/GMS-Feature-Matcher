// GridMatch.cpp : Defines the entry point for the console application.

#define USE_GPU 
#include "stdafx.h"
#include "GMS.h"

void imresize(Mat &src, int height){
	double ratio = src.rows * 1.0 / height;
	int width = static_cast<int>(src.cols * 1.0 / ratio);
	resize(src, src, Size(width, height));
}

Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier);

void GridMatch(Mat &img1, Mat &img2);

int _tmain(int argc, _TCHAR* argv[])
{
	//D:/DataSet/Strecha/fountain/urd/0000.png
	Mat img1 = imread("../data/teddy_left.png");
	Mat img2 = imread("../data/teddy_right.png");

	imresize(img1, 480);
	imresize(img2, 480);

#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0){ cuda::setDevice(0); }
#endif // USE_GPU
	GridMatch(img1, img2);


	return 0;
}


void GridMatch(Mat &img1, Mat &img2){
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_grid;

	clock_t bg, ed;
	Ptr<ORB> orb = ORB::create(10000);
	orb->setFastThreshold(0);
	bg = clock();
	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);
	ed = clock();
	cout << "Feature extraction time consuming: " << ed - bg << "ms" << endl;

	bg = clock();
#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
#endif
	ed = clock();
	cout << "Nearest neighbor matching time consuming: " << ed - bg << "ms" << endl;

	// GMS filter
	bg = clock();
	GMS gms;
	gms.init(img1.size(), img2.size(), kp1, kp2, matches_all);
	gms.setParameter(20, 20);
	matches_grid = gms.getInlier();
	ed = clock();
	cout << "GMS time consuming: " << ed - bg << "ms" << endl;
	cout << "Get total " << matches_grid.size() << " matches." << endl;

	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_grid);
	imshow("show", show);
	waitKey();
}


Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier) {
	Mat output(src1.rows, src1.cols + src2.cols, CV_8UC3);
	src1.copyTo(output.colRange(0, src1.cols));
	src2.copyTo(output.colRange(src1.cols, output.cols));

	for (size_t i = 0; i < inlier.size(); i++)
	{
		Point2f left = kpt1[inlier[i].queryIdx].pt;
		Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));

		line(output, left, right, Scalar(255, 0, 0));
	}

	for (size_t i = 0; i < inlier.size(); i++)
	{
		Point2f left = kpt1[inlier[i].queryIdx].pt;
		Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));

		circle(output, left, 1, Scalar(0, 255, 0), 2);
		circle(output, right, 1, Scalar(0, 255, 255), 2);
	}

	return output;
}
