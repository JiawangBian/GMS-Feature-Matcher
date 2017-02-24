#pragma once
#include "Header.h"
#include "GMS.h"

#define BUFFLE_LENGTH 10

struct ThreadParameter
{
	Ptr<ORB> orb;
	VideoCapture cap;
	volatile int cpu_index;
	volatile int gpu_index;
	int end;
	int height;
	Mat img0, desc0;
	vector<KeyPoint> kpt0;
	vector<Mat> imgs;
	vector<Mat> descriptors;
	vector<vector<KeyPoint>> kpts;

	ThreadParameter(Ptr<ORB> orb, VideoCapture cap, int height, Mat img0, vector<KeyPoint>&kpt0, Mat desc0, Mat img1, vector<KeyPoint>&kpt1, Mat desc1) {
		this->orb = orb;
		this->cap = cap;
		this->cpu_index = 1;
		this->gpu_index = 0;
		this->height = height;
		this->end = 0;
		this->img0 = img0;
		this->desc0 = desc0;
		this->kpt0 = kpt0;
		imgs.assign(BUFFLE_LENGTH,Mat());
		descriptors.resize(BUFFLE_LENGTH,Mat());
		kpts.assign(BUFFLE_LENGTH, vector<KeyPoint>());
		this->imgs[0] = img1;
		this->descriptors[0] = desc1;
		this->kpts[0] = kpt1;
	}
};

DWORD WINAPI extract_featuers(LPVOID pM)
{
	ThreadParameter* param = (ThreadParameter*)pM;
	Ptr<ORB> orb = param->orb;
	VideoCapture cap = param->cap;
	while (1) {
		if (param->cpu_index % 10 == param->gpu_index % 10) {
			Sleep(5);
			continue;
		}
		int index = param->cpu_index % 10;
		if (!cap.grab()){ break; }
		cap.retrieve(param->imgs[index]);
		imresize(param->imgs[index], param->height);
		orb->detectAndCompute(param->imgs[index], Mat(), param->kpts[index], param->descriptors[index]);
		param->cpu_index += 1;
	}
	param->end = 1;

	return 0;
}

DWORD WINAPI nn_match(LPVOID pM)
{
	ThreadParameter* param = (ThreadParameter*)pM;
	vector<DMatch> matches, mathces_grid;

#ifdef USE_GPU
	cuda::GpuMat gd1(param->desc0);
	cuda::GpuMat gd2;
	Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
#else
	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);
#endif

	while (param->end != 1) {
		if (param->gpu_index >= param->cpu_index) {
			Sleep(5);
			continue;
		}
		int index = param->gpu_index % 10;

#ifdef USE_GPU
		gd2.upload(param->descriptors[index]);
		matcher->match(gd1, gd2, matches);
#else
		Mat d1 = param->desc0;
		Mat d2 = param->descriptors[index];
		matcher->match(d1, d2, matches);
#endif

		// your code here
		vector<DMatch> matches_grid;
		GMS gms;
		gms.init(param->img0.size(), param->imgs[index].size(), param->kpt0, param->kpts[index], matches);
		gms.setParameter();
		matches_grid = gms.getInlier();
		Mat show = DrawInlier(param->img0, param->imgs[index], param->kpt0, param->kpts[index], matches_grid, 2);
		imshow("show", show);
		waitKey(1);

		// update
		param->gpu_index += 1;
	}

	return 0;
}

void testVideo(string filename){
	VideoCapture cap(filename);
	if (!cap.isOpened())	return;

	Ptr<ORB> orb = ORB::create(10000);
	orb->setFastThreshold(0);

	Mat img0, img1;
	vector<KeyPoint> kp0, kp1;
	Mat desc0, desc1;
	const int height = 480;
	const int total_frames = (int)cap.get(CAP_PROP_FRAME_COUNT);

	// img0 is query image
	cap >> img0;
	imresize(img0, height);
	orb->detectAndCompute(img0, Mat(), kp0, desc0);

	cap >> img1;  
	imresize(img1, height);
	orb->detectAndCompute(img1, Mat(), kp1, desc1);

	ThreadParameter param(orb, cap, height, img0, kp0, desc0, img1, kp1, desc1);

	clock_t bg = clock();

	HANDLE cpu_handle = CreateThread(NULL, 0, extract_featuers, (LPVOID)&param, 0, NULL);
	HANDLE gpu_handle = CreateThread(NULL, 0, nn_match, (LPVOID)&param, 0, NULL);

	WaitForSingleObject(cpu_handle, INFINITE);
	WaitForSingleObject(gpu_handle, INFINITE);

	clock_t ed = clock();
	
	cout << (ed - bg) * 1.0 / total_frames << "ms" << endl;
}
