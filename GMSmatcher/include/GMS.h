#pragma once
#include "Header.h"

#define LocalTheshold 1			// Threshold has two ways : global and local
#define TreshFactor 6			// factor for calculating threshold

int RotationPatterns[8][9] = {
	1,2,3,4,5,6,7,8,9,
	4,1,2,7,5,3,8,9,6,
	7,4,1,8,5,2,9,6,3,
	8,7,4,9,5,1,6,3,2,
	9,8,7,6,5,4,3,2,1,
	6,9,8,3,5,7,2,1,4,
	3,6,9,2,5,8,1,4,7,
	2,3,6,1,5,9,4,7,8
};

double ScaleRatio[5] = { 1.0 / 2, 1.0 / sqrt(2.0), 1.0, sqrt(2.0), 2 };

class GMS
{
private:
	// variable
	vector<DMatch> inlier, matches;
	vector<KeyPoint> kpt1, kpt2;
	vector<int> IdxToRightRegion, kpts_num_of_each_region, IsInlier;
	vector<map<int, int> > motion_number;
	Size sz1, sz2;		// size for left and right image
	

	// parameter
	double ACCEPT_SCORE;
	int number1, number2;

public:
	GMS() {};
	~GMS() {};

	// init variable
	void init(Size sz1_p, Size sz2_p, vector<KeyPoint> &kpt1_p, vector<KeyPoint> &kpt2_p,
		vector<DMatch> &matches_p)
	{
		sz1 = sz1_p;	sz2 = sz2_p;
		kpt1 = kpt1_p;	kpt2 = kpt2_p;
		matches = matches_p;
		IsInlier.assign(matches.size(), 0);
	}

	// set paramter
	void setParameter(int numR1 = 20, int numR2 = 20) {
		num1 = numR1;	num2 = numR2;
		number1 = num1 * num1;			number2 = num2 * num2;
		stepX1 = sz1.width / (num1 - 1);		stepY1 = sz1.height / (num1 - 1);
		stepX2 = sz2.width / (num2 - 1);		stepY2 = sz2.height / (num2 - 1);
		IdxToRightRegion.assign(number1, -1);
	}

	// get socre of a grid
	void getScoreAndThreshold(int idx_grid, int rp, double &score, double &thresh);

	// get inlier
	void run(int type, int rp);

	vector<DMatch> getInlier(int withrotation = 0);
	size_t getInlierSize() { return inlier.size(); }

private:
	int num1, num2, stepX1, stepX2, stepY1, stepY2;
	int getLeftRegion(DMatch &match, int type) {
		Point2i pt = kpt1[match.queryIdx].pt;
		switch (type)
		{
		case 2:	pt.x += stepX1 / 2; break;
		case 3: pt.y += stepY1 / 2; break;
		case 4:
			pt.x += stepX1 / 2;
			pt.y += stepY1 / 2;
			break;
		default:
			break;
		}
		if (pt.x >= sz1.width || pt.y >= sz1.height)
		{
			return -1;
		}
		else
		{
			return  (pt.x / stepX1 + pt.y / stepY1 * num1);
		}
	}
	int getRightRegion(DMatch &match, int type) {
		Point2i pt = kpt2[match.trainIdx].pt;

		switch (type) {
		case 2:	pt.x += stepX2 / 2; break;
		case 3: pt.y += stepY2 / 2; break;
		case 4:
			pt.x += stepX2 / 2;
			pt.y += stepY2 / 2;
			break;
		default:
			break;
		}
		if (pt.x >= sz2.width || pt.y >= sz2.height)
		{
			return -1;
		}
		else
		{
			return  (pt.x / stepX2 + pt.y / stepY2 * num2);
		}
	}

	vector<int> getNeighbor9(int idx, int num) {
		vector<int> neighbor(9);
		for (int yi = 0; yi < 3; yi++)
		{
			for (int xi = 0; xi < 3; xi++)
			{
				neighbor[xi + yi * 3] = idx + (yi - 1) * num + (xi - 1);
			}
		}
		return neighbor;
	}
};

void GMS::getScoreAndThreshold(int idx_grid, int rp, double &score, double &thresh) {
	score = 0.0;
	thresh = 0.0;

	// find neighbor
	vector<int> posRegion = getNeighbor9(idx_grid, num1);
	vector<int> motionRegion = getNeighbor9(IdxToRightRegion[idx_grid], num2);
	int num = 0;
	for (int j = 0; j < 9; j++)
	{
		int pos = posRegion[j];
		if (pos < 0 || pos >= number1 || kpts_num_of_each_region[pos] <= 0) continue;

		// motion predeict with rotation
		int k = RotationPatterns[rp][j] - 1;
		double positive_points = motion_number[pos][motionRegion[k]];
		score += positive_points;
		thresh += kpts_num_of_each_region[pos];
		num++;
	}

	thresh = sqrt(thresh / num) * TreshFactor;
}

void GMS::run(int type, int rp) {
	//initialize
	motion_number.clear();
	motion_number.resize(number1);
	kpts_num_of_each_region.resize(number1, 0);
	IdxToRightRegion.resize(number1, -1);
	vector<double> scores(number1, 0);

	// motion statistic 
	for (int i = 0; i < matches.size(); i++)
	{
		//assign motion
		int left_idx = getLeftRegion(matches[i], type), right_idx = getRightRegion(matches[i], type);
		if (left_idx < 0 || right_idx < 0)continue;

		kpts_num_of_each_region[left_idx] += 1;
		motion_number[left_idx][right_idx] += 1;
	}
	//	Mark IdxToRightRegion
	int numGrid = 0;
	for (int i = 0; i < number1; i++)
	{
		if (kpts_num_of_each_region[i] <= 0) continue;
		pair<int, int> max_region(0, 0);
		for (auto &p : motion_number[i]) { if (p.second > max_region.second) max_region = p; }
		IdxToRightRegion[i] = max_region.first;
		numGrid++;
	}

	double localthreshold = 0;
	double globalthreshold = sqrt(matches.size() / numGrid) * TreshFactor;
	if (!LocalTheshold) { ACCEPT_SCORE = globalthreshold; }
	for (int i = 0; i < number1; i++) {
		if (kpts_num_of_each_region[i] <= 0)	continue;
		getScoreAndThreshold(i, rp, scores[i], localthreshold);
		if (LocalTheshold) { ACCEPT_SCORE = localthreshold; }
		if (scores[i] < ACCEPT_SCORE) { IdxToRightRegion[i] = -2; }
	}

	// label inlier
	for (int i = 0; i < matches.size(); i++)
	{
		int left_idx = getLeftRegion(matches[i], type);
		int right_idx = getRightRegion(matches[i], type);
		if (left_idx < 0 || right_idx < 0)	continue;
		if (IdxToRightRegion[left_idx] == right_idx) {
			IsInlier[i] = 1;
		}
	}
}

vector<DMatch> GMS::getInlier(int withRS) {
	
	if (withRS == 0) {
		for (int i = 1; i <= 4; i++)
		{
			run(i, 0);
		}
	}
	else
	{
		int maxInlier = 0;
		vector<int> SavedInliers;

		int curInlier = 0;
		IsInlier.clear();

		for (int s = 0; s < 5; s++)
		{
			int left = 20; int right = static_cast<int>(left * ScaleRatio[s]);
			setParameter(left, right);

			for (int j = 0; j < 8; j++)
			{
				for (int i = 1; i <= 4; i++)
				{
					run(i, j);
				}

				// count
				for (int i = 0; i < matches.size(); i++)
				{
					if (IsInlier[i] == 1) {
						curInlier++;
					}
				}
				if (curInlier > maxInlier)
				{
					maxInlier = curInlier;
					SavedInliers.clear();
					SavedInliers = IsInlier;
				}
			}
		}

		IsInlier.clear();
		IsInlier = SavedInliers;
	}

	inlier.reserve(matches.size());
	for (int i = 0; i < matches.size(); i++)
	{
		if (IsInlier[i] == 1) {
			inlier.push_back(matches[i]);
		}
	}

	return inlier;
}


// utility
inline Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
	const int height = max(src1.rows, src2.rows);
	const int width = src1.cols + src2.cols;
	Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
	src1.copyTo(output(Rect(0,0,src1.cols,src1.rows)));
	src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

	if (type == 1)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(0, 255, 255));
		}
	}
	else if (type == 2)
 	{
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
			circle(output, left, 1, Scalar(0, 255, 255), 2);
			circle(output, right, 1, Scalar(0, 255, 0), 2);
		}
	}



	return output;
}

inline void imresize(Mat &src, int height){
	double ratio = src.rows * 1.0 / height;
	int width = static_cast<int>(src.cols * 1.0 / ratio);
	resize(src, src, Size(width, height));
}
