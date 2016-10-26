#pragma once
#include "stdafx.h"

#define LocalTheshold 0			// Threshold has two ways : global and local
#define LocalThreshFactor 2.71828	// factor for local threshold
#define GlobalTreshFactor 6			// factor for global threshold

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
	GMS(Size &sz1_p, Size &sz2_p, vector<KeyPoint> &kpt1_p, vector<KeyPoint> &kpt2_p,
		vector<DMatch> &matches_p)
	{
		init(sz1_p, sz2_p, kpt1_p, kpt2_p, matches_p);
	}

	// init variable
	void init(Size &sz1_p, Size &sz2_p, vector<KeyPoint> &kpt1_p, vector<KeyPoint> &kpt2_p,
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
	void getScoreAndThreshold(int idx_grid, double &score, double &thresh);
	
	// get inlier
	void run(int type);

	vector<DMatch> getInlier();
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

		switch (type){
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

void GMS::getScoreAndThreshold(int idx_grid, double &score, double &thresh) {
	score = 0.0;
	thresh = 0.0;

	// find neighbor
	vector<int> posRegion = getNeighbor9(idx_grid, num1);
	vector<int> motionRegion = getNeighbor9(IdxToRightRegion[idx_grid], num2);
	for (int j = 0; j < 9; j++)
	{
		int pos = posRegion[j];
		if (pos < 0 || pos >= number1 || kpts_num_of_each_region[pos] <= 0) continue;

		double positive_points = motion_number[pos][motionRegion[j]];
		score += positive_points;
		thresh += kpts_num_of_each_region[pos];
	}
	
	thresh = sqrt(thresh) * LocalThreshFactor;
}

void GMS::run(int type) {
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
	double globalthreshold = sqrt(matches.size() / numGrid) * GlobalTreshFactor;
	if (!LocalTheshold){ ACCEPT_SCORE = globalthreshold; }
	for (int i = 0; i < number1; i++) {
		if (kpts_num_of_each_region[i] <= 0)	continue;
		getScoreAndThreshold(i, scores[i], localthreshold);
		if (LocalTheshold){ ACCEPT_SCORE = localthreshold; }
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

vector<DMatch> GMS::getInlier() {
	for (int i = 1; i <= 4; i++)
	{
		run(i);
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

