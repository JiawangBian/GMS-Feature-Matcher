#include "mex.h"
#include "GMS.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	/* variable declarations here */
	if(nrhs != 4) throw runtime_error("unknown stereo match method: ");

	// Image 1
	int Img1h = mxGetM(prhs[0]), Img1w = mxGetN(prhs[0]);
	Mat img1(Img1w, Img1h, CV_8UC1);
	const uchar *data1 = (uchar *)mxGetPr(prhs[0]);
	memcpy(img1.data, data1, sizeof(uchar) * Img1h * Img1w);
	img1 = img1.t();

	// Image2
	int Img2h = mxGetM(prhs[1]), Img2w = mxGetN(prhs[1]);
	Mat img2(Img2w, Img2h, CV_8UC1);
	const uchar *data2 = (uchar *)mxGetPr(prhs[1]);
	memcpy(img2.data, data2, sizeof(uchar) * Img2h * Img2w);
	img2 = img2.t();

	// Number of Points: 10000 (default)
	const double *data3 = (double *)mxGetPr(prhs[2]);
	int num_keypoint = *data3;
	
	// Rotation or Not: not(default)
	const double *data4 = (double *)mxGetPr(prhs[3]);
	int rotate = *data4;
	
	// orb
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_grid;
	Ptr<ORB> orb = ORB::create(num_keypoint);
	orb->setFastThreshold(0);
	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);

	// NN Match
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
	
	// GMS
	GMS FM;
	FM.init(img1.size(), img2.size(), kp1, kp2, matches_all);
	FM.setParameter(20, 20);
	matches_grid = FM.getInlier(rotate);

	// write
	int matched_number = matches_grid.size();

	// write ratio
	plhs[0] = mxCreateNumericMatrix(2, matched_number, mxDOUBLE_CLASS, mxREAL);
	plhs[1] = mxCreateNumericMatrix(2, matched_number, mxDOUBLE_CLASS, mxREAL);
	double *p1 = (double *)mxGetPr(plhs[0]), *p2 = (double *)mxGetPr(plhs[1]);
	for (int i = 0; i < matched_number; i++)
	{
		const DMatch &match = matches_grid[i];

		p1[0] = kp1[match.queryIdx].pt.x;
		p1[1] = kp1[match.queryIdx].pt.y;
		p1 += 2;

		p2[0] = kp2[match.trainIdx].pt.x;
		p2[1] = kp2[match.trainIdx].pt.y;
		p2 += 2;
	}

}

