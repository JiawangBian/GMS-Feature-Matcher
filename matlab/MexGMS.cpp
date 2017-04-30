#include "mex.h"
#include "gms_matcher.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	/* variable declarations here */
	if(nrhs != 5) throw runtime_error("unknown stereo match method: ");

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
	
	// Scale or Not: not(default)
	const double *data4 = (double *)mxGetPr(prhs[3]);
	int scale = *data4;
	
	// Rotation or Not: not(default)
	const double *data5 = (double *)mxGetPr(prhs[4]);
	int rotate = *data5;
	
	
	// orb
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms;
	Ptr<ORB> orb = ORB::create(num_keypoint);
	orb->setFastThreshold(0);
	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);

	// NN Match
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
	
	// GMS
	int num_inliers = 0;
	std::vector<bool> vbInliers;
	gms_matcher gms(kp1,img1.size(), kp2,img2.size(), matches_all);
	num_inliers = gms.GetInlierMask(vbInliers, scale, rotate);

	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}

	// write
	int matched_number = matches_gms.size();

	// write ratio
	plhs[0] = mxCreateNumericMatrix(2, matched_number, mxDOUBLE_CLASS, mxREAL);
	plhs[1] = mxCreateNumericMatrix(2, matched_number, mxDOUBLE_CLASS, mxREAL);
	double *p1 = (double *)mxGetPr(plhs[0]), *p2 = (double *)mxGetPr(plhs[1]);
	for (int i = 0; i < matched_number; i++)
	{
		const DMatch &match = matches_gms[i];

		p1[0] = kp1[match.queryIdx].pt.x;
		p1[1] = kp1[match.queryIdx].pt.y;
		p1 += 2;

		p2[0] = kp2[match.trainIdx].pt.x;
		p2[1] = kp2[match.trainIdx].pt.y;
		p2 += 2;
	}

}

