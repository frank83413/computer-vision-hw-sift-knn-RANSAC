#include <cstdlib>
#include <algorithm>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <highgui.h>
#include "cv.h"
#include "hw2/function.h"
#include <fstream>
#include <cmath>
#include <ctime>

#include<string>
#include <opencv2/highgui/highgui.hpp> // For VS2015
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace std;
using namespace cv;
int main() {
	vector<Match> *match;
	int numberofpic = 0;
	cin >> numberofpic;
	match = new vector<Match>[numberofpic-1];
	vector<Match> finalmatch;
	// Start time
	time_t startTime = time(NULL);
	Mat H = Mat(3, 3, CV_32FC1);
	Mat targetim,targetimgray;
	// Image read
	Mat Image[10];
	Mat ImageGRAY[10];
	string picfile = "test/table/puzzle%d.bmp";
	char picfilebuffer[100];
	for (int i = 0; i < numberofpic - 1; i++)
	{
		sprintf(picfilebuffer, picfile.c_str(), i + 1);
		Image[i] = imread(picfilebuffer, IMREAD_COLOR);
		ImageGRAY[i] = imread(picfilebuffer, IMREAD_GRAYSCALE);
	}

	Mat feature;
	Image[numberofpic - 1] = imread("test/table/sample.bmp", IMREAD_COLOR);
	ImageGRAY[numberofpic - 1] = imread("test/table/sample.bmp", IMREAD_GRAYSCALE);
	targetim = imread("test/table/target.bmp", IMREAD_COLOR);
	targetimgray = imread("test/table/target.bmp", IMREAD_GRAYSCALE);

	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();


	// Feature detection
	vector<KeyPoint> keypoints[10];
	Mat descriptor[10];
	for (int i = 0; i < numberofpic; i++)
	{
		//detector.detect(Image[i], keypoints[i]);
		f2d->detect(ImageGRAY[i], keypoints[i]);
		cout << "Keypoints' number = " << keypoints[i].size() << endl;
		// Feature descriptor computation		
		//extractor.compute(Image[i], keypoints[i], descriptor[i]);
		f2d->compute(ImageGRAY[i], keypoints[i], descriptor[i]);
		cout << "Descriptor's size = " << descriptor[i].size() << endl;
		// Feature display on image		
		drawKeypoints(ImageGRAY[i], keypoints[i], feature, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		keypoints[i].resize(static_cast<int>(descriptor[i].rows));
		//imshow("I am feature point. > <", feature);
		//waitKey();
	}
	cout << descriptor[1].at<float>(100, 125) << endl;
	for (int i = 0; i < numberofpic - 1; i++) {
		match[i] = KNN(descriptor[i], descriptor[numberofpic - 1], 3);
	}
	for (int i = 0;i<numberofpic-1;i++) {
		H = ransac(match[i], 4, keypoints[i], keypoints[numberofpic - 1]);
		ImageGRAY[numberofpic - 1] = warping(ImageGRAY[i], ImageGRAY[numberofpic-1], H);
	}
	for (int i = 0; i<numberofpic - 1; i++) {
		H = ransac(match[i], 4, keypoints[i], keypoints[numberofpic - 1]);
		Image[numberofpic - 1] = warpingrgb(Image[i], Image[numberofpic - 1], H);
	}

	f2d->detect(ImageGRAY[numberofpic-1], keypoints[numberofpic-1]);
	f2d->compute(ImageGRAY[numberofpic-1], keypoints[numberofpic-1], descriptor[numberofpic-1]);

	f2d->detect(targetimgray, keypoints[numberofpic]);
	f2d->compute(targetimgray, keypoints[numberofpic], descriptor[numberofpic]);


	finalmatch = KNN(descriptor[numberofpic-1], descriptor[numberofpic], 3);
	H = ransac(finalmatch, 4, keypoints[numberofpic-1], keypoints[numberofpic]);
	targetim = warpingrgb(Image[numberofpic-1], targetim, H);
	//for (int i = 0; i<10; i++) {
	//	int k = 2;
	//	cout << match[k][i].dis<< ',' << match[k][i].pn << ',' << match[k][i].tn << endl;
	//}
	// End time
	time_t endTime = time(NULL);
	cout << "time: " << endTime - startTime << " s" << endl;
	
	// Show the result
	imshow("I am feature point. > <", targetim);
	imwrite("output2.jpg", targetim);
	waitKey();

	return 0;
}

