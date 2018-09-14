#include <cstdlib>
#include <algorithm>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <highgui.h>
#include "cv.h"
#include <fstream>
using namespace std;
using namespace cv;



class Match {
public:
	int pn;
	int tn;
	float dis;
};

Mat ransac(vector<Match>,int , vector<KeyPoint>, vector<KeyPoint>);
int returnminvalue(Mat);
vector<Match> KNN(Mat, Mat, int);
int** sort(float *dis, int K, int num);

int* randomarray(int , vector<KeyPoint>, vector<Match>);
Mat warping(Mat,Mat,Mat); 
Mat warpingrgb(Mat, Mat, Mat);