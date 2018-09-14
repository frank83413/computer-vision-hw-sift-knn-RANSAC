#include <cstdlib>
#include <algorithm>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <highgui.h>
#include "cv.h"
#include <fstream>
#include "hw2/function.h"

#define errorsort 1000
#define bound 300
using namespace std;
using namespace cv;

vector<Match> KNN(Mat desp, Mat dest, int K) {
	vector<Match> match;
	Match tempmatch;
	int matchcount=0;
	int count = 0;
	int despn = desp.rows;
	int destn = dest.rows;
	Mat tempdes=Mat(1,128,CV_32FC1);

	float *dis = new float[destn];
	int **disindex = new int*[K];
	for (int i = 0; i < K; i++)
	{
		disindex[i] = new int[2];
	}

	for (int i = 0; i < despn; i++) {
		//std::cout << "(i, j) = " << i << endl;
		for (int j = 0; j < 128; j++) {		
			tempdes.at<float>(0, j) = desp.at<float>(i, j);//break
		}
		//cout << desp.at<float>(i,1) << ','<<tempdes.at<float>(1, 1)<<endl;
		for (int j = 0; j < destn; j++) {
			dis[j] = 0;
			for (int k = 0; k < 128; k++) {
				
				dis[j] = dis[j] + abs(dest.at<float>(j, k) - tempdes.at<float>(0, k));
				if (dis[j] > bound)break;
			}
		}
		disindex = sort(dis, K, destn);
		
		for (int j = 0;j<K;j++) {
			
			if (disindex[j][0] != 1000) {
				//cout << disindex[j][0]<<','<<disindex[j][1] << endl;
				tempmatch.pn = i;
				tempmatch.tn = disindex[j][0];
				tempmatch.dis = disindex[j][1];
				match.push_back(tempmatch);
			}
		}
	}
	
	delete dis, disindex;
	return match;
}

int** sort(float *dis ,int K,int num) {
	int **tempdisindex = new int*[K];
	for (int i = 0;i<K;i++) {
		tempdisindex[i] = new int[2];
	}
	float tempx;
	bool get=false;
	for (int i = 0; i < K; i++) {
		tempx = bound;
		get = false;
		for (int j = 0; j < num; j++) {
			if (dis[j] < tempx) {
				tempx = dis[j];
				tempdisindex[i][0] = j;
				tempdisindex[i][1] = dis[j];
				get = true;
			}
		}		
		if (get)dis[tempdisindex[i][0]] = bound;
		else {
			tempdisindex[i][0] = errorsort;
		}
	}
	
	return tempdisindex;
}

Mat ransac(vector<Match> match, int ran, vector<KeyPoint> kpp, vector<KeyPoint> kpt) {
	float error=0;
	int len = match.size();
	int minvalueindex;
	Match *ranselect = new Match[ran];
	Mat select2xranx9 = Mat(ran * 2, 9, CV_32FC1);
	Mat select8x9 = Mat(8, 9, CV_32FC1);
	Mat eigenvalues = Mat(1, 9, CV_32FC1), eigenvectors = Mat(9, 9, CV_32FC1);
	Mat A = Mat(3, 3, CV_32FC1);
	Mat xy1x3new(3, 1,CV_32FC1);
	Mat xy1x3old(3, 1, CV_32FC1);
	float xp, yp, xt, yt;
	int *ranindex = new int[ran];
	int x, y ,x1,y1;
	bool ranfail = true;
	/*for (int i = 0; i < 4;i++) {
		cout << ranindex[i]<<endl;
	}*/
	//Mat eigenvalues3x3;
	int k = 0;
	while (ranfail) {
		error = 0;
		ranindex = randomarray(ran, kpp, match);
		for (int i = 0; i < ran; i++) {
			xp = kpp[match[ranindex[i]].pn].pt.x;
			yp = kpp[match[ranindex[i]].pn].pt.y;
			xt = kpt[match[ranindex[i]].tn].pt.x;
			yt = kpt[match[ranindex[i]].tn].pt.y;
			k = 2 * i;
			select2xranx9.at<float>(k, 0) = xp;
			select2xranx9.at<float>(k, 1) = yp;
			select2xranx9.at<float>(k, 2) = 1;
			select2xranx9.at<float>(k, 3) = 0;
			select2xranx9.at<float>(k, 4) = 0;
			select2xranx9.at<float>(k, 5) = 0;
			select2xranx9.at<float>(k, 6) = -xt*xp;
			select2xranx9.at<float>(k, 7) = -xt*yp;
			select2xranx9.at<float>(k, 8) = -xt;
			k = k + 1;
			select2xranx9.at<float>(k, 0) = 0;
			select2xranx9.at<float>(k, 1) = 0;
			select2xranx9.at<float>(k, 2) = 0;
			select2xranx9.at<float>(k, 3) = xp;
			select2xranx9.at<float>(k, 4) = yp;
			select2xranx9.at<float>(k, 5) = 1;
			select2xranx9.at<float>(k, 6) = -yt*xp;
			select2xranx9.at<float>(k, 7) = -yt*yp;
			select2xranx9.at<float>(k, 8) = -yt;
		}

		select8x9 = select2xranx9.t()*select2xranx9;
		cv::eigen(select8x9, eigenvalues, eigenvectors);
		minvalueindex = returnminvalue(eigenvalues);
		/*for (int i = 0;i<eigenvalues.rows;i++) {
			cout << eigenvalues.at<float>(i, 0) << endl;

		}
		cout << eigenvalues.at<float>(minvalueindex, 0) << endl;*/
		A.at<float>(0, 0) = eigenvectors.at<float>(minvalueindex, 0);
		A.at<float>(0, 1) = eigenvectors.at<float>(minvalueindex, 1);
		A.at<float>(0, 2) = eigenvectors.at<float>(minvalueindex, 2);
		A.at<float>(1, 0) = eigenvectors.at<float>(minvalueindex, 3);
		A.at<float>(1, 1) = eigenvectors.at<float>(minvalueindex, 4);
		A.at<float>(1, 2) = eigenvectors.at<float>(minvalueindex, 5);
		A.at<float>(2, 0) = eigenvectors.at<float>(minvalueindex, 6);
		A.at<float>(2, 1) = eigenvectors.at<float>(minvalueindex, 7);
		A.at<float>(2, 2) = eigenvectors.at<float>(minvalueindex, 8);
		/*A.at<float>(0, 0) = eigenvectors.at<float>(0, minvalueindex);
		A.at<float>(0, 1) = eigenvectors.at<float>(1, minvalueindex);
		A.at<float>(0, 2) = eigenvectors.at<float>(2, minvalueindex);
		A.at<float>(1, 0) = eigenvectors.at<float>(3, minvalueindex);
		A.at<float>(1, 1) = eigenvectors.at<float>(4, minvalueindex);
		A.at<float>(1, 2) = eigenvectors.at<float>(5, minvalueindex);
		A.at<float>(2, 0) = eigenvectors.at<float>(6, minvalueindex);
		A.at<float>(2, 1) = eigenvectors.at<float>(7, minvalueindex);
		A.at<float>(2, 2) = eigenvectors.at<float>(8, minvalueindex);*/

		for (int i = 0; i < ran; i++) {
			xy1x3old.at<float>(0, 0) = kpt[match[ranindex[i]].tn].pt.x;
			xy1x3old.at<float>(1, 0) = kpt[match[ranindex[i]].tn].pt.y;
			xy1x3old.at<float>(2, 0) = 1;
			xy1x3new = A.inv()*xy1x3old;
			xy1x3new.at<float>(0, 0) = xy1x3new.at<float>(0, 0) / xy1x3new.at<float>(2, 0);
			xy1x3new.at<float>(1, 0) = xy1x3new.at<float>(1, 0) / xy1x3new.at<float>(2, 0);
			x = xy1x3new.at<float>(0, 0);
			y = xy1x3new.at<float>(1, 0);
			x1 = kpp[match[ranindex[i]].pn].pt.x;
			y1 = kpp[match[ranindex[i]].pn].pt.y;
			//xy1x3new = A*xy1x3old;
			error=error+abs(x-x1)+abs(y-y1);
			
		}
		if (error < 2) {
			x = 1;
		}
		//cout << eigenvalues.at<float>(minvalueindex, 0) <<','<< minvalueindex <<endl;
		if (error <5)ranfail = false;
		//cout << error<<endl;
	}
	
	//cout << eigenvectors.at<float>(1,minvalueindex)<<endl;
	//cout << minvalueindex << endl;
	return A;
}

int returnminvalue(Mat eigenvalues) {
	float temp = eigenvalues.at<float>(0, 0);
	
	int minindex=0;
	for (int i=0 ; i<eigenvalues.rows; i++) {
		if (eigenvalues.at<float>(i, 0) < temp) {
			temp = eigenvalues.at<float>(i, 0);
			minindex = i;
		}
	}
	return minindex;
}

int* randomarray(int num, vector<KeyPoint> kp,vector<Match> match) {
	
	bool ranok = true;
	int *temp = new int[num];
	float **tempkp = new float*[num];
	for (int i = 0;i<num;i++) {
		tempkp[i] = new float[2];
	}
	int randnum;
	int count = 0;
	randnum = rand() % match.size();
	tempkp[count][0] = kp[match[randnum].pn].pt.x;
	tempkp[count][1] = kp[match[randnum].pn].pt.y;
	temp[count] = randnum;
	count++;
	while (count<num)
	{
		ranok = false;
		randnum = rand() % match.size();
		 //cout << kp.size();
		 for (int i = 0;i<count; i++) {
			 if (kp[match[randnum].pn].pt.x != tempkp[i][0] && kp[match[randnum].pn].pt.y != tempkp[i][1])ranok = true;
			 else {
				 ranok = false;
				 break;
			 }
		 }
		 if (ranok) {
			 tempkp[count][0] = kp[match[randnum].pn].pt.x;
			 tempkp[count][1] = kp[match[randnum].pn].pt.y;
			 temp[count] = randnum;
			 count++;
		 }
	}

	return temp;
}

Mat warping(Mat p,Mat t,Mat H) {
	Mat newpoint = Mat(3, 1, CV_32FC1);
	Mat oldpoint = Mat(3, 1, CV_32FC1);
	int count = 0;
	float x, y;
	for (int i = 0; i<t.rows; i++) {
		for (int j = 0; j < t.cols; j++) {

			oldpoint.at<float>(0, 0) = j;
			oldpoint.at<float>(1, 0) = i;
			oldpoint.at<float>(2, 0) = 1;
			newpoint = H.inv()*oldpoint;
			x = newpoint.at<float>(0, 0)/ newpoint.at<float>(2, 0);
			y = newpoint.at<float>(1, 0)/ newpoint.at<float>(2, 0);

			if (x > 0 && x < p.cols && y>0 && y < p.rows) {
				if (p.at<uchar>(y, x) != 0) {
					count++;
					t.at<uchar>(i, j) = p.at<uchar>(y,x);
				}
			}			
		}
	}
	//cout << count << endl;
	return t;
}

Mat warpingrgb(Mat p, Mat t, Mat H) {
	Mat newpoint = Mat(3, 1, CV_32FC1);
	Mat oldpoint = Mat(3, 1, CV_32FC1);
	int count = 0;
	float x, y;
	for (int i = 0; i<t.rows; i++) {
		for (int j = 0; j < t.cols; j++) {

			oldpoint.at<float>(0, 0) = j;
			oldpoint.at<float>(1, 0) = i;
			oldpoint.at<float>(2, 0) = 1;
			newpoint = H.inv()*oldpoint;
			x = newpoint.at<float>(0, 0) / newpoint.at<float>(2, 0);
			y = newpoint.at<float>(1, 0) / newpoint.at<float>(2, 0);

			if (x > 0 && x < p.cols && y>0 && y < p.rows) {
				if (p.at<Vec3b>(y, x)[0] != 0 && p.at<Vec3b>(y, x)[1] != 0 &&p.at<Vec3b>(y, x)[2] != 0) {
					count++;
					t.at<Vec3b>(i, j)[0] = p.at<Vec3b>(y, x)[0];
					t.at<Vec3b>(i, j)[1] = p.at<Vec3b>(y, x)[1];
					t.at<Vec3b>(i, j)[2] = p.at<Vec3b>(y, x)[2];
				}
			}
		}
	}
	//cout << count << endl;
	return t;
}