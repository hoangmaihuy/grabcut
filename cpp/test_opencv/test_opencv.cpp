// test_opencv.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <vector>
#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

Mat convertToMat(vector<Vec3d> pixels)
{
	Mat res = Mat(pixels.size(), 3 , CV_64FC1);
	for (int i = 0; i < pixels.size(); i++)
		for (int j = 0; j < 3; j++)
			res.at<double>(i, j) = pixels[i][j];
	return res;
}

int main()
{
	vector<Vec3d> pixels;
	pixels.push_back(Vec3d(1, 2, 3));
	pixels.push_back(Vec3d(4, 5, 6));
	pixels.push_back(Vec3d(4, 5, 6));
	pixels.push_back(Vec3d(4, 5, 6));
	Mat res = convertToMat(pixels);
	cout << res << " " << res.channels() << "\n";
	Mat cov = Mat(3, 3, CV_64FC1);
	Mat mean = Mat(1, 3, CV_64FC1);
	calcCovarMatrix(pixels, cov, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	cout << mean << " " << cov << "\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
