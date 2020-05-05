#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	vector<Vec3d> pixels;
	Mat cov;
	Vec3d mean;
	pixels.push_back(Vec3d(1, 2, 3));
	pixels.push_back(Vec3d(3, 4, 5));
}