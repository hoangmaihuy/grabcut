#pragma once
#include <opencv2/core/core.hpp>
#include <vector>

using namespace cv;
using namespace std;

typedef unsigned char Index;
typedef vector<Index> VecIndex;

class GMM
{
public:
	static const int K = 5;
	double weight[K];
	double mean[K][3];
	double cov[K][3][3];
	double det_cov[K];
	double inv_cov[K][3][3];
	// helper arrays
	vector<int> C[K], idx;
	vector<Vec3d> pixs[K];
	GMM() { }
	VecIndex init_components(const vector<Vec3d>& pixels);
	Index get_component(const Vec3d &pixel);
	double component_likelihood(const Vec3d &pixel, int k);
	double model_likelihood(const Vec3d &pixel);
	void learn(const vector<Vec3d>& pixels, const VecIndex& components);
};



