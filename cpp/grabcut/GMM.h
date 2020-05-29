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
	const static int K = 5;
	double weight[K];
	double mean[K][3];
	double cov[K][3][3];
	double det_cov[K];
	double inv_cov[K][3][3];
	// helper arrays
	vector<int> C[K], idx;
	vector<Vec3b> pixs[K];
	GMM() { }
	void init_components(const vector<Vec3b>& pixels, VecIndex &components);
	Index get_component(const Vec3b &pixel);
	double component_likelihood(const Vec3b &pixel, int k);
	double model_likelihood(const Vec3b &pixel);
	void learn(const vector<Vec3b>& pixels, const VecIndex& components);
};



