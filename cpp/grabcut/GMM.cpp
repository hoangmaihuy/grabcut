#include "GMM.h"

using namespace cv;
using namespace std;

#define For(i, a, b) for (int i = a; i <= b; i++)
#define REP(i, n) for (int i = 0; i < n; i++)

const double SINGULAR_FIX = 0.01;
const double EPS = 1e-6;

void calcMeanCov(const vector<Vec3d>& pixels, double (&mean)[3], double (&cov)[3][3])
{
	memset(mean, 0, sizeof(mean));
	memset(cov, 0, sizeof(cov));
	double diff[3];
	long long n = pixels.size();
	for (auto pixel : pixels) REP(i, 3) mean[i] += pixel[i];
	REP(i, 3) mean[i] /= n;
	if (n <= 1) return;
	for (auto pixel : pixels)
	{
		REP(i, 3) diff[i] = pixel[i] - mean[i];
		REP(i, 3) REP(j, 3) cov[i][j] += diff[i] * diff[j];
	}
	REP(i, 3) REP(j, 3) cov[i][j] /= (n - 1);
}

void calcDetInv(const double (&cov)[3][3], double& det, double(&inv)[3][3])
{
	double a[3][3];
	REP(i, 3) REP(j, 3) a[i][j] = cov[i][j];
	det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
		- a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
		+ a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
	while (det < EPS)
	{
		REP(i, 3) a[i][i] += SINGULAR_FIX;
		det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
			- a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
			+ a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
	}
	double inv_det = 1.0 / det;
	inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det;
	inv[1][0] = -(a[1][0] * a[2][2] - a[1][2] * a[2][0]) * inv_det;
	inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det;
	inv[0][1] = -(a[0][1] * a[2][2] - a[0][2] * a[2][1]) * inv_det;
	inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
	inv[2][1] = -(a[0][0] * a[2][1] - a[0][1] * a[2][0]) * inv_det;
	inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
	inv[1][2] = -(a[0][0] * a[1][2]- a[0][2] * a[1][0]) * inv_det;
	inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;
}

void calcEigen(double (&cov)[3][3], double &eival, double (&eivec)[3])
{
	Mat M(3, 3, CV_64F, cov);
	Mat E, V;
	eigen(M, E, V);
	eival = E.at<double>(0, 0);
	REP(i, 3) eivec[i] = V.at<double>(0, i);
}

VecIndex GMM::init_components(const vector<Vec3d>& pixels)
{
	double eivals[K];
	double eivecs[K][3];
	// All pixels in C[0]
	VecIndex components = VecIndex(pixels.size(), 0);
	REP(i, pixels.size()) C[0].push_back(i);
	calcMeanCov(pixels, mean[0], cov[0]);
	calcEigen(cov[0], eivals[0], eivecs[0]);
	for (int i = 1; i < K; i++)
	{
		int k = 0;
		for (int j = 1; j < i; j++)
			if (eivals[j] > eivals[k])
				k = j;
		idx = C[k];
		double sep = 0, val;
		pixs[i].clear(); pixs[k].clear(); C[k].clear(); C[i].clear();
		REP(t, 3) sep += eivecs[k][t] * mean[k][t];
		for (int j : idx)
		{
			val = 0;
			REP(t, 3) val += eivecs[k][t] * pixels[j][t];
			int t = (val <= sep ? i : k);
			C[t].push_back(j);
			pixs[t].push_back(pixels[j]);
			components[j] = t;
		}
		calcMeanCov(pixs[i], mean[i], cov[i]);
		calcEigen(cov[i], eivals[i], eivecs[i]);
		calcMeanCov(pixs[k], mean[k], cov[k]);
		calcEigen(cov[k], eivals[k], eivecs[k]);
	}
	REP(i, K)
	{
		weight[i] = (double)C[i].size() / pixels.size();
		calcDetInv(cov[i], det_cov[i], inv_cov[i]);
	}
	return components;
}

Index GMM::get_component(const Vec3d& pixel)
{
	Index k = 0;
	double val = component_likelihood(pixel, 0);
	for (Index i = 1; i < K; i++)
	{
		double tmp = component_likelihood(pixel, i);
		if (tmp > val)
		{
			val = tmp;
			k = i;
		}
	}
	return k;
}

double GMM::component_likelihood(const Vec3d& pixel, int k)
{
	if (weight[k] < EPS) return 0;
	double diff[3];
	REP(i, 3) diff[i] = pixel[i] - mean[k][i];
	double mult = 0, tmp;
	REP(i, 3)
	{
		tmp = 0;
		REP(j, 3) tmp += diff[j] * inv_cov[k][j][i];
		mult += tmp * diff[i];
	}
	return exp(-0.5 * mult) / det_cov[k];
}

double GMM::model_likelihood(const Vec3d& pixel)
{
	double s = 0;
	REP(i, K) s += component_likelihood(pixel, i);
	if (s < EPS) return 0;
	return -log(s);
}

void GMM::learn(const vector<Vec3d>& pixels, const VecIndex& components)
{
	REP(i, K)
	{
		C[i].clear(); pixs[i].clear();
	}
	int n = pixels.size();
	REP(i, n)
	{
		int k = components[i];
		C[k].push_back(i);
		pixs[k].push_back(pixels[i]);
	}
	REP(i, n)
	{
		if (pixs[i].empty()) continue;
		calcMeanCov(pixs[i], mean[i], cov[i]);
		calcDetInv(cov[i], det_cov[i], inv_cov[i]);
		weight[i] = (double)pixs[i].size() / n;
	}
}