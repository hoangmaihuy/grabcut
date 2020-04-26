from sklearn.mixture import GaussianMixture
import numpy as np

EPS = 1e-6
SINGULAR_FIX = 0.01


class GaussianMixtureModel(object):
	def __init__(self, K):
		self.K = K
		self.weight = np.empty((K,))
		self.mean = np.empty((K, 3))
		self.cov = np.empty((K, 3, 3))
		self.det_cov = np.empty((K,)).astype(np.float64)
		self.inv_cov = np.empty_like(self.cov)

	# Use k-means to cluster components
	def init_components(self, pixels):
		gmm = GaussianMixture(self.K)
		components = gmm.fit_predict(pixels)
		self.weight = gmm.weights_
		self.mean = gmm.means_
		self.cov = gmm.covariances_
		for k in range(self.K):
			self.det_cov[k] = np.linalg.det(self.cov[k])
			self.inv_cov[k] = np.linalg.inv(self.cov[k])
		return components

	# Define how a pixel fit into a component k of this model
	def component_likelihood(self, pixel, k):
		x = pixel - self.mean[k]
		return 1/np.sqrt(self.det_cov[k]) * np.exp(-0.5 * (x.T @ self.inv_cov[k] @ x))

	# Define how a pixel fit into this model by summing component_likelihood
	def model_likelihood(self, pixel):
		return -np.log(sum([self.component_likelihood(pixel, k) * self.weight[k] for k in range(self.K)]))

	# Assign to the most likelihood component
	def get_component(self, pixel):
		return np.argmax([self.component_likelihood(pixel, k) for k in range(self.K)])

	def get_components(self, pixels):
		return np.array([self.get_component(pixel) for pixel in pixels])

	def learn(self, pixels, components):
		for k in range(self.K):
			sub_pixels = pixels[components == k]
			# print(k, len(sub_pixels))
			if len(sub_pixels) == 0:
				self.weight[k] = 0.
				continue
			self.mean[k] = np.mean(sub_pixels, axis=0)
			self.cov[k] = np.cov(sub_pixels.T, bias=True)
			self.det_cov[k] = np.linalg.det(self.cov[k])
			while self.det_cov[k] < EPS:
				self.cov[k] += np.diag([SINGULAR_FIX for i in range(3)])
				self.det_cov[k] = np.linalg.det(self.cov[k])
			self.inv_cov[k] = np.linalg.inv(self.cov[k])
			self.weight[k] = len(sub_pixels) / len(pixels)
