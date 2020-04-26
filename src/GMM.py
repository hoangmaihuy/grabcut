from sklearn.mixture import GaussianMixture
import numpy as np


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
		for cov in self.cov:
			self.det_cov = np.linalg.det(cov)
			self.inv_cov = np.linalg.inv(cov)
		return components

	# Define how a pixel fit into a component k of this model
	def component_likelihood(self, pixel, k):
		return (np.log(self.weight[k]) + 1/2 * np.log(self.det_cov[k])
		        + 1/2 * (np.transpose(pixel - self.mean[k]) @ self.inv_cov[k] @ (pixel - self.mean[k])))

	# Define how a pixel fit into this model by summing component_likelihood
	def model_likelihood(self, pixel):
		s = 0.0
		for k in range(self.K):
			s += (self.weight[k] / np.sqrt(self.det_cov[k])
			      * np.exp(1 / 2 * (np.transpose(pixel - self.mean[k]) @ self.inv_cov[k] @ (pixel - self.mean[k]))))
		return -np.log(s)

	# Assign to the most likelihood component
	def get_component(self, pixel):
		likelihoods = np.fromiter(map(lambda k: self.component_likelihood(pixel, k), range(self.K)), dtype=np.float64)
		return np.argmin(likelihoods)

	def get_components(self, pixels):
		return np.fromiter(map(lambda pixel: self.get_component(pixel), pixels), dtype=np.uint8)

	def learn(self, pixels, components):
		for k in range(self.K):
			sub_pixels = pixels[components == k]
			self.mean[k] = np.mean(sub_pixels)
			self.cov[k] = cov = np.cov(sub_pixels.T)
			self.det_cov[k] = np.linalg.det(cov)
			self.inv_cov[k] = np.linalg.inv(cov)
			self.weight[k] = len(sub_pixels) / len(pixels)