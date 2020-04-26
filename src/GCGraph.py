from enum import IntEnum

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
import cv2 as cv
import maxflow

class Trimap(IntEnum):
	BGD = 0
	FGD = 1
	UKN = 2


class GCGraph(object):
	def __init__(self, img, gamma=50):
		self.img = img
		self.gamma = gamma
		self.beta = 0
		self.h, self.w = img.shape[:2]
		self.N = self.w * self.h
		self.pixels = self.img.reshape(self.N, 3)
		self.bgd_src = self.N
		self.fgd_sink = self.N+1
		self.edge_nums = self.N*10 - 6*(self.w + self.h - 4) - 20
		self.largest_weight = 0
		self.dx = np.array([1, -1, 0, 1]).astype(np.uint8)
		self.dy = np.array([0, 1, 1, 1]).astype(np.uint8)
		self.graph = None
		self.calculate_beta()

	def to_1D_coord(self, x, y):
		return y*self.w + x

	def calculate_beta(self):
		w, h, img = self.w, self.h, self.img
		print(h, w, img.shape)
		dist, num = 0., 0
		for y1 in range(h):
			for x1 in range(w):
				for k in range(4):
					x2, y2 = x1 + self.dx[k], y1 + self.dy[k]
					if 0 <= x2 < w and 0 <= y2 < h:
						dist += euclidean(img[y1, x1], img[y2, x2])**2
						num += 1
		self.beta = 0.5/(dist/num)

	def init_N_links(self):
		w, h, img = self.w, self.h, self.img
		for y1 in range(h):
			for x1 in range(w):
				s = self.to_1D_coord(x1, y1)
				for k in range(4):
					x2, y2 = x1 + self.dx[k], y1 + self.dy[k]
					if 0 <= x2 < w and 0 <= y2 < h:
						t = self.to_1D_coord(x2, y2)
						weight = self.gamma / euclidean((x1, y1), (x2, y2)) * np.exp(-self.beta * (euclidean(img[y1, x1], img[y2, x2])**2))
						self.largest_weight = max(self.largest_weight, weight)
						self.graph.add_edge(s, t, weight, weight)
						# print("n-links: ", s, t, weight)

	def build_graph(self, mask, bgdModel, fgdModel):
		self.graph = maxflow.Graph[float](self.N, self.edge_nums)
		self.graph.add_nodes(self.N)
		self.init_N_links()
		# Add T-links
		for i in range(self.N):
			if mask[i] == Trimap.BGD:
				bgd_w, fgd_w = self.largest_weight, 0
			elif mask[i] == Trimap.FGD:
				bgd_w, fgd_w = 0, self.largest_weight
			else:
				bgd_w, fgd_w = bgdModel.model_likelihood(self.pixels[i]), fgdModel.model_likelihood(self.pixels[i])
			if bgd_w < fgd_w:
				print("t-links: ", i, bgd_w, fgd_w)
			self.graph.add_tedge(i, bgd_w, fgd_w)

	def cut(self):
		self.graph.maxflow()
		return np.array([self.graph.get_segment(i) for i in range(self.N)]).astype(np.uint8)


if __name__ == '__main__':
	img = cv.imread('../test_imgs/HarryPotter5.jpg')
	graph = GCGraph(img)

