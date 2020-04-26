from enum import IntEnum

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
import cv2 as cv

class Trimap(IntEnum):
	BGD = 0
	FGD = 1
	UKN = 2


class GCGraph(object):
	def __init__(self, img, gamma=50, neighbor_radius=1):
		self.img = img
		self.neighbor_radius = neighbor_radius
		self.gamma = gamma
		self.beta = 0
		self.w, self.h = img.shape[:2]
		self.N = self.w * self.h
		self.pixels = self.img.reshape(self.N, 3)
		self.bgd_src = self.N
		self.fgd_sink = self.N+1
		self.edge_nums = self.N*10 - 6*(self.w + self.h - 4) - 20
		self.from_node = np.zeros((self.edge_nums, )).astype(np.uint32)
		self.to_node = np.zeros((self.edge_nums, )).astype(np.uint32)
		self.weight = np.zeros((self.edge_nums, )).astype(np.float64)
		self.largest_weight = 0
		self.graph = None
		self.m = 0
		self.save_m = 0
		self.init_N_links()

	def to_1D_coord(self, x, y):
		return y*self.w + x

	def calculate_beta(self):
		r, w, h, img = self.neighbor_radius, self.w, self.h, self.img
		dist = np.array([])
		for y1 in range(h):
			for x1 in range(w):
				for y2 in range(y1-r, y1+r+1):
					for x2 in range(x1-r, x1+r+1):
						if 0 <= x2 < w and 0 <= y2 < h:
							if x1 == x2 and y1 == y2:
								continue
							np.append(dist, euclidean(img[y1, x1], img[y1, x2])**2)
		self.beta = 1/(2*np.mean(dist))

	def add_edge(self, s, t, w):
		self.from_node[self.m] = s
		self.to_node[self.m] = t
		self.weight[self.m] = w
		self.m += 1

	def init_N_links(self):
		r, w, h, img = self.neighbor_radius, self.w, self.h, self.img
		for y1 in range(h):
			for x1 in range(w):
				s = self.to_1D_coord(x1, y1)
				for y2 in range(y1-r, y1+r+1):
					for x2 in range(x1-r, x1+r+1):
						if 0 <= x2 < w and 0 <= y2 < h:
							if x1 == x2 and y1 == y2:
								continue
							t = self.to_1D_coord(x2, y2)
							weight = self.gamma / euclidean((x1, y1), (x2, y2)) * np.exp(-self.beta * euclidean(img[y1, x1], img[y2, x2]))
							self.largest_weight = max(self.largest_weight, weight)
							self.add_edge(s, t, weight)
		self.save_m = self.m

	def build_graph(self, mask, bgdModel, fgdModel):
		self.m = self.save_m
		for i in range(self.N):
			if mask[i] == Trimap.BGD:
				bgd_w, fgd_w = self.largest_weight, 0
			elif mask[i] == Trimap.FGD:
				bgd_w, fgd_w = 0, self.largest_weight
			else:
				bgd_w, fgd_w = bgdModel.model_likelihood(self.pixels[i]), fgdModel.model_likelihood(self.pixels[i])
			self.add_edge(self.bgd_src, i, bgd_w)
			self.add_edge(i, self.fgd_sink, fgd_w)

	def min_cut(self):
		self.graph = csr_matrix((self.weight, (self.from_node, self.to_node)), shape=(self.N+2, self.N+2))
		max_flow = maximum_flow(self.graph, self.bgd_src, self.fgd_sink)
		return max_flow

if __name__ == '__main__':
	img = cv.imread('../test_imgs/lena_small.jpg')
	graph = GCGraph(img)

