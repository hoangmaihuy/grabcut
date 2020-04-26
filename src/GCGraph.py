import numpy as np
from scipy.spatial.distance import euclidean


class GCGraph(object):
	def __init__(self, img, gamma=50, neighbor_radius=1):
		self.img = img
		self.neighbor_radius = neighbor_radius
		self.gamma = gamma
		self.beta = 0
		self.w, self.h = img.shape[:2]
		self.N = self.w * self.h
		self.z = self.img.reshape(self.N, 3)
		self.source = self.N
		self.sink = self.N+1
		self.N_links = np.array([[], [], []]).astype(np.uint32)
		self.T_links = np.array([[], [], []]).astype(np.uint32)
		self.max_neighbor_weight = 0
		self.mean_distance = 0
		self.graph = None

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
							np.append(dist, euclidean(img[y1, x1], img[y1, x2])**2)
		self.beta = 1/(2*np.mean(dist))

	def init_N_links(self):
		r, w, h, img = self.neighbor_radius, self.w, self.h, self.img
		for y1 in range(h):
			for x1 in range(w):
				s = self.to_1D_coord(x1, y1)
				for y2 in range(y1-r, y1+r+1):
					for x2 in range(x1-r, x1+r+1):
						if 0 <= x2 < w and 0 <= y2 < h:
							t = self.to_1D_coord(x2, y2)
							np.append(self.N_links[0], s)
							np.append(self.N_links[1], t)
							c = self.gamma / euclidean((x1, y1), (x2, y2)) * np.exp(-self.beta * euclidean(img[y1, x1], img[y2, x2]))
							np.append(self.N_links[2], c)

