import os
from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn.mixture import GaussianMixture
import cv2 as cv

class Trimap(IntEnum):
    B = 0
    F = 1
    U = 2

class GrabCut:
    '''
    - The user initializes the trimap by only giving the background pixels. 
    The foreground set of pixels is set to the empty set. 
    The unknown pixels are set to the compliment of the background pixels.
    - Hard segmentation is performed to create a foreground and background pixel sets. 
    The background pixels go into the background pixel set. 
    The unknown pixels is the foreground set.
    - Create foreground and background GMMs based off the sets previously defined.
    '''
    def __init__(self, imagePath, n_components=5, iterCount=5, useCV=False):
        self.imagePath = imagePath
        self.img = cv.imread(imagePath)
        self.imgShape = self.img.shape[:2]
        self.N = self.imgShape[0] * self.imgShape[1]
        self.img = self.img.reshape(self.N, 3)
        self.n_components = n_components
        self.iterCount = iterCount
        self.useCV = useCV
        self.mask = np.zeros(self.imgShape, np.uint8)
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)
        self.weights = np.empty((2, n_components))
        self.means = np.empty((2, n_components, 3))
        self.covariances = np.empty((2, n_components, 3, 3))
        self.det_cov = np.empty((2, n_components)).astype(np.float64)
        self.inv_cov = np.empty_like(self.covariances)
        self.alpha = None
        self.components = np.empty((self.N, ), np.uint8)

    def init_with_rect(self, rect):
        self.mask[:, :] = Trimap.B
        x1, y1, x2, y2 = rect
        self.mask[y1:y2, x1:x2] = Trimap.F
        self.mask = self.mask.reshape((self.N, ))
        self.alpha = np.where(self.mask == Trimap.B, Trimap.B, Trimap.F)
        for alpha in range(2):
            pixels = self.img[self.alpha == alpha]
            gmm = GaussianMixture(self.n_components)
            self.components[self.alpha == alpha] = gmm.fit_predict(pixels)
            self.weights[alpha] = gmm.weights_
            self.means[alpha] = gmm.means_
            self.covariances[alpha] = cov = gmm.covariances_
            for k in range(self.n_components):
                self.det_cov[alpha, k] = np.linalg.det(cov[k])
                self.inv_cov[alpha, k] = np.linalg.inv(cov[k])

    def calculate_D(self, alpha, k, z):
        return (np.log(self.weights[alpha, k]) + 1/2*np.log(self.det_cov[alpha, k])
                + 1/2*np.transpose(z - self.means[alpha, k]) @ self.inv_cov[alpha, k] @ (z - self.means[alpha, k]))
    '''
    Assign a foreground and background GMM cluster to every pixel in the unknown set 
    based off the minimum distance to the respective clusters
    '''
    def assign_GMM(self):
        iter = [self.calculate_D(self.alpha[i], k, self.img[i]) for i in range(self.N) for k in range(self.n_components)]
        D = (np.fromiter(iter, dtype=np.float64)).reshape(self.N, self.n_components)
        self.components = np.apply_along_axis(np.argmin, 1, D)

    '''
    Learn GMM parameters based off the pixel data with the newly assigned foreground and
    background clusters. This step will get rid of the old GMM and create a new GMM with the
    assigned foreground and background clusters from every pixel.
    '''
    def learn_GMM(self):
        for alpha in range(2):
            total = np.count_nonzero(self.alpha == alpha)
            for k in range(0, self.n_components):
                indexes = np.where(np.logical_and(self.alpha == alpha, self.components == k))
                pixels = self.img[indexes]
                self.means[alpha, k] = np.mean(pixels)
                self.covariances[alpha, k] = cov = np.cov(pixels.T)
                self.inv_cov[alpha, k] = np.linalg.inv(cov)
                self.det_cov[alpha, k] = np.linalg.det(cov)
                self.weights[alpha, k] = len(pixels) / total

    '''
    A graph is constructed and Min Cut runs to estimate new foreground and background pixels
    '''
    def min_cut(self):
        pass

    def run(self, rect, init_mask):
        if self.useCV:
            mode = 0
            if init_mask is not None:
                mode = cv.GC_INIT_WITH_MASK
                self.mask[init_mask == Trimap.B] = cv.GC_BGD
                self.mask[init_mask == Trimap.F] = cv.GC_FGD
            if rect is not None:
                mode = cv.GC_INIT_WITH_RECT
            cv.grabCut(self.img, self.mask, rect, self.bgdModel, self.fgdModel, self.iterCount, mode)
            mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
            img = self.img*mask2[:, :, np.newaxis]
            dirname, filename = os.path.split(self.imagePath)
            filename, fileext = os.path.splitext(filename)
            resultPath = os.path.join(dirname, filename) + "_result" + fileext
            cv.imwrite(resultPath, img)
            return resultPath
        else:
            if rect is not None:
                self.init_with_rect(rect)
                for _ in range(self.iterCount):
                    self.assign_GMM()
                    self.learn_GMM()
                    self.min_cut()


if __name__ == '__main__':
    grabcut = GrabCut('../test_imgs/HarryPotter5.jpg')
    rect = (10, 10, 450, 600)
    grabcut.run(rect, None)

