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
            self.covariances[alpha] = gmm.covariances_
            self.means = gmm.means_

    def calculate_GMM_model(self):
        for alpha in range(0, 2):
            for k in range(0, self.n_components):
                pixels = self.img[self.alpha == alpha and self.components == k]

    def calculate_D(self, x, y):
        pass
    '''
    Assign a foreground and background GMM cluster to every pixel in the unknown set 
    based off the minimum distance to the respective clusters
    '''
    def assign_GMM(self):
        pass

    '''
    Learn GMM parameters based off the pixel data with the newly assigned foreground and
    background clusters. This step will get rid of the old GMM and create a new GMM with the
    assigned foreground and background clusters from every pixel.
    '''
    def learn_GMM(self):
        pass

    '''
    A graph is constructed and Min Cut runs to estimate new foreground and background pixels
    '''
    def graphcut(self):
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

if __name__ == '__main__':
    grabcut = GrabCut('../test_imgs/HarryPotter5.jpg')
    rect = (10, 10, 450, 600)
    grabcut.run(rect, None)

