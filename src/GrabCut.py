import os
from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import cv2 as cv

class Trimap(IntEnum):
    U = 0
    B = 1
    F = 2

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
    def __init__(self, imagePath, n_components=5, iterCount=5, useCV=True):
        self.imagePath = imagePath
        self.n_components = n_components
        self.iterCount = iterCount
        self.useCV = useCV
        self.mask = None
        self.img = None
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)

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
            if self.img is None:
                self.img = cv.imread(self.imagePath)
                self.mask = np.zeros(self.img.shape[:2], np.uint8)
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


