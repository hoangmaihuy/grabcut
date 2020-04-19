import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import cv2 as cv

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
    def __init__(self, imagePath, rect, n_components=5, iterCount=5, useCV=True):
        self.imagePath = imagePath
        self.rect = rect
        self.n_components = n_components
        self.iterCount = iterCount
        self.useCV = useCV

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

    def run(self):
        if self.useCV:
            img = cv.imread(self.imagePath)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            mask = np.zeros(img.shape[:2], np.uint8)
            cv.grabCut(img, mask, self.rect, bgdModel, fgdModel, self.iterCount, cv.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
            img = img*mask2[:, :, np.newaxis]
            print(img)
            dirname, filename = os.path.split(self.imagePath)
            filename, fileext = os.path.splitext(filename)
            savePath = os.path.join(dirname, filename) + "_result" + fileext
            cv.imwrite(savePath, img)
            return savePath


