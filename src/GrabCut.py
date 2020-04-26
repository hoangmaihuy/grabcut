import os
import time
from enum import IntEnum

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.spatial.distance import euclidean
from src.GMM import GaussianMixtureModel
import cv2 as cv


class Trimap(IntEnum):
    BGD = 0
    FGD = 1
    UKN = 2


class Matte(IntEnum):
    BGD = 0
    FGD = 1


def timeit(func):
    def wrapper(*args, **kw):
        time1 = time.time()
        result = func(*args, **kw)
        time2 = time.time()
        print(func.__name__, time2-time1)
        return result
    return wrapper


class GrabCut(object):
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
        self.w = self.imgShape[0]   # image width
        self.h = self.imgShape[1]   # image height
        self.N = self.imgShape[0] * self.imgShape[1]
        self.pixels = self.img.reshape(self.N, 3)    # 1D image
        self.n_components = n_components
        self.iterCount = iterCount
        self.useCV = useCV
        self.mask = np.zeros(self.imgShape, np.uint8)
        self.bgdModel = None
        self.fgdModel = None
        # Matte segmentation value,
        self.alpha = None
        # GMM components index
        self.components = np.empty((self.N, ), np.uint8)
        self.trimap_bgd = None
        self.trimap_fgd = None
        self.trimap_ukn = None
        self.matte_bgd = None
        self.matte_fgd = None

    @timeit
    def init_with_rect(self, rect):
        self.mask[:, :] = Trimap.BGD
        x1, y1, x2, y2 = rect
        self.mask[y1:y2, x1:x2] = Trimap.UGD
        self.mask = self.mask.reshape((self.N, ))
        self.alpha = np.where(self.mask == Trimap.BGD, Matte.BGD, Matte.FGD)
        self.trimap_bgd = np.where(self.mask == Trimap.B)
        self.trimap_fgd = np.where(self.mask == Trimap.F)
        self.trimap_ukn = np.where(self.mask == Trimap.U)
        self.matte_bgd = np.where(self.alpha == Matte.BGD)
        self.matte_fgd = np.where(self.alpha == Matte.FGD)
        self.bgdModel = GaussianMixtureModel(self.n_components)
        self.fgdModel = GaussianMixtureModel(self.n_components)
        self.components[self.matte_bgd] = self.bgdModel.init_components(self.pixels[self.matte_bgd])
        self.components[self.matte_fgd] = self.fgdModel.init_components(self.pixels[self.matte_fgd])

    '''
    Assign a foreground and background GMM cluster to every pixel in the unknown set 
    based off the minimum distance to the respective clusters
    '''
    @timeit
    def assign_GMM(self):
        self.components[self.matte_bgd] = self.bgdModel.get_components(self.pixels[self.matte_bgd])
        self.components[self.matte_fgd] = self.fgdModel.get_components(self.pixels[self.matte_fgd])

    '''
    Learn GMM parameters based off the pixel data with the newly assigned foreground and
    background clusters. This step will get rid of the old GMM and create a new GMM with the
    assigned foreground and background clusters from every pixel.
    '''
    @timeit
    def learn_GMM(self):
        self.bgdModel.learn(self.pixels[self.matte_bgd], self.components[self.matte_bgd])
        self.fgdModel.learn(self.pixels[self.matte_fgd], self.components[self.matte_fgd])

    @timeit
    def build_graph(self):
        pass

    @timeit
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
                self.bgdModel = np.zeros((1, 65), np.float64)
                self.fgdModel = np.zeros((1, 65), np.float64)
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
    grabcut = GrabCut('../test_imgs/lena.jpg')
    rect = (10, 10, 250, 250)
    grabcut.run(rect, None)

