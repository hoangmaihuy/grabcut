import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

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
    def __init__(self, rect, GMM_components):
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


