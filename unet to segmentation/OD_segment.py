#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:44:17 2019

@author: bixi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift
from PIL import Image



img = cv2.imread('1.jpg')
#blur = cv2.pyrMeanShiftFiltering(img, 10, 30, termcrit=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 15, 15))
#cv2.imwrite('indiv_result/3.jpg',blur)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
r_channel = img[:,:,2]
g_channel = img[:,:,1]
b_channel = img[:,:,0]

closed = cv2.morphologyEx(r_channel, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, None, iterations=50)
closed = cv2.dilate(closed, None, iterations=50)
_,pre = cv2.threshold(closed,252,255,cv2.THRESH_BINARY)
#means, stddev = cv2.meanStdDev(closed)
cv2.imwrite('report1.jpg',pre)
#shapes = closed.shape
#ms_clf = MeanShift(bin_seeding=True)
#colour_samples = np.column_stack((closed.flatten(),closed.flatten(),closed.flatten()))
#ms_labels = ms_clf.fit_predict(colour_samples).reshape(shapes)


#thres = (245*means)//135
#plt.hist(r_channel.ravel(),bins=51,density=1,alpha=0.7) # Get the histogram of red channel
#_,result = cv2.threshold(closed,thres,255, cv2.THRESH_BINARY)
#plt.imshow(result)
