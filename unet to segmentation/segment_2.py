#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:26:58 2019

@author: bixi
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed

img = cv2.imread("test2.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ret,thres = cv2.threshold(gray,170,255,cv2.THRESH_BINARY)
#kernel = np.ones((3,3),np.uint8)
#opening = cv2.morphologyEx(thres,cv2.MORPH_OPEN,kernel, iterations = 2)
#sure_bg = cv2.dilate(opening,kernel,iterations=3)
#dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
#ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#
#sure_fg = np.uint8(sure_fg)
#unknown = cv2.subtract(sure_bg,sure_fg)
#
#ret, markers = cv2.connectedComponents(sure_fg)
#markers = markers+1
#markers[unknown==255] = 0
#markers = cv2.watershed(img,markers)
#img[markers == 0] = [255,0,0]
distance = ndi.distance_transform_edt(gray)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)))
markers = ndi.label(local_maxi)[0]
ws_labels = watershed(-distance, markers, mask=gray)

plt.imshow(ws_labels)