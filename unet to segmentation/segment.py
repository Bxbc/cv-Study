#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:22:09 2019

@author: bixi
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

img = cv2.imread("test2.jpg",0)
#row,col,channel = img.shape
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gradx = cv2.Sobel(gray,ddepth=cv2.CV_16S,dx=1,dy=0)
#grady = cv2.Sobel(gray,ddepth=cv2.CV_16S,dx=0,dy=1)
#absx = cv2.convertScaleAbs(gradx)
#absy = cv2.convertScaleAbs(grady)
#gradient = cv2.addWeighted(absx,10,absy,10,2)
#gradient = cv2.subtract(absx, absy)
#gradient = cv2.convertScaleAbs(gradient)
laplacian = np.array([[0, -1, 0],[-1, 8, -1], [0,-1,0]])
#kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
lplcn = cv2.filter2D(img,-1, laplacian)
#dst = cv2.filter2D(img,-1,lplcn)
#k = ndimage.convolve(gray,kernel)
#c = cv2.Canny(gray,20,100)
#gray = gray+c
_,result = cv2.threshold(lplcn, 167, 255, cv2.THRESH_BINARY)
plt.imshow(result)
