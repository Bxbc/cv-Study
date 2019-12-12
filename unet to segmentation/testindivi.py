#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:36:45 2019

@author: bixi
"""

import glob
import cv2
import evaluate
import numpy as np

masks = glob.glob('mask/*.tif')
images = glob.glob('images/*.jpg')
masks.sort()                        # get the masks as sequence
images.sort()                       # get the images as sequence
img = cv2.imread(images[0])
im = img.flatten()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
recall = []
accura = []
jacca = []
percen = 0.15
cuts = int(2848*4288*(1-percen))
for i in range(len(images)):
    img = cv2.imread(images[i])
    mask = cv2.imread(masks[i],0)
    r = img[:,:,2]
    closed = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=50)
    closed = cv2.dilate(closed, None, iterations=50)
    flats = closed.flatten()
    flats.sort()                    # flatten the image
    thres = flats[cuts]         # set the area of optic disc region
    _,closed = cv2.threshold(closed,thres,255,cv2.THRESH_BINARY)  
    ac = evaluate.jaccard_ac(closed,mask)     
    jacca.append(ac)
    ac = evaluate.accuracy(closed,mask)
    accura.append(ac)
    ac = evaluate.recall(closed,mask)
    recall.append(ac)
    
np.save('recall.npy',recall)
np.save('jacc.npy',jacca)
np.save('acc.npy',accura)
