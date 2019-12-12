#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:24:02 2019

@author: bixi
"""

import glob
import numpy as np
import cv2

def training_data_pre(path1,path2):
    mask_pic = glob.glob(path1)
    ori_pic = glob.glob(path2)
    mask_pic.sort()
    ori_pic.sort()
    data_size = len(mask_pic)
    width,length,_ = cv2.imread(mask_pic[0]).shape
    imgs = np.ndarray((data_size,width,length),dtype = np.uint8)
    labels = np.ndarray((data_size,width,length),dtype = np.uint8)
    for i in range(data_size):
        img = cv2.imread(ori_pic[i],0)
        mask = cv2.imread(mask_pic[i],0)
        # Do normalization
        _,mask = cv2.threshold(mask,1,255, cv2.THRESH_BINARY)
        mask = mask // 255
        minpix = np.min(img)
        maxpix = np.max(img)
        img = (img-minpix)/(maxpix-minpix)
        # Do the resize
#        img = cv2.resize(img,(268,178))
#        mask = cv2.resize(img,(268,178))
        imgs[i] = img
        labels[i] = mask
    np.save('data/train_imgs.npy',imgs)
    np.save('data/train_labels.npy',labels)

def testing_data_pre(path1,path2):
    mask_pic = glob.glob(path1)
    ori_pic = glob.glob(path2)
    mask_pic.sort()
    ori_pic.sort()
    data_size = len(mask_pic)
    width,length,_ = cv2.imread(mask_pic[0]).shape
    imgs = np.ndarray((data_size,width,length),dtype = np.uint8)
    labels = np.ndarray((data_size,width,length),dtype = np.uint8)
    for i in range(data_size):
        img = cv2.imread(ori_pic[i],0)
        mask = cv2.imread(mask_pic[i],0)
        imgs[i] = img
        labels[i] = mask
    np.save('data/test_imgs.npy',imgs)
    np.save('data/test_labels.npy',labels)
    
if __name__ == '__main__':
    path1 = r'Train/masks_Haemorrhages/*.tif'
    path2 = r'Train/original_retinal_images/*.jpg'
    path3 = r'Test/masks_Haemorrhages/*.tif'
    path4 = r'Test/original_retinal_images/*.jpg'
    training_data_pre(path1,path2)
    testing_data_pre(path3,path4)
    