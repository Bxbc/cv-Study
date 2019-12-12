# Jaccard Similarity Coefficient

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

#from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import accuracy_score
#jac = jaccard_similarity_score(img,pre,normalize=True,sample_weight=None)


# the img is the ground truth 
# the pre is the predicted images
# they should have the same size
# and both them are the single channel
def jaccard_ac(pre,img):
    num_img = cv2.countNonZero(img)
    num_pre = cv2.countNonZero(pre)
    cop = img.copy()
    cop[pre==0] = 0
    joint = cv2.countNonZero(cop)
    jac = joint/(num_img+num_pre-joint)
    return jac

# true Positive / (True Positive + False Negative)
def recall(pre,img):
    num_pre = cv2.countNonZero(img)
    cop = img.copy()
    cop[pre==0]=0
    tp = cv2.countNonZero(cop)
    return tp/num_pre

def accuracy(pre,img):
    _,img = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
    _,pre = cv2.threshold(pre,1,255,cv2.THRESH_BINARY)
    return accuracy_score(img,pre,normalize=True,sample_weight = None)


if __name__ == '__main__':
    pre = cv2.imread('5.jpg',0)
    img = cv2.imread('masks_Hard_Exudates/IDRiD_55_EX.tif',0)
    print(recall(pre,img))
    print(jaccard_ac(pre,img))
    print(accuracy(pre,img))