#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:17:10 2019

@author: bixi
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

def draw_line():
    loss = np.load('3/test_loss.npy')
    loss2 = np.load('3/test_loss2.npy')
    size = len(loss)
    X = np.arange(1,size+1)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_title('Influence of Segmentation')
    plt.xlabel('epoch')
    plt.ylabel('testing loss')
    my_y_ticks = np.arange(0,1,0.05)
    plt.yticks(my_y_ticks)
    plt.plot(X, loss, color="r", linestyle="--", marker="*", linewidth=1.0,label='segment')
    plt.plot(X,loss2,color="b", linestyle="-", marker="+", linewidth=1.0,label='resize')
    plt.legend(loc='best')
    plt.savefig("compare_test.png",dpi=600)
    plt.show()

def draw_3d():
    img = cv2.imread('testcombine/0.tif',0)
    img = cv2.resize(img,(100,100))
    height,width = img.shape
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0,width)
    Y = np.arange(0,height)
    X, Y = np.meshgrid(X, Y)
    Z = img
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()
    
jac = np.load('jacc.npy')
acc = np.load('acc.npy')
rec = np.load('recall.npy')
size = len(jac)
X = np.arange(1,size+1)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.set_title('Evaluation of Segmentation')
plt.xlabel('image')
plt.ylabel('evaluate')
my_y_ticks = np.arange(0,1,0.05)
plt.yticks(my_y_ticks)
plt.plot(X,jac, color="r", linestyle="--", marker="*", linewidth=1.0,label='jaccard')
plt.plot(X,acc,color="b", linestyle="-", marker="+", linewidth=1.0,label='accuracy')
plt.plot(X,rec,color="b", linestyle=":", marker="^", linewidth=1.0,label='recall')
plt.legend(loc='best')
plt.savefig("15%.png",dpi=600)
plt.show()