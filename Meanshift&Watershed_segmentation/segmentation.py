#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:43:41 2019

@author: bixi

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift

from PIL import Image

size = 100, 100

img_names = ["pics/strawberry.png", "pics/shapes.png"]
#ext_names = ["pics/coins.png", "pics/two_halves.png"]

images = [i for i in img_names]
#ext_images = [i for i in ext_names]


def plot_three_images(figure_title, image1, label1,
                      image2, label2, image3, label3):
    fig = plt.figure()
    fig.suptitle(figure_title)

    # Display the first image
    fig.add_subplot(1, 3, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)

    # Display the second image
    fig.add_subplot(1, 3, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)

    # Display the third image
    fig.add_subplot(1, 3, 3)
    plt.imshow(image3)
    plt.axis('off')
    plt.title(label3)

    plt.show()


for img_path in images:
    img = Image.open(img_path)
    img.thumbnail(size)  # Convert the image to 100 x 100
    # Convert the image to a numpy matrix
    img_mat = np.array(img)[:, :, :3]
                    # record the shape of image
          
    r_channel = img_mat[:,:,0]
    g_channel = img_mat[:,:,1]
    b_channel = img_mat[:,:,2]
    shapes = b_channel.shape
    colour_samples = np.column_stack((r_channel.flatten(),g_channel.flatten(),b_channel.flatten()))
    ms_clf = MeanShift(bin_seeding=True)
    ms_labels = ms_clf.fit_predict(colour_samples).reshape(shapes)
    
    

    # img_gray = (r_channel+g_channel+b_channel)//3
    img_gray = img.convert('L')
    distance = ndi.distance_transform_edt(img_gray)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)))
    markers = ndi.label(local_maxi)[0]
    ws_labels = watershed(-distance, markers, mask=img_gray)
    plot_three_images(img_path, img, "Original Image", ms_labels, "MeanShift Labels",
                      ws_labels, "Watershed Labels")
    
    
    
    
    
    
    