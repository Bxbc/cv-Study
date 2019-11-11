# COMP9517 Project
# Individual Component
# XI BI
# z5198280

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift

from PIL import Image



#length,width = img.shape
#for i in range(length):
#    for j in range(width):
#        if img[i][j]>170 and img[i][j]<200:
#            img[i][j] = 255
#        else:
#            img[i][j] = 0
#            
#plt.imshow(img)
#cv2.imwrite("test.tif",img)
#size = 100,100
#img = Image.open("1.jpg")
#img_gray = img.convert('L')
#distance = ndi.distance_transform_edt(img_gray)
#local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((1000, 1000)))
#markers = ndi.label(local_maxi)[0]
#ws_labels = watershed(-distance, markers, mask=img_gray)
#plt.imshow(ws_labels)

#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(100,100))
#noise_kernel = np.ones((100,100),np.uint8)
#img = cv2.imread("1.jpg")
#img = cv2.morphologyEx(img, cv2.MORPH_OPEN, noise_kernel)
#plt.imshow(img)
#img = cv2.dilate(img,kernel)
#plt.imshow(img)
#cv2.imwrite("nosie.jpg",img)

img = cv2.imread("1.jpg")
noise_kernel = np.ones((100,100),np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, noise_kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(100,100))
img = cv2.dilate(img,kernel)
plt.imshow(img)

#remove_noise = cv2.medianBlur(img,121)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(100,100))
#new_img = cv2.dilate(remove_noise,kernel)
#gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
#_,result = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
#plt.imshow(result)
