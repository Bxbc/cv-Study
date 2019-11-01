# Computer Version
# XI BI

import cv2
import numpy as np

# the name is the location of image that will be processed
def contrast_stretching(name):
    image = cv2.imread(name)
    max_pix = float('-inf')
    min_pix = float('inf')
    for i in image:
        for j in i:
            for m in j:
                if max_pix < m:
                    max_pix = m
                if min_pix > m:
                    min_pix = m
    for i in range(len(image)):
        for j in range(len(image[0])):
            for m in range(len(image[0][0])):
                image[i][j][m] = (image[i][j][m]-min_pix)*(255/(max_pix-min_pix))
    cv2.imwrite('contrast_streching.jpg',image)
    return 0


def lapls_img(name):
    image = cv2.imread(name)
    dst = cv2.Laplacian(image,-1,3)
    cv2.imwrite('Q4_laplacian.jpg',dst)
    return 0
#lapls_img('contrast_streching.jpg')

image = cv2.imread('ansel_adams.jpg')
