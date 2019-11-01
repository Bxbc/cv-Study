# Test the SIFT algorithm

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import sys

class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector=self.get_detector(params)
        self.norm=norm

    def get_detector(self, params):
        if params is None:
            params={}
            params["n_features"]= 49
            params["n_octave_layers"]=3
            params["contrast_threshold"]=0.04
            params["edge_threshold"]=10
            params["sigma"]=1.6

        detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=params["n_features"],
                nOctaveLayers=params["n_octave_layers"],
                contrastThreshold=params["contrast_threshold"],
                edgeThreshold=params["edge_threshold"],
                sigma=params["sigma"])

        return detector

# Rotate an image
#
# image: image to rotate
# x:     x-coordinate of point we wish to rotate around
# y:     y-coordinate of point we wish to rotate around
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image
def rotate(image, x, y, angle):
    angle = -angle
    rot_center = (x,y)
    rot_mat = cv2.getRotationMatrix2D(rot_center,angle,1)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],flags=cv2.INTER_LINEAR)
    return result


# Get coordinates of center point.
#
# image:  Image that will be rotated
# return: (x, y) coordinates of point at center of image
def get_img_center(image):
    image_center = tuple(np.array(image.shape)//2)
    return image_center[0],image_center[1]


if __name__ == '__main__':
    # Read image with OpenCV and convert to grayscale
    '''
    img = cv2.imread('lab2_sample_images/Eiffel_Tower.jpg',cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('lab2_sample_images/road_sign.jpg',cv2.IMREAD_COLOR)
    # Initialize SIFT detector
    sift = SiftDetector()
    sift_detectors = sift.get_detector(None)
    
    # Store SIFT keypoints of original image in a Numpy array
    kp = sift_detectors.detect(img,None)
    
    img_key = cv2.drawKeypoints(img,kp,None)

    # Rotate around point at center of image.
    center_x,center_y = get_img_center(img2)
    
    img_rotate = rotate(img2,center_x,center_y,45)
    cv2.imwrite('task_1.jpg',img_key)
    cv2.imwrite('rotate_45.jpg',img_rotate)
    # Degrees with which to rotate image
    
    '''
    img_ori = cv2.imread('road_sign.jpg',cv2.IMREAD_COLOR)
    img_rot = cv2.imread('rotate_90.jpg',cv2.IMREAD_COLOR)
    sift_ori = SiftDetector()
    sift_rot = SiftDetector()
    ori_detect = sift_ori.get_detector(None)
    rot_detect = sift_rot.get_detector(None)
    kp1,des1 = ori_detect.detectAndCompute(img_ori,None)
    kp2,des2 = rot_detect.detectAndCompute(img_rot,None)
    # Number of times we wish to rotate the image
    '''
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    # BFMatcher with default params
    

        # Rotate image
        

        # Compute SIFT features for rotated image
        
        # Apply ratio test
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = cv2.DrawMatchesFlags_DEFAULT)
    img_res = cv2.drawMatchesKnn(img_ori,kp1,img_rot,kp2,matches,None,**draw_params)
    plt.imshow(img_res,),plt.show()
    cv2.imwrite('90_degree_matches.jpg',img_res)


        # cv2.drawMatchesKnn expects list of lists as matches.
    '''
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    img_res = cv2.drawMatchesKnn(img_ori,kp1,img_rot,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_res),plt.show()
    cv2.imwrite('90_degree_matches.jpg',img_res)
    
    
    