### Read ME

#### SIFT Algorithm (Scale Invariant Feature Transform)

**SIFT** is a well-known algorithm in computer vision to detect and describe local features in images. Its applications include object recognition, robotic mapping and navigation, image stitching, 3D modelling, video tracking and others.

A **SIFT** feature is a salient keypoint that corresponds to an image region and has an associated descriptor. SIFT computation is commonly divided into two steps:

* detection
* description

At the end of the detection step, and for each feature detected, the SIFT algorithm establishes:

* keypoint spatial coordinates (x, y)
* keypoint scale
* keypoint dominant orientation

After the detection step, the description step computes a distinctive fingerprint of 128 dimensions for each feature. The description obtained is designed to be invariant to scale and rotation. Moreover, the algorithm offers decent robustness to noise, illumination gradients and affine transformations.


**1.Computer SIFT Features**

* Extract SIFT features with default parameters and show the keypoints on the image.

**2.Rotate the image and Computer the SIFT features**

* Extract SIFT features and show the keypoints on the rotated image

**3.Rotational invariance of SIFT features**

* For each rotated image, match its **SIFT descriptors** with those from the original image based on the nearest neighbour distance ratio method.
* Draw the matches between the two images.
