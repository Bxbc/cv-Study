### Read me

#### Contrast Stretching

Contrast in an image is a measure of the range of intensity values within an image, and is the difference between the maximum and minimum pixel values. The full contrast of an 8-bit image is 255(max)-0(min)=255, and anything less than that results in a lower contrast image. Contrast stretching attempts to improve the contrast of an image by stretching (linear scaling) the range of intensity values. Assume that *Or* is the original image and *Tr* is the transformed image. Let a and b be the min and max pixel values allowed in an image (8-bit image, a=0 and b=255), and let c and d be the min and max pixel values in a given image, then the contrast stretched image is given by the function:

<center>Tr=(Or-c)*((b-a)/(d-c))+a</center>

#### Image Shapening with Laplacian

Image sharpening is a technique to try enhance the visible texture in the image. A common way to perform image sharpening is by calculating the second derivative of an image, the Laplacian. The following kernel calculates a discrete approximation to the Laplacian:

<center>L = [[0,-1,0],[-1,4,-1],[0,-1,0]]