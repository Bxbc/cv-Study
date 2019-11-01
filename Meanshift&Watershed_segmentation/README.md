### Read Me

#### The Algorithms

**MeanShift:**
the “mode-seeking” algorithm is a clustering algorithm that assigns pixels to clusters by iteratively shifting points towards the mode, where the mode is the value with the highest number of datapoints.

**Watershed:**
is a transformation which aims to segment the regions of interest in a grayscale image. This method is particularly useful when two regions of interest are close to each other — i.e, their edges touch. This technique treats the image as a topographic map, with the intensity of each pixel representing the height.

#### Do the Segmentation
The sample images “strawberry.png” and “shapes.png” are used here.

**1.Use the MeanShift algorithm to segment the two images**

* Step 1: Once you read the images into numpy arrays, extract each colour channel (R, G, B) so you can use each as a variable for classification. To do this you will need to convert the colour matrices into a flattened vector as depicted in the image below.
* Step 2: You can then use the new flattened colour sample matrix (10000 x 3 if your original image was 100x100) as your variable for classification.
* Step 3: Use the MeanShift’s fit_predict function to perform a clustering and save the cluster labels (which we want to observe).

**2.Use a Watershed transformation to segment the grayscale versions of the two images.**

* Step 1: Convert the image to grayscale.
* Step 2: Calculate the distance transform of the image.
* Step 3: Generate the watershed markers as the ‘clusters’ furthest away from the background.
* Step 4. Perform watershed on the image. This is the part where the image is “flooded” and the water sinks to the “catchment basins” based on the markers found in step 3.

**References:**

* [Agarwal, R. (2015). Segmentation using Watershed Algorithm in Matlab [Video].](https://www.youtube.com/watch?v=K5P5rjDiZzk)
* [Meanshift Algorithm for the Rest Of Us](http://www.chioka.in/meanshift-algorithm-for-the-rest-of-us-python/)
