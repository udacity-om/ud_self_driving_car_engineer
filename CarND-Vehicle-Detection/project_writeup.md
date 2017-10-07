**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier SVM classifier
* Apply a color transform and append binned color features to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./result/car_non_car.jpg
[image2]: ./result/visualize_color_space.jpg
[image3]: ./result/car_non_car_ycrcb.jpg
[image4]: ./result/visualize_hog_on_diff_color_spaces.jpg
[image5]: ./result/visualize_hog_on_ycrcb_diff_orient.jpg
[image6]: ./result/car_non_car_ycrcb_hog.jpg
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

###Histogram of Oriented Gradients (HOG)

####1. Extracting HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and their channels. Below is the image showing the best channels of each color space:

![alt text][image2]

After going through the forum, I decided to use YCrCb color space. Below is the image showing the YCrCb color space(channel Y to visualize better) for car and non-car images:

![alt text][image3]

I then explored `skimage.hog()` with parameters: `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` on different color spaces and below is the result:

![alt text][image4]

I also explored `skimage.hog()` on color space YCrCb(Y channel) with parameters `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and different `orientations`:

![alt text][image5]

####2. Final choice of HOG parameters.

I tried various combinations of color spaces and parameters as shown in the above section. `YCrCb` looks better and is also recommended in forums. `orientations=10` does a good job so I decided to use 10 instead of higher orientation as it would increase the features with no additional information.

Here is an example using the `YCrCb` color space(Y channel) and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for car and non-car image:

![alt text][image6]

####3. Training a classifier using selected HOG and color features.

I used grid search to select the best SVM classifier. The parameters used are `{'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':[1, 5, 10, 15]}`. The grid search resulted in `{'C': 5, 'kernel': 'rbf'}` as the best parameters with mean score 0.99465. This best classifier gave a test score of 0.99634. I then saved the required parameters in `classifier_params.pkl` file.

###Sliding Window Search

I used the function `find_cars()` provided by Udacity which uses number of cells to move instaed of overlap value. The `find_cars()` function finds cars in the portion of the image bounded in y by `ystart` and `ystop`. The cars appear bigger when closer to the camera and smaller when away from camera, so different scales are needed for different portions of the image. I created `get_detections()` function which calls find_cars() with different `(ystart, ystop, scale)` values. The image is divided into four parts and 'find_cars()' is called four times but with different scale values. 

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

