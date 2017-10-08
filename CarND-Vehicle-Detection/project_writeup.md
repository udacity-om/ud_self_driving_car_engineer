**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier SVM classifier
* Apply a color transform and append binned color features to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_non_car.jpg
[image2]: ./output_images/visualize_color_space.jpg
[image3]: ./output_images/car_non_car_ycrcb.jpg
[image4]: ./output_images/visualize_hog_on_diff_color_spaces.jpg
[image5]: ./output_images/visualize_hog_on_ycrcb_diff_orient.jpg
[image6]: ./output_images/car_non_car_ycrcb_hog.jpg
[image7]: ./output_images/sliding_fixed_window.jpg
[image8]: ./output_images/first_portion.jpg
[image9]: ./output_images/second_portion.jpg
[image10]: ./output_images/third_portion.jpg
[image11]: ./output_images/fourth_portion.jpg
[image12]: ./output_images/portions_combined.jpg
[image13]: ./output_images/test_image_portion.jpg
[image14]: ./output_images/test_image_portion_ycrcb.jpg
[image15]: ./output_images/test_image_portion_ycrcb_hog.jpg
[image16]: ./output_images/test_image_portion_sliding_window.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

###Histogram of Oriented Gradients (HOG)

####1. Extracting HOG features from the training images.

The code for this step is contained in the file classifier.ipynb

I started by reading in all the `vehicle` and `non-vehicle` images(cell 2, classifier.ipynb).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and their channels(cell 15, classifier.ipynb). Below is the image showing the best channels of each color space:

![alt text][image2]

I decided to use YCrCb color space Y channel(cell 16, classifier.ipynb). Below is the image showing the YCrCb color space, channel Y for car and non-car images:

![alt text][image3]

I then explored `skimage.hog()`(using the helper function `get_hog_features()`) with parameters: `orientations=8`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` on different color spaces and below is the result(cell 17, classifier.ipynb):

![alt text][image4]

I also explored `skimage.hog()` on color space YCrCb(Y channel) with parameters `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` but with different `orientations`(cell 18, classifier.ipynb):

![alt text][image5]

####2. Final choice of HOG parameters.

I tried various combinations of color spaces and parameters as shown in the above section. `YCrCb Channel Y` looked better and is also recommended in forums. `orientations=8` does a good job so I decided to use 8 instead of higher orientation as it would increase the features adding to more computational time.

Here is an example using the `YCrCb` color space(Y channel) and HOG parameters of `orientations=8`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` for car and non-car image(cell 19, classifier.ipynb):

![alt text][image6]

####3. Training a classifier using selected HOG and color features.

The extracted features were normalized using `StandardScaler()` from `sklearn` package and then split into training and testing datasets(cell 11, classifier.ipynb). 

I used grid search to select the best SVM classifier. The parameters used are `{'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':[1, 5, 10, 15]}`. The grid search resulted in `{'C': 5, 'kernel': 'rbf'}` as the best parameters. This best classifier gave a test score of 0.9943. I then saved the required parameters in `classifier_params.pkl` file(cell 12,13 and 14, classifier.ipynb).

###Sliding Window Search

I used the function `find_cars()`(cell 3, vehicle_detection.ipynb) provided by Udacity to do the sliding window search with little modification. The `find_cars()` function searches for a car using window which moves accross an image portion. The `find_cars()` function finds cars in the portion of the image bounded in `y` by `ystart` and `ystop` and in `x` by `xstart` and `xstop`. Below is an example of sliding window seacrh on lower half of the image with fixed window size:

![alt text][image7]

####2. Optimize performance of the classifier. Test images to demonstrate how the pipeline works.

The classifier's job was to search for car in every window of the image using the sliding window search. The classifier was trained on a fixed image size of 64x64 but the car can appear anywhere in the image. The cars appear bigger when closer to the camera and smaller when away from camera. So to help the classifier, different scales were used in different portions of the image. I created `get_detections()`(cell 4, vehicle_detection.ipynb) function which calls find_cars() with different `(ystart, ystop, xstart, xstop, scale)` values.

First portion:  `(ystart, ystop, xstart, xstop, scale)` = (400, 500, 600, 1100, 1)

![alt text][image8]

Second portion:  `(ystart, ystop, xstart, xstop, scale)` = (400, 570, 700, 1280, 1.2)

![alt text][image9]

Third portion:  `(ystart, ystop, xstart, xstop, scale)` = (400, 620, 800, 1280, 1.5)

![alt text][image10]

Fourt portion:  `(ystart, ystop, xstart, xstop, scale)` = (400, 690, 900, 1280, 1.6)

![alt text][image11]

All portions combined:

![alt text][image12]

Ultimately I searched on the four scales using YCrCb Y-channel HOG features plus spatially binned color in the feature vector, which provided a nice result. Here are some example images:

Selecting only a portion of the image:

![alt text][image13]

Converting the portion to `YCrCb Y Channel` color space:

![alt text][image14]

Extracting HOG features of the portion:

![alt text][image15]

Sliding Window Search:

![alt text][image16]

---

### Video Implementation

####1. Here's a [link to my video result](./project_video_output.mp4)

####2. Implementing filter for false positives and method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

![alt text][image13]

Combining detections using heatmap. :

![alt text][image14]

Bounding box for individual cars: 

![alt text][image15]

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

