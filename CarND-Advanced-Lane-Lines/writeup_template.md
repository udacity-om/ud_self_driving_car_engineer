**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Compute the perspective tranform matrix.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./results/undistorted.jpg "Undistorted"
[image2]: ./results/warped.jpg "Warped"
[image3]: ./results/undistorted_lane_image.jpg "Undistorted Lane Image"
[image4]: ./results/warped_lane_image.jpg "Warped Lane Image"
[image5]: ./results/lane_filter_1.jpg "Lane Filter 1"
[image6]: ./results/lane_filter_2.jpg "Lane Filter 2"
[image7]: ./results/histogram_lane_image.jpg "Histogram Lane Image"
[image8]: ./results/finding_lane_pixels_1.jpg "Finding Lane Pixels 1"
[image9]: ./results/finding_lane_pixels_2.jpg "Finding Lane Pixels 2"
[image10]: ./results/final_image.jpg "Final Image"
[image11]: ./results/estimating_lane_pixels.jpg "Final Image"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Helper Functions

To help modularize and reduce the size of final code, I have created helper_functions.py which holds small functions as below:
* undistortImage : undistort, using `cv2.undistort()`, an image given camera matrix and distortion coefficients. 
* warpImage : warp, using `cv2.warpPerspective()`, an image given the transform matrix
* unwarpImage : unwrap, using `cv2.warpPerspective()`, an image given inverse transform matrix
* absSobelThresh : apply Sobel filter, using `cv2.Sobel()`, in the given orientation
* getColorFilter : apply color filter, using `cv2.inRange()`, in the given range

### Camera Calibration

The code for this step is contained in the IPython notebook located in "./camera_calibration.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I then save the camera calibration maatrix and distortion coefficients in camera_params.pkl file. I applied this distortion correction to the test image using the helper function `undistortImage()` and obtained this result: 

![alt text][image1]

### Perspective Transform

The code for this step is contained in the IPython notebook located in "./perspective_transform.ipynb".

First I choose four points on the source image and corresponding desired points on the destination image. The transformation matrix was found by using the `cv2.getPerspectiveTransform()` function. I also found the inverse transformation matrix by swapping source and destination points. I saved these matrices in transform_matrix.pkl file. I then applied this trasformation to the test image using `warpImage()` helper function and obtained the below result. The four red dots on the left image are source points and 4 red dots on the right image are destination points. The source points were selected such that they lie on the lane lines of the original image. The destination points were adjusted until the lanes lines appeared parallel in the warped image. 

Final source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570, 470      | 335, 0        | 
| 720, 470      | 965, 0        |
| 1130, 720     | 980, 720      |
| 200, 720      | 320, 720      |

![alt text][image2]

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

The image is corrected for distortion(line 10, cell 11, advanced_lane_lines.ipynb) as mentioned in  Camera Calibration section above. Helper function `undistortImage()` is used to undistort the image as shown below. Both images may look same but notice the difference in the hood in the bottom corners:

![alt text][image3]

#### 2. Example of the transformed image.

The image is warped(line 13, cell 11, advanced_lane_lines.ipynb) as mentioned in Perspective Transform section above. Helper function `warpImage()` is used to warp the image as shown below:

![alt text][image4]

#### 3. Color Filtering and gradient filtering to create a thresholded binary image.

The lane lines are in either yellow or white color, so I used yellow and white color filters on the RGB image to generate a binary image(line 3-6, cell 5, advanced_lane_lines.ipynb). I used the website http://colorizer.org/ to help me get range for yellow and white colors(line 2-5, cell 2, advanced_lane_lines.ipynb). I also applied sobel x filter to generate another binary image(line 9-11, cell 5, advanced_lane_lines.ipynb). The threshold was decided after several trail and error. The color filter and sobel filter compliment each other by detecting parts of lane lines which the other filter cannot. I combined these two binary images to form the final binary image(line 14-15, cell 5, advanced_lane_lines.ipynb). Here's an example of my output for this step.

![alt text][image5]

An example where the color and sobel filters compliment each other. The Yellow-White color filter is able to extract the bottom part of the lane lines and the Sobel X filter is able to extract the top part of the lane lines  :

![alt text][image6]

#### 4. Identifing lane-line pixels and fit their positions with a polynomial

Identification of lane pixels is done by the function `findLanePixels()` (line 27-28, cell 11, advanced_lane_lines.ipynb))

Identifying lane pixels involved the following steps:
* Taking histogram along all the columns in the lower half of the image
    * I added all pixels values along each column in the image(line 50, cell 12, advanced_lane_lines.ipynb)    
    
![alt text][image7]

* Finding x values corresponding to peaks in the histogram
    * Peaks in histogram are good indicator of lane lines. The x value corresponding to the peaks can be considered as x-position of the base of the lane lines(line 56-61, cell 12, advanced_lane_lines.ipynb)
* Sliding window approach to identify lane pixels
    * The base of the lane lines can be used as good starting point to start searching for lane lines. I placed a sliding window, with fixed height and width, around the line center to find the line pixels. I then followed the line all the way up, re-centering the window whenever required and possible, to find the entire line pixels(line 64-102, cell 12, advanced_lane_lines.ipynb)
  
Using the lane pixels I fit a 2nd order polynomial, x = ay^2 + by + c (line 31-32, cell 11, advanced_lane_lines.ipynb), and the result is as shown below:

![alt text][image8]

Another image showing how the window re-centers for curved lanes

![alt text][image9]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

* The radius of curvature is calculated using the function `findRadiusOfCurvature()` (line 83-84, cell 11, advanced_lane_lines.ipynb)
   * The radius of curvature is given by: (1 + (x')^2)^1.5 / abs(x''). Taking the derivatives of x = ay^2 + by + c,
   * First derivative: x' = 2ay + b
   * Second derivative: x'' = 2a
   * Substituting in the radius of curvature formula, we get (1 + (2ay + b)^2)^1.5 / abs(2a). This is implemented in line 4-12, cell 6, advanced_lane_lines.ipynb

* The position of the vehicle is calculated using the function `findCarPosition()` (line 106, cell 11, advanced_lane_lines.ipynb)
  * I found the position of the veicle by taking the difference of center of image and center of lane lines. The center of the image is half of the image width(1280/2 = 640). The center of the lane lines is the average of x position of right and left lanes at the bottom of the image, i.e x value corresponding to y = image height(720). This is implemented in line 3-10, cell 7, advanced_lane_lines.ipynb

#### 6. Example image of the result plotted back down onto the road such that the lane area is identified clearly.

The lane is drawn onto the warped image using `cv2.fillPoly()` function and then the image is unwarped using helper function `unwarpImage()`. This unwarped image is combined with the original image using `cv2.addWeighted()` function to get the final image. Using `cv2.putText()` function, the radius of curvature and position of car are displayed on the final image. I implemented these steps in lines 109-135, cell 11, advanced_lane_lines.ipynb.  Here is an example of my result on a test image:

![alt text][image10]

---

### Pipeline (video)

I then tested the pipeline on a video and here's the [link to my video result](./project_video_output.mp4)

The pipleline used for the image and video is the same. Here i will explain about the exta code that was required to handle video(back to back images). 

To handle both images and video, I defined a class called `Line()`(cell 12, advanced_lane_lines.ipynb). The lane lines are instances of this class. This helps in easier handling of variables from image to image. 

As explained in the image pipeline section above, the lane lines in an image are calculated using the histogram and sliding window method. We dont have to do this for every image of the video as changes between two consecutive images are very less. Taking this as an advantage and also to reduce execution time, the lane pixels in the next image are found by doing a local search around the previously fit polynomial. The switching between the functions `findLane()` (which finds the lane pixels from scratch) and `estimateLane()` (which estimates the lane pixels using previously fit polynomial) is done by the function `findLanePixels()` (lines 4-7, cell 9, advanced_lane_lines.ipynb). If the lane pixels are not at all found, either by `findLane()` or `estimateLane()`, then the previously found pixels are used which were stored in the class variable `prev_allx` (lines 11-18, cell 9, advanced_lane_lines.ipynb). This takes care of bad frames. The green shaded area below shows where I searched for the lines:

![alt text][image11]

Even though the lane pixels are found, due to the coloring/gradient filter limitations under different lighting conditions of the road, the found pixels might differ from the previous image considerably and this may lead to glitchy lane finding and glitchy radius of curvature and car position. To smoothen these changes, the polynomials of the last 10 images are stored in class variable `good_fit()`  and their average is stored in class variable `best_fit()` (lines 137-138, cell 12, advanced_lane_lines.ipynb). This best polynomial is used for the rest of the calculations. 

There could be scenarios were the images are so bad that the found pixles vary drastically from the previous images. In these cases the current image is completely discared and previous best polynomial is used. The check for very bad pixels is done by the function `sanityCheck()` (line 87-88, cell 11, advanced_lane_lines.ipynb). This function checks for the lane width and the consistency of left and right radius of curavture. If the lane width is not within +/-0.5 m of the known lane width(this number was finalized to 3m after running through the project video) or if the left and right curavtures dont agree with each other within an experimentally calculated threshold, then the current frame doesnt pass the sanity check and is dicarded. These checks are implemented in lines 8-21, cell 8, advanced_lane_lines.ipynb.

If I get 10 consecutive bad images, then the lane lines of the next frame is calculated from scratch. This is implemented in lines 90-103, cell 11, advanced_lane_lines.ipynb.

---

### Discussion

#### 1. Problems / issues faced in implementation of this project.  Where is the pipeline likely to fail?  What can be done to make it more robust?

Below are the problems that I faced during the implementation:
* Lane selection filters
    * I spent a lot of time experimenting with different filters. First i tried to work on HLS color space but i failed to get the thresholds which would work on all test images. 
    * I then decided to use gradient threshold in x direction on the L and S channels of the HLS color space. This too failed as i wasnt able to get optimum thresholds.
    * After going through the discussions in the forum, I decided to use RGB color to extract yellow and white colors combined with gradient thresold in x direction on gray image. This helped a lot and after few experiments I zeroed on to the thresholds. 
* Handling images where no line pixels could be selected
    * Such images were causing the video pipeline to fail as the size of list storing pixel positions would be zero and case exceptions. this was handled by using previous pixels if current pixels couldnt be found.
* Smoothening the glitches
    * Deciding the number of previous frames to store and what to store was another time consuming task. After experimenting with storing lane pixels and storing polynomials, I decided on storing polynomials for the last 10 images 

Due to the limitations of the color/gradient filter used, the pipeline may fail if there is a long stretch of brightly lit road. This can be handled with a more robust filtering technique. Another scenario where the pipeline could fail is roads with steep curve as the points chosen for warping the image is hardcoded to one kind of road. This can be handled by choosing the points on the fly.
