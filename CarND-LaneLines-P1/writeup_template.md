# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteRight_Gray.jpg "Grayscale"
[image2]: ./test_images_output/solidWhiteRight_Canny.jpg "Canny"
[image3]: ./test_images_output/solidWhiteRight_Masked.jpg "Masked"
[image4]: ./test_images_output/solidWhiteRight_Hough.jpg "Hough"
[image5]: ./test_images_output/solidWhiteRight_Final.jpg "Final"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline does the following:
-> Converts an image to grayscale
-> Applies gaussian blur with kernal size 7
-> Detects edges in the image using Canny edge detector
-> Masks only the part of image which is of interest, i.e part of image which contains lane lines
-> Runs Hough transform on the masked image which returns two lines, for left and right lanes
-> The final image shows the two lines on the original image

In order to draw a single line on the left and right lanes, I modified the draw_lines() function as follows:
-> For each line in the image i find its slope. I discard the close to vertical and horizontal lines. For all other lines, its slope and points are stored in a list. The lines belonging to the left and right lanes are seperated based on the sign of the slope
-> For each image the mean of the slopes and points are taken. Then the mean of the mean of slopes and points are calculated till this current frame in the video. This is helpful when the lanes are not captured for a particular image. The lane lines can still be drawn using the previous slope and points.
-> Then the intersecpt is calculated using the formula b = y-mx
-> The y value for the top and bottom of the lane lines for both left and right lanes are fixed. A suitable value is selected based on trail and error so that the value will do justification for all videos
-> Again using the formula, x = (y-b)/m, the x values are calculated.
-> Using the 4 points, top and bottom points of left and right lanes, lines are created using cv2.line

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1][image2][image3][image4][image5]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming is that the lane lines may intersect during curves. Other shortcoming is that the lane lines are not identified correctly when the road is not black


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to reduce the lane lines length during curves.

Another potential improvement could be to properly identify weak edges(the case where the road is not black)
