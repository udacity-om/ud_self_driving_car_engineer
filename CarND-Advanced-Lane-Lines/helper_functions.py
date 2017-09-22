import cv2
import numpy as np

def undistortImage(img, mtx, dist):
   #Undistort the image
   undist = cv2.undistort(img, mtx, dist, None, mtx)
   return undist
    
def warpImage(img, matrix, image_size):
   # Warp the image using OpenCV warpPerspective()
   warped_image = cv2.warpPerspective(img, matrix, image_size)
   return warped_image  
    
def unwarpImage(img, inv_matrix, image_size):
   # Warp the image using OpenCV warpPerspective()
   unwarped_image = cv2.warpPerspective(img, inv_matrix, image_size)
   return unwarped_image 
  
def absSobelThresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Take the derivative in x or y given orient = 'x' or 'y'
    if('x' == orient):
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8((abs_sobel/np.max(abs_sobel))*255)
    # Create a mask of 1's where the scaled gradient magnitude 
    grad_binary = np.zeros_like(scaled_sobel)
    # is > thresh_min and < thresh_max
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary output image
    return grad_binary

def magThresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # Calculate the magnitude 
    abs_sobely = np.sqrt(np.square(sobelx) + np.square(sobely))
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8((abs_sobely/np.max(abs_sobely))*255.0)
    # Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # Return this mask as your binary output image
    return mag_binary

def dirThreshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(dir_grad)
    dir_binary[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    # Return this mask as your binary output image
    return dir_binary
    
def combinedThreshold(image):
   # Apply each of the thresholding functions
   gradx = absSobelThresh(image, orient = 'x', sobel_kernel = 5, thresh = (50, 200))
   grady = absSobelThresh(image, orient = 'y', sobel_kernel = 5, thresh=(50, 200))
   mag_binary = magThresh(image, sobel_kernel = 3, mag_thresh = (20, 150))
   dir_binary = dirThreshold(image, sobel_kernel = 3, thresh = (0.6, 1.1))   
   combined_binary = np.zeros_like(dir_binary)
   combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
   return combined_binary

def getColorFilter(image, thresh_low, thresh_high):
   filtered_image = cv2.inRange(image, thresh_low, thresh_high)
   return filtered_image