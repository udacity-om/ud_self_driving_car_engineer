#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Model_Architecture.JPG "Model Visualization"
[image2]: ./examples/center_2017_09_05_08_43_37_429.jpg "Center Lane Driving"
[image3]: ./examples/center_2017_09_07_09_02_14_821.jpg "Recovery Image 1"
[image4]: ./examples/center_2017_09_07_09_02_14_889.jpg "Recovery Image 2"
[image5]: ./examples/center_2017_09_07_09_02_14_958.jpg "Recovery Image 3"
[image6]: ./examples/center_2017_09_07_09_02_15_027.jpg "Recovery Image 4"
[image7]: ./examples/center_2017_09_07_09_02_15_098.jpg "Recovery Image 5"
[image8]: ./examples/center_2017_09_07_09_02_15_165.jpg "Recovery Image 6"
[image9]: ./examples/center_2017_09_07_09_02_15_232.jpg "Recovery Image 7"
[image10]: ./examples/center_2017_09_07_09_02_15_300.jpg "Recovery Image 8"
[image11]: ./examples/center_2017_09_07_09_02_15_368.jpg "Recovery Image 9"
[image12]: ./examples/center_2017_09_05_08_42_50_473.jpg "Normal Image"
[image13]: ./examples/center_2017_09_05_08_42_50_473_Flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a preprocessing steps of cropping, resizing, normalizing and mean centering. (model.py line)
This is followed by a modified LeNet architecture. (model.py lines)

ReLU layers introduce nonlinearity and speeds up training. (code line 20) 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also collected data by driving in the opposite direction. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep building upon a simple network, reduce the mean squared error of training and validation and to check that the car drove within the track.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it also works on images as input. I had to make changes in the last layer as I needed only one output whereas LeNet had 10 outputs.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. To reduce training time I cropped the image to concentrate only on the road and resized the image to (64, 64, 3). To better train my network I randomly choose images from the three cameras with corresponding adjustment to steering angle. I normalized and mean centered the images as part of preprocessing. I found that my first model(LeNet) did well with respect to keeping the vehicle on track but the mean square error plot showed that the model was overfitting at the end of 5 epochs. 

To combat the overfitting, I modified the model by adding another Fully Connected layer. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle had difficulty recovering from sides. To improve the driving behavior in these cases, I modified the pre-processing section so that the training images were in RGB format. This is the same format which drive.py uses.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 

Convolution Layer with 6 filter of size 5x5
ReLU activation
Convolution Layer with 6 filter of size 5x5
ReLU activation
Flatten the layer
Fully Connected layer with 120 neurons
Fully Connected layer with 84 neurons
Fully Connected layer with 10 neurons
Output neuron

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I also recorded two laps on track one but moving in the opposite direction so that the model doesnt get biased to one steering direction.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer towards the center when it approaches the sides of road. These images show what a recovery looks like starting from left side of the road:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

To augment the data sat, I also flipped images and angles to further reduce the bias towards one steering direction. For example, here is an image that has then been flipped:

![alt text][image12]
![alt text][image13]

After the collection process, I had 18074 number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
