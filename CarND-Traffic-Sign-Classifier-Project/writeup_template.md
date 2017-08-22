#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"

[image3]: ./Loss_Plots/Loss1.jpg "Loss 1"
[image4]: ./Loss_Plots/Loss2.jpg "Loss 2"
[image5]: ./Loss_Plots/Loss3.jpg "Loss 3"
[image6]: ./Loss_Plots/Loss4.jpg "Loss 4"
[image7]: ./Loss_Plots/Loss5.jpg "Loss 5"
[image8]: ./Loss_Plots/Loss6.jpg "Loss 6"
[image9]: ./Loss_Plots/Loss7.jpg "Loss 7"
[image10]: ./Loss_Plots/Loss8.jpg "Loss 8"
[image11]: ./Loss_Plots/Loss9.jpg "Loss 9"
[image12]: ./Loss_Plots/Loss10.jpg "Loss 10"

[image13]: ./web_images/AheadOnly.jpg "AheadOnly"
[image14]: ./web_images/RoadWork.jpg "RoadWork"
[image15]: ./web_images/SpeedLimit70.jpg "SpeedLimit70"
[image16]: ./web_images/Stop.jpg "Stop"
[image17]: ./web_images/TurnRight.jpg "TurnRight"

[image18]: ./Training_Data_Distribution.jpg "Training_Data_Distribution"
[image19]: ./Validation_Data_Distribution.jpg "Validation_Data_Distribution"
[image20]: ./Testing_Data_Distribution.jpg "Testing_Data_Distribution"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. The data set is of German traffic signs. Each sign belongs to one of the 43 signs(also called classes). The dataset contains multiple images of same sign.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in training, validation and testing dataset

![alt text][image18]
![alt text][image19]
![alt text][image20]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because having color images increases the complexity of the model as it will have to work on 3 color channels. Grayscale is combination of the three channels into ingle channel. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data so that the training time can be reduced. Other advantage of normalizing is to makes all the feature values to have comparable range but in image data all the pixel values are already between 0 to 255.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	       | Description	        	                         | 
|:---------------------:|:---------------------------------------------:| 
| Input         	       | 32x32x1 Grayscale image   		                  | 
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 28x28x6 	   |
| RELU			               |						                                         |
| Max pooling	      	   | 2x2 stride,  outputs 14x14x6 			              | 
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 10x10x16 	  |
| RELU			               |						                                         |
| Max pooling	      	   | 2x2 stride,  outputs 5x5x16 			               | 
| Flatten	      	       | Flatten data from conv layer2     |
| Fully connected	      | input 400, output 120	 	                     |
| RELU			               |						                                         |
| Dropout	              | 				                                          |
| Fully connected	      | input 120, output 84				                      |
| RELU			               |						                                         |
| Dropout	              | 				                                          |
| Fully connected	      | input 84, output 43				                       |
|			                    |                                             		|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer with batch size 128, epoch 60, keep probability of 0.6 for training and learning rate 0.0005

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.959 
* test set accuracy of 0.933

I opted for an iterative method:
* 1) I first tried with the default LeNet architecture, default hyperparameters (Mu = 0, Sigma = 0.1, Learn_Rate = 0.001, Epoch = 10) and images as it is without any pre-processing and got training/validation accuracy of 0.983/0.863
* 2) To increase the accuracies i first grayscaled and normalized the images and every other parameters same as before. This time got  training/validation accuracy of 0.983/0.863. Plotting the loss graph showed me that the model was overfitting

![alt text][image3]

* 3) With my limited knowledge at this stage, to reduce overfitting, i reduced Sigma to 0.05, reduced Learn_Rate to 0.0005 and increased Epoch to 20. The accuracies decreased, training/validation accuracy of 0.975/0.859, and the model was still overfitting

![alt text][image4]

* 4) After going through the forums regarding overfitting, i decided to add dropouts after fully connected layer1 and fully connected layer2 with a Keep_Prob of 0.7. Other parameters same as previous. This model showed some improvement with training/validation accuracy of 0.962/0.907 but validation accuracy still less than 0.93

![alt text][image5]

* 5) So decided to play with Keep_Prob = 0.8. With this model the training/validation accuracy was 0.969/0.898. So no luck here and also the model looked like overfitting.

![alt text][image6]

* 6) Next thought of reducing the Keep_Prob to 0.4. With this the accuracies reduced to training/validation accuracy of 0.937/0.879.

![alt text][image7]

* 7) Still playing with Keep_Prob, i increased it to 0.5 and the accuracies improved to training/validation accuracy of 0.966/0.916

![alt text][image8]

* 8) Next I kept increasing Keep_Prob to 0.6 and Epoch to 30 and there was slight improvement in accuracies, training/validation accuracy of 0.983/0.916

![alt text][image9]

* 9) In the next model, i started tuning Sigma to 0.1 and increased Epoch to 60. This resulted in a good boot in accuracies, training/validation accuracy of 0.999/0.95

![alt text][image10]

* 10) Experimenting more to improve the model, going through forums, i decided to add dropouts after every layer. Also to provide lower level information to the fully connected layer for better classification, I provided both convolution layer outputs to first fully connected layer. Also reduced Epoch to 50. This resulted in a slight reduction in accuracy, training/validation accuracy of 0.994/0.943

![alt text][image11]

* 11) So decided to go back to model 9, Mu = 0, Sigma = 0.1, Learn_Rate = 0.0005, Epoch = 60 and dropout after fully connected layer 1 and fully connected layer 2. During this run the accuracies slightly changed to training/validation accuracy of 0.998/0.959. I am choose thsi as my best model.

![alt text][image12] 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image13] ![alt text][image14] ![alt text][image15] 
![alt text][image16] ![alt text][image17]

The first image might be difficult to classify because of green color behind the sign
The second image might be difficult to classify because of reflection
The third image might be difficult to classify because of blue and green color behind the sign
The fourth image might be difficult to classify because of blurring
The fifth image is not difficult to classify


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only      		| Ahead Only   									| 
| Road Work     			| Road Work 										|
| Speed Limit 70					| Ahead Only											|
| Stop	      		| Stop				 				|
| Turn Right			| Turn Right      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is less comapred to the test accuracy of 0.933.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 30th cell of the Ipython notebook.

For the first image, the model is very sure that this is a Ahead Only sign and the image is of Ahead Only sign
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Ahead Only    									| 
| 0     				| Yield 										|
| 0					| No Passing											|
| 0	      			| Turn Left Ahead					 				|
| 0				    | Speed Limit 60      							|

For the Second image, the model is very sure that this is a Road Work sign and the image is of Road Work sign
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Road Work    									| 
| 0     				| Bicycle Crossing 										|
| 0					| Dangerous Curve To The Right											|
| 0	      			| Slippery Road					 				|
| 0				    | Genaral Caution      							|

For the Third image, the model is relatively sure that this is a Ahead Only sign and the image is of Speed Limit 70	sign
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.46         			| Ahead Only    									| 
| 0.33     				| Dangerous Curve To Left									|
| 0.21					| Speed Limit 60											|
| 0.005	      			| Speed Limit 20					 				|
| 0.001				    | Go Straight Or Right      							|

For the Fourth image, the model is relatively sure that this is a Stop sign and the image is of Stop sign
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Stop    									| 
| 0.008     				| Speed Limit 70 										|
| 0.003					| Speed Limit 30											|
| 0.000003	      			| Speed Limit 20					 				|
| 0.0000004				    | Go Straight Or Right      							|

For the Fifth image, the model is relatively sure that this is Turn Right Ahead sign and the image is of Turn Right Ahead sign
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Turn Right Ahead    									| 
| 0.001     				| Roundabout Mandatory 										|
| 0.0008					| Stop										|
| 0.0008	      			| Speed Limit 30					 				|
| 0.00007				    | Keep Left    							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
