#**Traffic Sign Recognition** 

##Writeup

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

[image1]: ./examples/Visualization.png "Visualization"
[image2]: ./examples/Train_data_dist.png "Training Data Distribution"
[image3]: ./examples/Valid_data_dist.png "Validation Data Distribution"
[image4]: ./examples/Test_data_dist.png "Test Data Distribution"
[image5]: ./mysigns2/3.jpg "Traffic Sign 1"
[image6]: ./mysigns2/11.jpg "Traffic Sign 2"
[image7]: ./mysigns2/25.jpg "Traffic Sign 3"
[image8]: ./mysigns2/28.jpg "Traffic Sign 4"
[image9]: ./mysigns2/stop.jpg "Traffic Sign 5"
[image10]: ./examples/lenet5.png "LeNet Architecture"
[image11]: ./examples/Test_accuracy.png "Training Accuracy"
[image12]: ./examples/valid_accuracy.png "Validation Accuracy"
[image13]: ./examples/Prediction_results.png "Prediction Results"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43
* The minimum number of signs in a training class is 180 and the max is 2010

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
![alt text][image1]

It is a bar chart showing how the data distrubution
![alt text][image2]
![alt text][image3]
![alt text][image4]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I converted the images to grayscale. I did this so the model did not have to worry about color of the stop signs and it could solely focus on the features of each sign. This would result in a better model results in the training and validation data.

The other preprocessing I did was to normalized the image data because the model performes best (required) on data that has zero mean and equal variance. This is due to the fact that during the process of training our network, we multiply(weights) and add (biases) to these initial inputs in order to cause activations that we then backpropogate with the gradients to train the model.

We'd like in this process for each feature to have a similar range so that our gradients don't go out of control and that we only need one global learning rate multiplier.

The data set was also shuffled before training so that the model would not consider the order of the images a feature to learn.

I explored the idea of generating additional test images by roatating, scaling, distorting the original testing database, but since my model was already meeting the performance criteria, I decided it wasn't needed.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model used the LeNet architecture as a started point but then another addtional 5x5 convolutional layer was added along with replacing subsampling with max pooling. It then flattened the output of the final convolutional layer and fed that into 3 fully connected layers with a 50% dropout between all but the final output layer. Relu activation was used.

My final architecture consisted of the following layers:


Architecture

Layer 1: Convolutional. Input = 32x32x1. 5x5 Conv Output = 28x28x6.

Activation. Relu

Pooling. max pool -> Output 14x14x6.

Layer 2: Convolutional. Input 14x14x6

Activation. Relu

Pooling. Max pool -. Output 5x5x16

Layer 2a: Conv 5x5

Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. Output = 412

Layer 3: Fully Connected. In 412 Out 122

Activation. relu

Layer 4: Fully Connected. In 122 Out 84 

Activation. relu

Layer 5: Fully Connected (Logits). In 84 Out 43 classes

Return the result of Layer 5

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 126 for 10 epochs and a learning rate of 0.00095. I used a 50 percent drop out between my first two fully connected layers . The optimizer used was the "AdamOptimizer."

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.941
* test set accuracy of 0.923

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LeNet
* What were some problems with the initial architecture?
Poor performance
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I added an extra convolutional layer along with adding dropout between the last two layers and replacing subsampling with max pooling. It was changed due to underfitting and poor accuracy on both the training and validation sets.

* Which parameters were tuned? How were they adjusted and why?

Number of epochs was adjusted. It started at 10 and I increased it to 15, but after seeing overfitting in the training data it was reduced to 10. The learning rate was slightly adjusted to result in a slower training rate but I higher accuracy on training and validation sets.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolution works well to extract the features that define a given traffic sign. Dropout also helped by not allowing the model to rely on one certain feature of a traffic sign but rather learn multiple ways to claffisy different signs.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image9] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The children crossing sign might be difficult to classify becuase it contains another sign just below it that the training set does not contain.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

My Data Set Accuracy = 0.800


Image 1
Image Accuracy = 1.000

Image 2
Image Accuracy = 1.000

Image 3
Image Accuracy = 0.667

Image 4
Image Accuracy = 0.750

Image 5
Image Accuracy = 0.800


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.935. For this number of images, this is close to the test set accuracy.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

For all of the images but the children crossing with another additional sign below it the model was very sure (97-100%)

However, for the children crossing sign, it miss identified it as an end of all speed and passing limits sign (32). 

![alt text][image13]

