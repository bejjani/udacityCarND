# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./writeup_img/imbalanced.png "imbalanced"
[image2]: ./writeup_img/original.png "original"
[image3]: ./writeup_img/stretched.png "stretched"
[image4]: ./writeup_img/translated.png "translated"
[image5]: ./writeup_img/rotated.png "rotated"
[image6]: ./writeup_img/noise.png "noise"
[image7]: ./writeup_img/signs_sample.png "sample"
[image8]: ./writeup_img/augmented_1.png "augmented_1"
[image9]: ./writeup_img/augmented_2.png "augmented_2"
[image10]: ./writeup_img/augmented_3.png "augmented_3"
[image11]: ./writeup_img/new_predictions.png "new_predictions"

## Rubric Points

---
### Writeup

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy calculate summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the traing data is distributed amongst the classess

![alt text][image1]

Highly imbalanced set that will bias the model towards the more prominent classes.


Here's a sample of some of the pictures

![alt text][image7]

most pictures seem to be centered but with variable quality specially in terms of brightness.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


* grayscale
* stretch
* translate
* rotate
* added noise
* normalize to have mean~0 and std~0.5

Here is an example of a traffic sign image after grayscaling.

![alt text][image2]

Stretched

![alt text][image3]

Translated

![alt text][image4]

Rotated

![alt text][image5]

Added noise

![alt text][image6]


These transformations were applied at random on the original training set with random (but bounded distortion scale).
That helped us in achieviung class balance by augmenting the underepresented classes. An additioonal 1% oversampling has also been added to allow an oversampling of the majority class as well.

The oversampling achieves a balance in the dataset while at the same time make the model more robust by adding additional distortion. Grayscaling has been used for simplicity by reducing hte model complexity to 1 inout channel and the literature seems to indicate that there is no loss in performance by doing so.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image         				| 
| Convolution     	| 1x1 stride, 5x5 kernel, same padding, outputs 32x32x32, + Bias	|
| Max pooling	      	| 2x2 stride, 2x2 kernel, outputs 16x16x32 				|
| Convolution     	| 1x1 stride, 5x5 kernel, same padding, outputs 16x16x64, + Bias 	|
| Max pooling	      	| 2x2 stride, 2x2 kernel, outputs 8x8x64|
| Fully connected		| 4096x512 + Bias|
|ReLu|
| Dropout		| 0.5        									|
| Fully connected		| 512x84 + Bias|
|ReLu|
| Dropout		| 0.5        									|
| Fully connected		| 84x43 + Bias|
|Softmax|
| Crossentropy loss		||


 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

It is a simple model with 2 Conv layers with bias and maxpooling followed by 3 fully connected layers with bias and 50% dropout except for the output layer that is fed into a softmax.

Adam was used as na optimizer and a weight decay was applied to drop the learning rate by half after 10,000 iterations


| Hyperparameter         		|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Batch size         		| 128         				| 
| epochs     	| 27	|
| learning rate | 0.001 (reduced by half every 10,000 iterations)|
| dropout | 50% |
| initialization of fully connected layer weights | truncated normal stddev=1/sqrt(n_inputs)|
| initialization of conv layers weights | truncated normal stddev=0.01|
| initialization of biases | 0.0 |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of last batch ~91%
* validation set accuracy of 96.1%
* test set accuracy of 94.1%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The architecture was not changed except for the number of neurons in the first fully connected layer. It was increased once if was observed that the model was not learning well after teh first few epochs with a 50% dropout. That was an indication that the model did not have enough capacity to learn the actual data to be abel to add a dropout. The capacity increase in the FC layer helped.

* What were some problems with the initial architecture?
Low capacity. My initializations of the weights were also too high. I reduced the std for the conv layer to 0.01 and for the FC as described above which helped getting the model to start learning and avoid numerical instability. Setting the biases to 0.0 also helped with convergence speed compared to a random initialization and even to when set to 1.0

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Didn't run into overfitting due miostly to the introduction of the agressive dropout of 50%

If a well known architecture was chosen:
* What architecture was chosen? LeNet with just 2 Conv layers
* Why did you believe it would be relevant to the traffic sign application? Natural first choice for any image classification task due to the translation invariance property and the shared parameters in a conv layer that reduces model capacity
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? achieved over 94% accuracy on testing set in just over 6min of training on a single GPU
 

###Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image11]

Potential difficulties in classifying those images

| Image			        |     difficulty	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Vehicles      		| Vandalized sign making it hard to recognize  					| 
| Pedestrians     			| Sign has a round shape instead of hte triangular one  	|
| Slippery Road					| Covered in snow											|
| Children crossing	      		| Glare on the sign and additional sign at the bottom					 	|
| Road Work			| Not conventional sign that sits on the ground and has additional text at the bottom      |

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     explanation	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Vehicles      		| vandalization made it hard for hte model to guess and associated it to a bumpy road sign. It would also be hard for a human being to identify other than for the clue that it is a round sign. A bumpy road sign is a triangular one which suggests that the model does not rely/learn to use that feature in identifying a bumpy road? | 
| Pedestrians     			| Classifed as Genral Caution due to the unobserved sign before. Not sure what happened there sign the pedestrian is clear but looks like the shape of the sign through the model off contradicting my previous comment  	|
| Slippery Road					| Covered in snow and classifed as Children Crossing with high confidence. That's conserning seing such a hich confidence with such a poor quality image.											|
| Children crossing	      		| Right classification despite the glare 					 	|
| Road Work			| Correct classfication despite the unconventional positioning of the sign      |


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. The test cases were chosen specificly to trip the model and showcase how fragile the model could be in a real life scenario without having a certain certainty built in the model to express how confident the model is in its decision and whether it should be trusted or not. This uncertainty could be observed for the Slippery Road sign with a softmax of ~0.55 but it is still quite high in my opinion

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Answered in previous question and numbers reported in the picture

For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


