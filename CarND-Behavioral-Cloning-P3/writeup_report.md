# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_driving.png "central driving"
[image2]: ./examples/processed.jpg "cropped from all 3 cameras"
[image3]: ./examples/recovery_right.png "Recovery Image"
[image4]: ./examples/recovery_left.png "Recovery Image"
[image5]: ./examples/original.jpg "original"
[image6]: ./examples/flipped.jpg "flipped"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of two convolution layers with 5x5 kernel and depths 6. First conv layer has a stride of 2 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 
Pooling layers are added after each conv. We also added a pooling layer right after normalizing and cropping the image. That acted as a cheap way of bluring the image to help with generalization and protect against overfitting.

The conv layers are followed by 3 dense layers with a dropout of 0.5 between the first 2 layers. The last fully connected layer with the output has no dropouts.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Nesterov variant of the adam optimizer (NADAM), so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving in the other direction. That resulted in 6,944 frames which is relatively low and can be prone ot overfitting if not careful. 

It is important to note that we purposely chose the lowest resolution and quality while capturing data in the simulator to first reduce complexity of the needed model and also as a mean of helping the generalization of the model. That is just intuitive considering that often in image processing we get to add some noise to our image or reduce it in size to achieve similar purposes.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start simple with no image processing and using a linear regression model as a baseline.
We later added convolutional layers with a LeNet architecture and using early stoping whenever we observed an increase in validation loss. We augmented the data with flipped images and the images from the side cameras with corrected angles +25 and -25 degrees.
Results started to look better but we were still leaving the track.
Scaling the image to have mean~0 and std~0.5 helped in terms of convergence speed and resulted in a smoother driving. The car was actually able to clear the track at this stage.

We decided to explore further by adding cropping which should reduce the information load and potentially help. We faced overfitting problems since the data was easier to learn yet the model, even with such a small capacity, was able to overfit. We aleviated the problem by adding an additional max pooling layer right after the cropping and before feeding the image into the first conv layer. That served the purpose of forcing the model to generalize better.

We also tried a variant of hte Nvidia architecture which was overkill for our application here considering that the footage from the simulator was not too hard when compared to ral footage that the Nvidia team was working with.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

We were frequently running the simulator to see how well the car was driving around track one as an addtional mean of assessing whether we were doing any progress in the right direction.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
The best final model was actually the one with the least capacity as described in the following section

#### 2. Final Model Architecture

```python
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,subsample=(2,2),activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, we first recorded almost a full lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

We then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if starting to diverge from track. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]

We also drove in the opposite direction for a bit.

The while data gathering resulted in 6,944 frames.

To augment the data sat, we also flipped images and angles thinking that this would help prevent just learning to steer left. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

We also used images from the side cameras and corrected for the angle by adding/substracting 25 degrees to the steering angle. We croppe the image to remove the bottom and upper part which do not add any information to the model.

![alt text][image2]

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the validation loss monotically dicreasing. I used a Nesterov variant Adam optimizer so that manually training the learning rate wasn't necessary. Nesterov usually helps with convergence speed by applying a look ahead approach to the momentum before applying the gradient.

A generator was used to generate and augment the data on the CPU in batches while the network was being trained on the CPU. Image cropping and normalization is done as part of the network