# Vehicle Detection Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/weaklearners_bbox.jpg
[image4]: ./examples/weaklearners_voting.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for the method is contained in `featurizer.py`
Line #160 in `classifer.py` and line #46 in `finder.py` invoke this method

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

The HOG parameters selection, mostly the channel selection, was performed as part as the hyperparameter tuning of the whole pipeline including the classifier and the bouding box detection.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained different SVM models (liner and rbf) with `classifier.py` that takes care of performing the parameter tuning for the SVM through cross-validation. The colorspace and the hog channels are specified as parameters but are not tuned during the cross-validation.
I ended up using just HOG features with different colormaps, channels, and different SVM learners (linear and rbf) and combined them through an `emulated` majortiy voting mechanism used in ensemble modeling. I describe my motivation and approach in more detail in the next sections.

But in general I kept the orient=9,pix_per_cell=8,cell_per_block=2 fixed for HOG. Features were scaled to zero mean and unit variance at line #104 in `classifer.py` and #77 in `finder.py`

The learnes that made it in the final ensemble model are as follows:
1. `clf_pickle_YUV_2`: YUV color space with just channel 2 HOG extraction and with an rbf kernel
2. `clf_pickle_YCrCb_2_linear`: YCrCb color space with just channel 2 HOG extraction and with a linear kernel
3. `clf_pickle_YUV_1_linear`: YUV color space with just channel 2 HOG extraction and with a linear kernel


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The serach is implemented in `finder.py`. I decided to limit the search space to ```ystart=400, ystop=656``` corresponding to the lower section of the frame where it is more lkely to find cars close enough to care about. I perform a rescaling twice to vary the search patch size lines #103 and #106. The decision to reduce the scans was intentional to reduce the inference time when performing the detection.
The overlap of the patches is controlled by `cells_per_step = 1` at line #41 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on two scales using YCrCb 3-channel at first with an rbf kernel. The results were mixed so I dicided to ensemble more models together. Doing so was expensive from the training side so I decided to subsample the training set and be more selective in my frame choice. Since for this project we are just driving in the fast lane, training to recognize cars coming from the left was not needed and hence were flipped to add more observations to righ vehicles. Rear shots were dropped and the non-vehicle pictures were subsampled to balance the negative and positive class ratios.
I then started training more models with different combinations of HOG channels and colorspace still with mixed results. Using ensemble methods was the obvious choice at this stage the alternative being to add more features. I trained weaker linear classifers and ensembled them by using the heatmap functionality that was already in palce to emulate a majority vote. In `main.py`, I set `heat_threshold = 12` at line #27 to achieve this.
I could have settled on training stringer classifiers at the expence of long evaluation time but that was not an option given the long video and also given the number of iterations needed to tune the pipeline.

Here I show an image of how the weak learners detect tons of false positives but since we are emulating a majority voting mechanism, like in ensemble modeling for a classifer, we are able to get rid of the false positive during the heatmap generation and only keep the high confidence detection boundery boxes.

![alt text][image3]
![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

This step in the pipeline is implemented in `heat.py`. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

I could have implemented an additional smoothing mechanism by trcking the boxes through successive frames which should allow for a more robbust detection. Because of the lack of time, this was not explored.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My detection pipeline is very slow due to the ensembe approach I used requiring running differnet models. That is obviously a shortcoming that would prevent this model from making it into a real-time system. I should have used more features and stick with one strog model instead.
The complexity of the pipeline and the number of different parameters to tune at every stage and their tight interdependency made the whole tuning long and painful. An automated tuning framework could have been put in place to eliminate the repetitive tasks.
State of the art approaches like YOLO and SSD eliminate the need for window searches speeding the detection which is vital in autonomous driving.
