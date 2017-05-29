# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on the work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/out.png "fittedLines"

---

### Reflection

### 1. Pipeline descirption

My pipeline consisted of the following steps.
1. Grayscale
2. Gaussian blur
3. Canny
4. Maske region with polygon
5. Infer segments using Hough transform
6. Combine segments into two lines, one for the left line and one for the right

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...
1. Grouping segements with same slope sign together
2. Fiting linear model through the vertices of all the segments in a group (used L1 since less susceptible to outliers than L2 hence avoiding supper noisy lines when applied to video stream)


#### Test on the provided test images: 

![test images][image1]

#### Test on the footage:

available under ~/test_videos_output/


### 2. Shortcomings with pipeline

1. Accumulation of errors: As we make errors, in a real scenario, the car will start swerving bit by bit out of the right trajectory until the camera that is centrally mounted will start capturing footage that this algortihm is not designed to deal with.
2. Highly dependend on visibility of lines. Won't work (or will requier lots of heuristics) in other conditions like night, snow, dirt roads.
3. Not sure it will be able to deal with sharper turn unless we fit a quadratic function through the hough segments.
4. We saw in the challenging task how easily we could trip the model and make think that the rail is a line (wouldn't want to be in taht car)
5. The region mask is hardcoded and is always assuming that the car is centered between the lines. This is one of the reason contributing to the aforementioned problem of accumulated errors raised in point 1

### 3. Possible improvements to pipeline

1. Fit quadratic function through segments instead of lines
2. Dynamic region mask to pull back the car on track in case if starts swerving