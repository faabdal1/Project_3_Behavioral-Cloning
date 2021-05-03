# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./writeup/cnn.png "Nvidia Model"
[image1]: ./writeup/loss_gen_long.png "30 epoch"
[image2]: ./writeup/center.jpg "Center"
[image3]: ./writeup/start.jpg "Recovery Image"
[image4]: ./writeup/close.jpg "Recovery Image"
[image5]: ./writeup/recover.jpg "Recovery Image"
[image6]: ./writeup/flip.jpg "Normal Image"
[image7]: ./writeup/flip2.jpg "Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

I have modified the default speed of 9 mph to 20 mph.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

I have implemented convolutional neural network introduce by nVidia in this research paper: https://arxiv.org/pdf/1604.07316.pdf

The CNN inroduced in the paper:
![nVidia CNN][image0]

I have implemented that CNN due to recommendation by the course but also found it sutaible for this task after reading the research paper. 

I have tried to use Lenet but the vehicle could not stay on road very well. I have also tried to implement pretrained network discussed during the course but my data set was not sufficient to well train it and mainly the design of such nework was sutaible for object classification, not vehicle control (it was still able to drive but not to stay within lanes around the track).

#### 1. An appropriate model architecture has been employed

To normalize the input data, I have preprocessed incoming data. I have centered the images around zero with small standard deviation. (model.py lines 89)

Also I have cropped and kept only relevant part of the image for the training (cropped sky and unnecesary background). (model.py lines 90).

This preprocesed data was used as an input into the newtwork itroduced above. Input image size into normalization part is 160x320x3 and output is 45x320x3  

The nVidia network was then implemented without any changes to description above. (model.py lines 93-103)

#### 2. Attempts to reduce overfitting in the model

To prevent overfitting in the model, the model was trained and validated on different data sets to ensure that the model was not overfitting (code line 24-25). 20% of the data set was alocated for validation set. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I have tried to train the model over 30 epoch to see when the data starts to overfit and when I should stop the training. 

Figure below shows that the loss of validation set descreases only until epoch 10 (maybe only 6) and then it starts slowly rising. Due to that, I have selected final number of training epoch to 10.

![30 epoch][image1]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 106).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving in oposite direction, flipping the images. 

To capture good driving behavior, I first recorded four laps on track one using center lane driving. I have done two laps using mouse to steer and two laps using keyboard to steer. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it gets too close to the curb. These images show what a recovery looks like starting from top left to top right and bottom left:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would eliminate that the track has most of the turns to the left. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I have also recorded one lap driving in the opposite direction.

After the collection process, I had 33 066 (plus 33 066 flipped) data points. 

I randomly shuffled the data set and put 20% of the data into a validation set. 

I then preprocessed this data in normalization layer mentioned above.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by image above. I used an adam optimizer so that manually training the learning rate wasn't necessary.


### Conclusion

The driving behaviour of the vehicle is sufficient to get around the track without any collision. There is poor driving behaviour on straight parts of the road. I expect that it was caused by collecting data where there weren't many parts when the car went just straight.   