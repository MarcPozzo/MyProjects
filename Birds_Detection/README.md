#Goal

- The medium term objective is to propose an open source project proposing various methods of image recognition, from training to assessment applicable on any set of images. 
- This portfolio is based on the techniques used for the bird recognition project conducted by INRAE.

In the meantime, I suggest you show you the code and the project specific to bird recognition.

# Introduction

The puropose is to detect a maximum of birds in fields.
Farmers loose a part of their harverst because birds eat seeds in their fields. That is why we want to detect them in order to make them run away.
In this regard Custom Convolutional Neural Networks such as Lenet, VGG16 or YoloV2 was used.  

# Instructions for a quick try :
A test with YoloV2 is now available for a small sample of pictures. If you want to test it, make sure the libraries are installed with the required version (see below) :
- tensorflow=1.13.1
- Keras=2.2.4
- openCV=3.4.2

If Requirements are satisfied, then type :
- cd Birds_Detection/Demonstration
- python3 inference.py


# Strategies and methods used


2 methods types of method can be used :
1) Manually extract objects by images' differences and then analyze extracted objects with neural network (for exemple VGG16, Lenet).
2) Use specialized neural network which automaticaly extract objects and analyze them ( for exemple Yolo, RCNN, ...).  

## Manual extraction

For an image when a bird has just appeared, if you make the substraction with this image and the previous one (when there was still no bird), an important differences should be noted in the area of the bird appearence. 
However, several small differences may also appear in other areas of the image if there are, for example, new shadows that have formed (see the image below):
![imageRectanglesTest_light](https://user-images.githubusercontent.com/30336936/95189697-7c992200-07ce-11eb-9201-d5c96e27b020.jpg)


These square areas are extracted and then should be analyzed.
Below you can see in green the square areas predicted by Lenet CNN as birds and in blue the square areas not predicted as a bird.


![FO](https://user-images.githubusercontent.com/30336936/94801959-1fc0f480-03e7-11eb-9986-534e52c07f3a.jpg)

For our use case we only need to detect birds in fields, the rest of the picture is not useful this is why I set a mask.


Before setting the mask
![pic3_mask](https://user-images.githubusercontent.com/30336936/95606479-ad8d8700-0a5a-11eb-9aaf-8e0574aef498.jpg)


After setting the mask
![pic3_wout_mask](https://user-images.githubusercontent.com/30336936/95606657-ed546e80-0a5a-11eb-82e3-8c83413b10c7.jpg)

## Model with automatic extraction

Now we are using YoloV2. Birds are represented by blue squares and a rabit is represented with green squares. The number above the squares represents the percentage of certainty of predicting the obtained class.

![Yolo_detection](https://user-images.githubusercontent.com/30336936/95216255-2342ea00-07f2-11eb-893b-e65cda60e1b1.png)

We can notice that well trained YoloV2 can detect birds and other animals that a human can barely find with his eyes (see below the original picture).
![byolo](https://user-images.githubusercontent.com/30336936/95218793-f47a4300-07f4-11eb-82ed-b2a380168ef5.jpg)


# Description of folders 

- Demonstration : folder containing a summary of the steps taken in the other files
- Matériels : data ressources such as tables, images, CNNs saved ... 
- Archives : old scripts not yet usable, searching and testing new lines of approach






