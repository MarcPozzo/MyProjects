# Introduction

The puropose is to detect a maximum of birds in fields with a raspberrypi. This raspberrypi is connected to a camera taking a new picture every 12 secondes. 
Birds eat seeds in fields and farmers loose a part of their harverst, that is why we want to detect them in order to make them fear. 
Custom Convolutional Neural Network was used such as Lenet, VGG16 or YoloV2.

# Instruction for quick try :
A test with YoloV2 is now available for a small sample of pictures. If you want to test it, make sure the libraries are installed with the required version see above :
- tensorflow=1.13.1
- Keras=2.2.4
- openCV=3.4.2

If Requirements are satisfied then type :
- cd Birds_Detection/Demonstration
- python3 inference.py


#Strategies and methods useds


2 methods types of method can be used :
1) Manually extract objects by images differences and then analyse extracted objects with neural network (for exemple VGG16, Lenet, ...).
2) Use specialzed neural network which automaticaly extract objects and analyze them ( for exemple Yolo, RCNN, ...).  

1) Manually extraction

For a given image when a bird has just appeared, if you make the substraction with this image and the previous one (when there was still no bird), an important difference will appear in the area  of the bird. 
Several smaller difference can appear in other area of the picture if there are for exemple new shadows. In the below picture the square represent difference with previous image:
![imageRectanglesTest_light](https://user-images.githubusercontent.com/30336936/95189697-7c992200-07ce-11eb-9201-d5c96e27b020.jpg)


These square areas are extracted and consider as to be analalized.
Below you can see in green the square predict as birds and in blue the are area not predict as a bird.





Below an image of Birds detected (green squares) by Lenet CNN

![FO](https://user-images.githubusercontent.com/30336936/94801959-1fc0f480-03e7-11eb-9986-534e52c07f3a.jpg)

For our use we only need to detect birds in fields, the rest of the picture is not useful. 
We can notice that several objects are identified outside of the field. 
Altought in this case these objects aren't identified as birds, it could be a source of failure and this is not useful.
That is why we will set a mask to hide the unuseful part.
Below the image before and after to set the mask. 



Several objects are identified outside of the field

Before to set the mask
![pic3_mask](https://user-images.githubusercontent.com/30336936/95606479-ad8d8700-0a5a-11eb-9aaf-8e0574aef498.jpg)


After to set the mask
![pic3_wout_mask](https://user-images.githubusercontent.com/30336936/95606657-ed546e80-0a5a-11eb-82e3-8c83413b10c7.jpg)

2) Model with automatic extraction

Now we are using YoloV2. Birds are represented by blue squares and a rabit is represented with green square.

![Yolo_detection](https://user-images.githubusercontent.com/30336936/95216255-2342ea00-07f2-11eb-893b-e65cda60e1b1.png)

We can notice that YoloV2 well trained can detect birds and other animals that a human can barely find ( see below the original picture).
![byolo](https://user-images.githubusercontent.com/30336936/95218793-f47a4300-07f4-11eb-82ed-b2a380168ef5.jpg)




Model_Results
Run VGG16, Lenet or Yolov2 and displays the number of birds find and the number of False Positive

Train : Train Different version of custom Lenet neural networks 

bin : script used by raspberry pi

Matériels : data ressources such as tables, images, CNNs saved ... 

Research_and_optimization_of_parameters : search new parameter such as new classes, and try to find the best parameters for difference images 

Positionnement : script to switch positions in photo view to position in map view
Parameter_GPU : first explorations to make the GPU work




