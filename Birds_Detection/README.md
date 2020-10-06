The puropose is to detect a maximum of birds in fields with a raspberrypi. This raspberrypi is connected to a camera taking a new picture every 12 secondes. 
Birds eat seeds in fields and farmers loose a part of their harverst, that is why we want to detect them in order to make them fear. 
Custom Convolutional Neural Network was used such as Lenet or YoloV2.




2 methods types of method can be used :
- 1) Manually extract objects by images differences and then analyse extracted objects with neural network (for exemple VGG16, Lenet, ...).
- 2) Use specialzed neural network which automaticaly extract objects and analyze them ( for exemple Yolo, RCNN, ...).  

1) Manually extraction

For a given image when a birds has just appeared, if you make the substraction with this one and the previous one when there was still no bird, an important difference will appear in the area  of the bird. 
Several smaller difference can appear in other area of the picture if there are new shadows for exemple. In the below picture the square represent difference with previous image:
![imageRectanglesTest_light](https://user-images.githubusercontent.com/30336936/95189697-7c992200-07ce-11eb-9201-d5c96e27b020.jpg)


These square areas are extracted and consider as to be analalized.
Below you can see in green the square predict as birds and in blue the are area not predict as a bird.





Below an image of Birds detected (green squares) by Lenet CNN

![FO](https://user-images.githubusercontent.com/30336936/94801959-1fc0f480-03e7-11eb-9986-534e52c07f3a.jpg)


Model_Results
Run VGG16, Lenet or Yolov2 and displays the number of birds find and the number of False Positive

Train : Train Different version of custom Lenet neural networks 

bin : script used by raspberry pi

Matériels : data ressources such as tables, images, CNNs saved ... 

Research_and_optimization_of_parameters : search new parameter such as new classes, and try to find the best parameters for difference images 

Positionnement : script to switch positions in photo view to position in map view
Parameter_GPU : first explorations to make the GPU work




