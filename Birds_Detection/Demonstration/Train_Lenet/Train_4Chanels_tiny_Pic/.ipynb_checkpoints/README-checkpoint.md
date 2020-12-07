# Training and extraction of tiny images of 4 chanels with Lenet Neural Network

## Introduction

- Whole images may contain small objects to be identified, the objective of this section is to extract these objects to be identified into a database of small images
- The Lenet algorithm is then trained on this basis, then evaluated.


## Files description

- get_4C_tiny_images.py : script to generate a 4th chanels from 'anoted tiny images from GBR or HSV method' (for Train_im_anote_HSV,Train_im_anote_GBR )
- Train_im_anote_HSV,Train_im_anote_GBR : train with Lenet on "annoted tiny images" for whom add a 4chanel
- functions_4C.py This script gathers the functions used in the other script of this folder
- archive : contains the fitting script with the False Negatif (doesn't ready now)
    - aggregate_4C.py: aggregate a 4th Chanel for false positive and animal caught by the system
    - Train_4C_image_caught: Train a 4th chanel Lenet model on anoted images, false positives and animals caught
    - get_4C_tiny_images_alias.py is an allias of get_4C_tiny_images.py
    - get_4C_chanel.py this script is an  older allias of get_4C_tiny_images.py


## Requirement

- The objects must be annotated in a csv file according to their class and their location in the image. 
- Refer to the csv file to see the information that the DataFrame should contain.
- Your images are in a file outside the directory called 'Pic_dataset/' this folder must be at '../../../../Pic_dataset/'
- libraries to load : pandas=0.25.3, cv2=3.4.2


## Instruction

- Type python3 get_4C_tiny_images.py to generate tiny images of 4 dimensions eventually with a zoom or a dezoom. You have 2 options HSV or BGR (explained in Note section)
- Then train and evaluate your tiny images with Train_im_anote_BGR.ipynb or Train_im_anote_HSV.ipynb file in Jupyter according to the option you choose.



## Note

- What is the Difference between HSV mode and GBR?
    - In GBR mode : Difference of the 3 chanles in GBR and then convert them in Gray scale
    - In HSV mode : Difference of the 3 chanles in HSV mode and then convert themn in Gray scale
- Please feel free to change any parameters as your conviniance.





