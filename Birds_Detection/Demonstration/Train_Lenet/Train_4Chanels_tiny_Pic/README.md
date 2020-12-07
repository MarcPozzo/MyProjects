# Extract 4th Chanel Picture and train them

- get_4C_tiny_images.py : script to generate a 4th chanels from 'anoted tiny images from GBR or HSV method' (for Train_im_anote_HSV,Train_im_anote_GBR )
- Train_im_anote_HSV,Train_im_anote_GBR : train with Lenet on "annoted tiny images" for whom add a 4chanel
- functions_4C.py This script gathers the functions used in the other script of this folder
- archive : contains the fitting script with the False Negatif (doesn't ready now)
    - aggregate_4C.py: aggregate a 4th Chanel for false positive and animal caught by the system
    - Train_4C_image_caught: Train a 4th chanel Lenet model on anoted images, false positives and animals caught
    - get_4C_tiny_images_alias.py is an allias of get_4C_tiny_images.py
    - get_4C_chanel.py this script is an  older allias of get_4C_tiny_images.py


Difference between HSV mode and GBR.
In GBR mode : Difference of the 3 chanles in GBR and then convert them in Gray scale
In HSV mode : Difference of the 3 chanles in HSV mode and then convert themn in Gray scale