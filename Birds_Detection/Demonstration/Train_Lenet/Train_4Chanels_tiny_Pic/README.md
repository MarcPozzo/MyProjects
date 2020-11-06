This folder gathers models train with Lenet architectures.

get_4C_chanel.py : script to generate a 4th chanels from 'anoted tiny images from GBR or HSV method' (for Train_im_anote_HSV,Train_im_anote_GBR )
Train_im_anote_HSV,Train_im_anote_GBR : train with Lenet on "annoted tiny images" for whom add a 4chanel






archive : contains the fitting script with the False Negatif (doesn't ready now)
aggregate_4C.py: aggregate a 4th Chanel for false positive and animal caught by the system
Train_4C_image_caught: Train a 4th chanel Lenet model on anoted images, false positives and animals caught