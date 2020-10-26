#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 08:48:29 2020

@author: marcpozzo
"""

from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import config
#from imutils import paths
import numpy as np
import pickle
import random
import os
import pandas as pd
import cv2

"""
imagettes=pd.read_csv(Mat_path+"imagettes.csv")
imagettes=imagettes[imagettes["filename"].isin(imagettes_to_keep)]
"""



#Paramètres par défaut
data_path="../../../.."
Mat_path="../../Materiels/"
path_folder=data_path+"DonneesPI/"
fichierClasses= Mat_path+"Table_Labels_to_Class.csv" # overwritten by --classes myFile
frame=pd.read_csv(fichierClasses,index_col=False) # table of species into classes 


#Select only animals categories
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=pd.read_csv(Mat_path+"imagettes.csv")
imagettes=to_reference_labels (imagettes,"classe",frame)
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 
liste_folders=['./DonneesPI/timeLapsePhotos_Pi1_4','./DonneesPI/timeLapsePhotos_Pi1_3','./DonneesPI/timeLapsePhotos_Pi1_2','./DonneesPI/timeLapsePhotos_Pi1_1','./DonneesPI/timeLapsePhotos_Pi1_0']
imagettes=imagettes[imagettes["path"].isin(liste_folders)]



# load the VGG16 network and initialize the label encoder
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)
le = None


      
       
       
       


def open_imagettes(imagettes):


    images=[]
    imagettes_copy=imagettes.copy()
    liste_name_test=list(imagettes["filename"].unique())
    
    for name_test in liste_name_test:
        One_image=imagettes_copy[imagettes_copy["filename"]==name_test]
        big_image_path=data_path+One_image["path"].iloc[0][1:]+"/"+name_test
        big_image=cv2.imread(big_image_path) 
        
        for i in range(len(One_image)):
            imagette=One_image.iloc[i]
            xmin, ymin, xmax,ymax=imagette[['xmin', 'ymin', 'xmax','ymax']]
            im_caugh=big_image[ymin:ymax,xmin:xmax]
            image_r=cv2.resize(big_image, (224, 224))
            if image is not None:
                images.append(image_r)
                
    images=np.array(images)       
    return images


images=open_imagettes(imagettes)



imagePath="/Users/marcpozzo/Documents/Projet_Git/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-17-14.jpg"

imagePath="/Users/marcpozzo/Documents/Projet_Git/DonneesPI/timeLapsePhotos_Pi1_2/"

image = load_img(imagePath, target_size=(224, 224))
		image = img_to_array(image)
			# preprocess the image by (1) expanding the dimensions and
			# (2) subtracting the mean RGB pixel intensity from the
			# ImageNet dataset
		image = np.expand_dims(image, axis=0)
		image = imagenet_utils.preprocess_input(image)

			# add the image to the batch
		batchImages.append(image)

		# pass the images through the network and use the outputs as
		# our actual features, then reshape the features into a
		# flattened volume
	batchImages = np.vstack(batchImages)
	features = model.predict(batchImages, batch_size=config.BATCH_SIZE)
    features = model.predict(image[:10])
	features = features.reshape((features.shape[0], 7 * 7 * 512))