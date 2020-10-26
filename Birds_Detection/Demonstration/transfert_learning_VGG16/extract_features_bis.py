#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 08:48:29 2020

@author: marcpozzo
"""

#Rajouter preprocess input
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

import ast



#Paramètres par défaut
data_path="../../../.."
Mat_path="../../Materiels/"
path_folder=data_path+"DonneesPI/"
fichierClasses= Mat_path+"Table_Labels_to_Class.csv" # overwritten by --classes myFile
frame=pd.read_csv(fichierClasses,index_col=False) # table of species into classes 



#Transform labels to new one already known
def to_reference_labels (df,class_colum,frame):

    #flatten list in Labels_File
    cat=[]
    for i in range(len(frame["categories"]) ):
        cat.append( frame["categories"][i] )

    liste = [ast.literal_eval(item) for item in cat]

    # set nouvelle_classe to be the "unified" class name
    for j in range(len(frame["categories"])):
        #classesToReplace = frame["categories"][j].split(",")[0][2:-1]
        className = frame["categories"][j].split(",")[0][2:-1]
        #df["nouvelle_classe"]=df["classe"].replace(classesToReplace,className)
        df[class_colum]=df[class_colum].replace(liste[j],className)

    return df




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
            image_r=cv2.resize(im_caugh, (224, 224))
            if image_r is not None:
                images.append(image_r)
                
    images=np.array(images)       
    return images


#Select only animals categories
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=pd.read_csv(Mat_path+"imagettes.csv")
imagettes=to_reference_labels (imagettes,"classe",frame)
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 
liste_folders=['./DonneesPI/timeLapsePhotos_Pi1_4','./DonneesPI/timeLapsePhotos_Pi1_3','./DonneesPI/timeLapsePhotos_Pi1_2','./DonneesPI/timeLapsePhotos_Pi1_1','./DonneesPI/timeLapsePhotos_Pi1_0']
imagettes=imagettes[imagettes["path"].isin(liste_folders)]
images=open_imagettes(imagettes)




# load the VGG16 network and initialize the label encoder
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)
le = None


      
       
images_db=images[:2]
features = model.predict(images_db)
nb_features=7 * 7 * 512
features_re = features.reshape((features.shape[0],nb_features ))    
       

column_names = ["f_"+str(i) for i in range(nb_features)]
tableau_features = pd.DataFrame(columns = column_names)
for  vec in  features_re:
    tableau_features.loc[len(tableau_features)] = vec






