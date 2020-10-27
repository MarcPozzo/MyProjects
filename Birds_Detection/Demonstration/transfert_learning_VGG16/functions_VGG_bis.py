#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:36:31 2020

@author: marcpozzo
"""
import ast
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split





def open_imagettes(images,liste_name_test,data_path):


    images_list=[]
    imagettes_copy=images.copy()
    
    i=0
    for name_test in liste_name_test:
        i+=1
        One_image=imagettes_copy[imagettes_copy["filename"]==name_test]
        big_image_path=data_path+name_test
        big_image=cv2.imread(big_image_path)
        if big_image is None:
            print(name_test,i)
        
        for i in range(len(One_image)):
            imagette=One_image.iloc[i]
            xmin, ymin, xmax,ymax=imagette[['xmin', 'ymin', 'xmax','ymax']]
            im_caugh=big_image[ymin:ymax,xmin:xmax]
            im_caugh=tf.keras.preprocessing.image.img_to_array(im_caugh)

            image_r=cv2.resize(im_caugh, (224, 224))
            if image_r is not None:
                images_list.append(image_r)
                
    images=np.array(images_list)       
    return images



def get_features_to_df(images,model,liste_name_test,data_path,bird=True):
    
    

    images=open_imagettes(images,liste_name_test,data_path)
    list_birds=["corneille","faisan","pigeon","oiseau"]
    if bird==True:
        label=1
        images=images[images["classe"].isin(list_birds)]
    if bird==False:
        label=0
        images=images[images["classe"].isin(list_birds)==False]
        
    
    features = model.predict(images)
    nb_features=7 * 7 * 512
    features_re = features.reshape((features.shape[0],nb_features )) 
    features_name = ["f_"+str(i) for i in range(nb_features)]
    column_names=features_name+["Classe"]
    tableau_features = pd.DataFrame(columns = column_names)
    for  vec in  features_re:
        array=np.append(vec,label)
        tableau_features.loc[len(tableau_features)] = array
        
    return tableau_features



def get_features_to_df_bis(images_df,model,liste_name_test,data_path,bird=True):
    
    #parameters
    NB_FEATURES=7 * 7 * 512 #Number of output in the fold
    list_birds=["corneille","faisan","pigeon","oiseau"] #classes to keep
    
    #Get output of VGG16
    images=open_imagettes(images_df,liste_name_test,data_path)
    features = model.predict(images)
    features_resize = features.reshape((features.shape[0],NB_FEATURES )) 
    features_name = ["f_"+str(i) for i in range(NB_FEATURES)]
   
    
    #Gather features and label in a dataframe
    if bird==True:
        label=1
        images_df=images_df[images_df["classe"].isin(list_birds)]
    if bird==False:
        label=0
        images_df=images_df[images_df["classe"].isin(list_birds)==False]
    column_names=features_name+["classe"]
    tableau_features = pd.DataFrame(columns = column_names) 
    for  vec in  features_resize:
        array=np.append(vec,label)
        tableau_features.loc[len(tableau_features)] = array
        
    return tableau_features



def get_tables(imagettes,model,liste_imagettes,data_path):
    tableau_birds_features=get_features_to_df_bis(imagettes,model,liste_imagettes,data_path,bird=True)
    tableau_other_features=get_features_to_df_bis(imagettes,model,liste_imagettes,data_path,bird=False)
    tableaux=[tableau_birds_features,tableau_other_features]
    tableau_features=pd.concat(tableaux)
    tableau_features=tableau_features.sample(frac=1).reset_index(drop=True)
    return tableau_features








