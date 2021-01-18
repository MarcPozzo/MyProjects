#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:18:33 2020

@author: marcpozzo
"""
#After having trained Neural Networks on tiny images, algos are tested on entire images here.
#Please select a range of variable similar to the best set you found with the script Extraction_Evaluation.py



#Import Librarires
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import functions_Lenet_VGG as fn
import numpy as np



#Parameters to choose 
neurone_name="Models/drop_out.50"
parameters_=[("bird_large","light",25,50),("bird_large","light",25,-1),(17,-8,53,-1)] #Parameters of diff used in the loop
targets_=['corneille', 'faisan', 'pigeon']



#Paramètres par défaut
data_path='../../../../Pic_dataset/'
Mat_path="../../Materiels/"
neurone_feature=Mat_path+neurone_name
CNNmodel  = load_model(neurone_feature,compile=False)
Images=pd.read_csv(Mat_path+"images.csv")
dict_anotation_index_to_classe=np.load(Mat_path+"dic_labels_indices.npy",allow_pickle='TRUE').item() 
Images=Images[Images["classe"].isin( targets_ )]


no_targets_=list( set(list(dict_anotation_index_to_classe.keys()))-set(targets_))
defaults_indices_=[]
for el in no_targets_:
    defaults_indices_.append(dict_anotation_index_to_classe[el])


#Gather and sort the names of all picture (with and without bird ) in a list
images_birds_=list(Images["filename"].unique()) # only images containig birds
images_=fn.order_images(data_path)




#loop apply in a range of pic and in a range of pictures to evaluate the number of True Positif and False Positf and save results in list
for parameter in parameters_:    
    blurFact,contrast,blockSize,maxAnalDL=parameter
    base_name=str(blurFact)+"-"+str(contrast)+"-"+str(blockSize)+"-"+str(maxAnalDL)+".txt"
    (NB_FP,NB_TP)=(0,0)
    
    for name_test in images_birds_[:2]:                    
        index_of_ref=images_.index(name_test)-1
        name_ref=images_[index_of_ref]
        print(name_test,name_ref)
        TP,FP=fn.Evaluate_Lenet_prediction_bis ( Images , name_test , name_ref  , CNNmodel ,data_path, dict_anotation_index_to_classe, defaults_indices_,contrast, blockSize, blurFact )  
        NB_TP+= TP
        NB_FP+=FP
 
       
           
    """            
    with open(Mat_path+"FP-"+base_name, "wb") as fp:   #Pickling
            pickle.dump(nb_FP_ds_, fp)
                    
    with open(Mat_path+"TP-"+base_name, "wb") as fp:   #Pickling
            pickle.dump(nb_TP_ds_, fp)
    """
                
