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



#Parameters to choose (please refer you to the script Extraction_Evaluation.py to set blurFact,contrast,blockSize)
blurFact=17
contrast=-8
blockSize=53
neurone_name="Models/drop_out.50" #please refer you to ../train_Lenet/train_your_tiny_images.ipynb
targets_=['corneille', 'faisan', 'pigeon']#Targets_ are gathered if Precise==Fase (default Value) in Evaluate_Lenet_prediction



#Paramètres par défaut
data_path='../../../../Pic_dataset/'
Mat_path="../../Materiels/"
neurone_feature=Mat_path+neurone_name
CNNmodel  = load_model(neurone_feature,compile=False)
Images=pd.read_csv(Mat_path+"images.csv")
dict_anotation_index_to_classe=np.load(Mat_path+"dic_labels_indices.npy",allow_pickle='TRUE').item() 
Images=Images[Images["classe"].isin( targets_ )]
base_name="blurFact"+"_"+str(blurFact)+"_"+"contrast"+"_"+str(contrast)+"_"+"blockSize"+"_"+str(blockSize)+".csv"

no_targets_=list( set(list(dict_anotation_index_to_classe.keys()))-set(targets_))
ntarget_classes_=[]
for el in no_targets_:
    ntarget_classes_.append(dict_anotation_index_to_classe[el])


#Gather and sort the names of all picture (with and without bird ) in a list
images_birds_=list(Images["filename"].unique()) # only images containig birds
images_=fn.order_images(data_path)


thresholds_=[0.1,0.3,0.5,0.7,0.9]#thresholds_ is the probability threshold above the object can be identified as the target

#loop apply in a range of pic and in a range of pictures to evaluate the number of True Positif and False Positf and save results in list

nb_FP_ds_=[]
nb_TP_ds_=[]
base_name="Lenet_Evaluation.csv"
for thresh in thresholds_:    
    NB_TP=0
    NB_FP=0
    for name_test in images_birds_[:2]:                    
        index_of_ref=images_.index(name_test)-1
        name_ref=images_[index_of_ref]
        print(name_test,name_ref)
        TP,FP=fn.Evaluate_Lenet_prediction( Images , name_test , name_ref  , CNNmodel ,data_path, dict_anotation_index_to_classe, ntarget_classes_,contrast, blockSize, blurFact,thresh=thresh )  
        NB_TP+= TP
        NB_FP+=FP
    nb_FP_ds_.append(NB_FP)
    nb_TP_ds_.append(NB_TP)
    print("For these parameters the total number of TP is: ",NB_TP)
    print("For these parameters the total number of FP is: ",NB_FP)
    
 
       
           
                
with open("FP-"+base_name, "wb") as fp:   #Pickling
            pickle.dump(nb_FP_ds_, fp)
                    
with open("TP-"+base_name, "wb") as fp:   #Pickling
            pickle.dump(nb_TP_ds_, fp)
    
                
