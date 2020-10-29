#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:18:33 2020

@author: marcpozzo
"""
#Ici on compare une image avec sa référence ...


import pandas as pd
import functions_Lenet_VGG_work_in_progress as fn
import os
from os.path import basename, join
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import load_model
import pickle
#from os import chdir




#Paramètres par défaut
data_path='../../../../Pic_dataset/'
Mat_path="../../Materiels/"
neurone_feature=Mat_path+"Models/drop_out.50"
CNNmodel  = load_model(neurone_feature,compile=False)
Images=pd.read_csv(Mat_path+"images.csv")
Imagettess=pd.read_csv(Mat_path+"imagettes.csv")


#imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 




Diff_image_FP_by_folder,Diff_image_animals_by_folder,Diff_image_image_total_by_folder,nb_FP_liste,nb_TP_liste,nb_FN1_liste,nb_FN2_liste=[[] for i in range(7)]

#bird_prob
liste_parameter=[("bird_large","light",25,50),("bird_large","light",25,-1),("bird_large","light",17,20)]

#Gather and sort the names of all picture (with and without bird ) in a list
images_=[]
for r, d, f in os.walk(data_path):
    for file in f:
        if '.jpg' in file:
            images_.append(basename(join(r, file)))                       
images_.sort()   

#Gather the names of picture with birds
images_birds_=list(Images["filename"].unique()) # images with birds
(nb_folder_TP,nb_folder_FP)=(0,0)

for parameter in liste_parameter:
    focus,method,blockSize,maxAnalDL=parameter
    base_name=focus+"-"+method+"-"+str(blockSize)+"-"+str(maxAnalDL)+".txt"
    nb_FP_liste=[]
    nb_TP_liste=[]
    
     
for name_test in images_birds_:
                   
    index_of_ref=images_.index(name_test)-1
    name_ref=images_[index_of_ref]
    print(name_test,name_ref)
    imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates,liste_Diff_birds,nb_oiseaux=fn.Lenet_prediction(name_test,name_ref,folder,CNNmodel,
                                                                                                                                                       blockSize=blockSize,thresh=0.5,
                                                                                                                                                       blurFact=17,chanels=3,contrast=-8,
                                                                                                                                                       maxAnalDL=maxAnalDL,method=method,
                                                                                                                                                       mask=True,focus=focus)
                                                                                                                        
                                                                                                                        
           
    nb_TP_birds=len(TP_birds)
    nb_FP=len(FP)
    nb_TP_thresh=len(TP_estimates)
    nb_FP_thresh=len(FP_estimates)
     
    nb_folder_TP+= nb_TP_birds
    nb_folder_FP+=nb_FP

       
    nb_FP_liste.append(nb_folder_FP)
    nb_TP_liste.append(nb_folder_TP)
   
       
            
    with open(Mat_path+"FP-"+base_name, "wb") as fp:   #Pickling
        pickle.dump(nb_FP_liste, fp)
                
    with open(Mat_path+"TP-"+base_name, "wb") as fp:   #Pickling
        pickle.dump(nb_TP_liste, fp)
                
            
