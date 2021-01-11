#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:18:33 2020

@author: marcpozzo
"""
#Ici on compare une image avec sa référence ...


#Maintenant ce qui se passe c'est qu'il faut vérifier ce que c'est FP et FP_estimates ... . Les renommer peut être... . 
#Mais est-ce qu'on a vraiment besoin des deux ... ? 



import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import functions_Lenet_VGG as fn
#from os import chdir



#Paramètres par défaut
data_path='../../../../Pic_dataset/'
Mat_path="../../Materiels/"
neurone_feature=Mat_path+"Models/drop_out.50"
CNNmodel  = load_model(neurone_feature,compile=False)
Images=pd.read_csv(Mat_path+"images.csv")
Images=Images[Images["classe"].isin(['corneille', 'faisan', 'lapin','chevreuil', 'pigeon'])]
parameters_=[("bird_large","light",25,50),("bird_large","light",25,-1),("bird_large","light",17,-1)]
#Initialization
#Diff_image_FP_by_folder,Diff_image_animals_by_folder,Diff_image_image_total_by_folder,nb_FP_liste,nb_TP_liste,nb_FN1_liste,nb_FN2_liste=[[] for i in range(7)]




#Gather the names of picture with birds
images_birds_=list(Images["filename"].unique()) # only images containig birds


#Gather and sort the names of all picture (with and without bird ) in a list
images_=fn.order_images(data_path)




#loop apply in a range of pic and in a range of pictures to evaluate the number of True Positif and False Positf and save results in list
for parameter in parameters_:
    focus,diff_mod3C,blockSize,maxAnalDL=parameter
    base_name=focus+"-"+diff_mod3C+"-"+str(blockSize)+"-"+str(maxAnalDL)+".txt"
    (NB_FP,NB_TP)=(0,0)
    for name_test in images_birds_[:2]:                    
        index_of_ref=images_.index(name_test)-1
        name_ref=images_[index_of_ref]
        print(name_test,name_ref)
        TP,FP=fn.Evaluate_Lenet_prediction_bis ( Images , name_test , name_ref  , CNNmodel ,data_path,index=True,blurFact=17,contrast=-8 )  
        NB_TP+= TP
        NB_FP+=FP
 
       
           
    """            
    with open(Mat_path+"FP-"+base_name, "wb") as fp:   #Pickling
            pickle.dump(nb_FP_ds_, fp)
                    
    with open(Mat_path+"TP-"+base_name, "wb") as fp:   #Pickling
            pickle.dump(nb_TP_ds_, fp)
    """
                
