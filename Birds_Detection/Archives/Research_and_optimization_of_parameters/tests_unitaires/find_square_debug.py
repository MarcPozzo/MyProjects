#!/usr/bin/env python3
# 
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:45:28 2020

@author: pi




"""
#Il faudra comparer les derniers élements de batch_filtre avec les images de test_unit fleche batchImages
from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
# Isolate a small part of the picture and extract the imagette in this area

#from os import chdir
#chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
from keras.applications.vgg16 import preprocess_input
import functions as fn
from keras.models import Model, load_model
from skimage.measure import compare_ssim
from imutils import grab_contours
import cv2
import os
#import os
import pandas as pd
import joblib
import time
#from keras.applications.vgg16 import preprocess_input
import numpy as np


output_Images_path="/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/"



#Paramètres à choisir
maxAnalDL=-1
height=2448
width=3264



x_pix_min=9
y_pix_min=7
x_pix_max=100
y_pix_max=100




#Autres Paramètres
path="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/6classes_zoom/"

c3poFolder="/mnt/VegaSlowDataDisk/c3po_interface/"
filtre_RL = joblib.load(c3poFolder+"bin/output/RL_annotation_model")
#Model1 = joblib.load(c3poFolder+"bin/output/model.cpickle")
filtre_RL = joblib.load(c3poFolder+"bin/output/RL_annotation_model")
Model1 = joblib.load(c3poFolder+"bin/output/model.cpickle")

table_labels="testingInputs/conversion_cat_generator.csv"
conv_labels=pd.read_csv(table_labels)





coef_filtre_RQ=pd.read_csv("testingInputs/coefs_filtre_RQ.csv")
coef=coef_filtre_RQ
#table=pd.read_csv("table_EK000228.JPG.csv")



name2 = "EK000228.JPG"
imageA = cv2.imread("testingInputs/EK000227.JPG")
imageB = cv2.imread("testingInputs/"+name2)
 






conv_labels=conv_labels[conv_labels["str_cat"]!="tracteur"]
conv_labels=conv_labels[conv_labels["str_cat"]!="voiture"]
conv_labels=conv_labels[conv_labels["str_cat"]!="oiseau"]
labels=conv_labels["str_cat"].unique()






cnts=fn.filtre_light(imageA,imageB)
#Jusqu'à là
#path_anotation="testingInputs/oiseau_lab_Alex.csv"
path_anotation="empty"
#table_add=pd.read_csv("testingInputs/oiseau_lab_Alex.csv")
coef_filtre=coef_filtre_RQ

#Neural_models=["zoom_0.9:1.3_flip","6c_rob","zoom_1.3","drop_out.50","z1.3"]
name_model="z1.3"
neurone_features=path+name_model


    #Prediction modèle
model = load_model(neurone_features,compile=False)
CNNmodel = Model(inputs=model.input, outputs=model.layers[-1].output)


    
#intervalle=[200,300]
intervalle=[0,1070]
zoom=10
"""table,imageRectangles,batchImages_filtre,batchImages = fn.find_square(imageB,intervalle,zoom,
                                                            name2,cnts,
                                                            maxAnalDL, # can be set to -1 to analyse everything
                                                            CNNmodel,labels,filtre_RL,
                                                            x_pix_max,y_pix_max,x_pix_min,y_pix_min
                                                            ,coef_filtre=coef_filtre_RQ)"""



#On va comparer les images de ici et probablement test unitaires
batchImages = []
liste_table = []
imageSize= 28



    

#    np.empty((1,5), dtype = "int")  

#A coller ici
#





#On récupère les coordonnées des pixels différent par différence
for ic in range(0,len(cnts)):
    

    (x, y, w, h) = cv2.boundingRect(cnts[ic])
    name = (os.path.split(name2)[-1]).split(".")[0]
    name = name + "_" + str(ic) + ".JPG"
    f = pd.Series(dtype= "float64")
    f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   

        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif
    if( (f.xmax-f.xmin)<x_pix_max and (f.ymax-f.ymin)<y_pix_max # birds should less than 500 pixels wide and 350 high
       and (f.xmax-f.xmin)>x_pix_min and (f.ymax-f.ymin)>y_pix_min): # according to distribution in annotations
        subI, o, d, imageRectangles = fn.GetSquareSubset(imageB,f,verbose=False)
        subI = fn.RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        #subI = np.expand_dims(subI, axis=0)
        # subI = preprocess_input(subI)
        batchImages.append(subI)
            
        liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))





if path_anotation!="empty":
    table_add=pd.read_csv(path_anotation)
    annontation_reduit=(table_add.iloc[:,6:12]).drop("index",axis=1)
    for i in range(len(annontation_reduit)):
        name = (os.path.split(name2)[-1]).split(".")[0]
        name = name + "_" + str(i) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = annontation_reduit["xmin"].iloc[i], annontation_reduit["xmax"].iloc[i], annontation_reduit["ymin"].iloc[i], annontation_reduit["ymax"].iloc[i]
        subI, o, d, imageRectangles = fn.GetSquareSubset(imageB,f,verbose=False)
        subI = fn.RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        #subI = np.expand_dims(subI, axis=0)
        batchImages.append(subI)
    
        liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))







####




#dimension 2651,5      
table = pd.DataFrame(np.vstack(liste_table))
table.iloc[:,1:]=table.iloc[:,1:].astype(int)

if path_anotation=="empty":
    batchImages_stack = np.vstack(batchImages)
    batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
    table,index_possible_birds=fn.filtre_quantile(table,coef_filtre,height=2448,width=3264)
    table["possible_bird"]=filtre_RL.predict(np.array(table.iloc[:,1:]))
    table=(table[table["possible_bird"]=="O"])
    p_bird=table.index
    table.drop("possible_bird",axis=1,inplace=True)     
    index_possible_birds=list(set(index_possible_birds).intersection(p_bird)) 
    batchImages_filtre = [batchImages_stack_reshape[i] for i in (index_possible_birds)]
    batchImages=batchImages_filtre
    
    

#table=table.head(26)
#A coller ici ! 
if (zoom!=1) and (path_anotation=="empty"):
    
    liste_xmax=[]
    liste_xmin=[]
    liste_ymax=[]
    liste_ymin=[]
    for i in range(len(table)):
        XMAX=table["xmax"].iloc[i]
        XMIN=table["xmin"].iloc[i]
        YMAX=table["ymax"].iloc[i]
        YMIN=table["ymin"].iloc[i] 
        
        #MAX=table["xmax"].iloc[i]=agrandissement(XMIN,XMAX,zoom)
        largeur_min,largeur_max=fn.agrandissement(XMIN,XMAX,zoom)
        largeur_min=int(round(largeur_min))
        largeur_max=int(round(largeur_max))
        liste_xmin.append(largeur_min)
        liste_xmax.append(largeur_max)
        
        profondeur_min,profondeur_max=fn.agrandissement(YMIN,YMAX,zoom)
        profondeur_min=int(round(profondeur_min))
        profondeur_max=int(round(profondeur_max))
        liste_ymin.append(profondeur_min)
        liste_ymax.append(profondeur_max)
        
        
        #On ajoute les éléments de la dataframe provenant des anotations de Alex
        liste_xmax=liste_xmax
        liste_xmin=liste_xmin
        liste_ymax=liste_ymax
        liste_ymin=liste_ymin
    
 
    tables=table.copy()
    table["xmax"]=liste_xmax
    table["xmin"]=liste_xmin
    table["ymax"]=liste_ymax
    table["ymin"]=liste_ymin


    batchImages= []   
    "On modifie cette partie en remplaçant les x w y h par les annotation de table, attetion au problème de taille"
    for ic in range(0,len(table)):
    

        
        name = (os.path.split(name2)[-1]).split(".")[0]
        name = name + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = table["xmin"].iloc[ic], table["xmax"].iloc[ic] , table["ymin"].iloc[ic] , table["ymax"].iloc[ic] 
   

        #Maintenant on va ajuster les carrés jusqu'a trouver un resultat positif

        subI, o, d, imageRectangles = fn.GetSquareSubset(imageB,f,verbose=False)
        subI = fn.RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)



#features=preprocess_input(np.array(batchImages_stack))

#features=features.reshape((-1, 28,28,3))
batchImages=np.vstack(batchImages)
estimates = CNNmodel.predict(batchImages.reshape(-1,28,28,3))
#estimates = CNNmodel.predict(batchImages_stack_reshape)
if path_anotation!="empty":
    l_estimates=len(estimates)
    estimates=estimates[l_estimates-25:l_estimates]
    table=table.tail(25)
    table = table.rename(columns={0: 'imagettename', 1: 'xmin', 2: 'xmax', 3: 'ymin', 4: 'ymax'})
    table.iloc[:,1:5]=(table.iloc[:,1:5]).astype(int)
    arg_resulat_square=estimates.argmax(axis=1)
    print(arg_resulat_square)
#estimates_red=estimates[0:25]


for i in labels:
    table[i]=0
        
        
        
    
colonne=0    
for categorie in labels:
    for l in range(len(table)):
        table[categorie].iloc[l]=estimates[l,colonne]
    colonne+=1
#A supprimer ensuite    
#subI, o, d, imageRectangles = fn.GetSquareSubset(imageB,f,verbose=False)        
for i in range(len(table)):   
    xmin=int(table.iloc[i,1])
    xmax=int(table.iloc[i,2])
    ymin=int(table.iloc[i,3])
    ymax=int(table.iloc[i,4])
    #cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2) 
    max_col=np.amax(table.iloc[i,5:],axis=0)
    #if table.iloc[i,5]>=max_col:
    #if (i>len(table)-50) or( (i>1000) and (i<1500)) :
    if  (i>intervalle[0]) and (i<intervalle[1])  :

        #cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)
        if table.iloc[i,10]==max_col:
                cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2) 
        elif table.iloc[i,5]==max_col:
                cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 255, 255), 2) 
            
        else:
                cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 


cv2.imwrite("Output_images/ZOOM"+str(zoom)+".jpg",imageRectangles)










