#!/usr/bin/env python3
# 
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:45:28 2020

@author: pi
"""

# Isolate a small part of the picture and extract the imagette in this area

from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
import functions as fn
from keras.models import Model, load_model
from skimage.measure import compare_ssim
from imutils import grab_contours
import cv2

#import os
import pandas as pd
import joblib
import time
#from keras.applications.vgg16 import preprocess_input
import numpy as np


output_Images_path="/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/"



#Paramètres à choisir
methode="Marc"#ou Marc "Khalid"


nb_classe_finale=6
neurone_features='6c_rob'# ou '6c_rob'ou'model.h9'#'model.h2_court'
maxAnalDL=-1 #if maxAnalDL=-1 alle images
type_photo="exterieur"#ou "interieur si on veut celle du bureau ou" exterieur pour champs"
filtre_type="light" #ou "ssim"ou"light"
name="zoom_0.9:1.3_flip"


#Autres Paramètres
path="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/6classes_zoom/"
neurone_features=path+name
c3poFolder="/mnt/VegaSlowDataDisk/c3po_interface/"
#filtre_RL = joblib.load(c3poFolder+"bin/output/RL_annotation_model")
#Model1 = joblib.load(c3poFolder+"bin/output/model.cpickle")
filtre_RL = joblib.load(c3poFolder+"bin/output/RL_annotation_model")
Model1 = joblib.load(c3poFolder+"bin/output/model.cpickle")

table_labels="testingInputs/conversion_cat_generator.csv"
conv_labels=pd.read_csv(table_labels)





coef_filtre_RQ=pd.read_csv("testingInputs/coefs_filtre_RQ.csv")
coef=coef_filtre_RQ
#table=pd.read_csv("table_EK000228.JPG.csv")
height=2448
width=3264



x_pix_min=1
y_pix_min=1
x_pix_max=350
y_pix_max=500





if type_photo=="exterieur":
    name2 = "EK000228.JPG"
    imageA = cv2.imread("testingInputs/EK000227.JPG")
    imageB = cv2.imread("testingInputs/"+name2)
 

if type_photo=="interieur":
    name2="image_2020-03-03_09-53-14.JPG"
    imageA = cv2.imread("testingInputs/image_2020-03-03_09-52-49.JPG")
    imageB = cv2.imread("testingInputs/"+name2)





if nb_classe_finale==6:
    conv_labels=conv_labels[conv_labels["str_cat"]!="tracteur"]
    conv_labels=conv_labels[conv_labels["str_cat"]!="voiture"]
    conv_labels=conv_labels[conv_labels["str_cat"]!="oiseau"]
    
    
if nb_classe_finale==2:
   conv_labels= conv_labels[(conv_labels["str_cat"]=="autre") | (conv_labels["str_cat"]=="oiseau")]  
labels=conv_labels["str_cat"].unique()


#start_time = time.time()





blurFact = 25
a=cv2.GaussianBlur(imageA,(blurFact,blurFact),sigmaX=0)
b=cv2.GaussianBlur(imageB,(blurFact,blurFact),sigmaX=0)


# Diff et filtrage
if filtre_type=="ssim":
    (score, diff) = compare_ssim(a, b, full=True,multichannel=True)
    diff = (diff * 255).astype("uint8")
    cv2.imwrite("testingInputs/diff.jpg", diff)
    blur = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("testingInputs/diffblur.jpg", blur)
    #                    cv2.imwrite("ablur.jpg",blur)
    thresh = cv2.threshold(src=blur, thresh= 210,maxval=255,type=cv2.THRESH_BINARY_INV)[1]
    cv2.imwrite("testingInputs/thresh.jpg", thresh)
    threshS = cv2.dilate(thresh,(3,3))
    threshS = cv2.erode(threshS,(3,3),iterations=1)
    cv2.imwrite("testingInputs/threshS.jpg", threshS)
        #    cv2.imwrite(subImagesDir2+"diffsTrAgg-"+timeStamp2+".JPG",threshS)





#faire une fonction et la mettre dans functions.py
if filtre_type=="light":
    img2 = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    img1 = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
    absDiff2 = cv2.absdiff(img1, img2)
    diff = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("testingInputs/diff.jpg", diff)
    th2 = cv2.adaptiveThreshold(src=diff,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
                                thresholdType=cv2.THRESH_BINARY,blockSize=221,C=-30) # adaptation de C à histogram de la photo ?

    cv2.imwrite("testingInputs/thresh2.jpg", th2)
    th2Blur=cv2.GaussianBlur(th2,(blurFact,blurFact),sigmaX=0)
    cv2.imwrite("testingInputs/thresh2Blur.jpg", th2Blur)
    th2BlurTh = cv2.adaptiveThreshold(src=th2Blur,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
            thresholdType=cv2.THRESH_BINARY,blockSize=121,C=-30) # adaptation de C à histogram de la photo ?
    threshS=th2BlurTh
    cv2.imwrite("testingInputs/th2BlurTh.jpg", th2BlurTh)



        # defines corresponding regions of change
cnts = cv2.findContours(threshS.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = grab_contours(cnts)


table_add=pd.read_csv("testingInputs/oiseau_lab_Alex.csv")
coef_filtre=coef_filtre_RQ

#Neural_models=["zoom_0.9:1.3_flip","6c_rob","zoom_1.3","drop_out.50","z1.3"]
Neural_models=["z1.3"]

for name in Neural_models:
    print(" ")
    print("prediction avec le modèle: ",name)
    print(" ")
    neurone_features=path+name


    #Prediction modèle
    model = load_model(neurone_features,compile=False)
    CNNmodel = Model(inputs=model.input, outputs=model.layers[-1].output)


    if (name=="z1.3") or (name=="drop_out.50") or ( name=="zoom_0.9:1.3_flip")  or ( name=="6c_rob") :
    
        #intervalle=[200,300]
        intervalle=[1030,1041]
        zoom=1
        table,imageRectangles,batchImages_filtre,batchImages = fn.find_square_reverse(imageB,intervalle,zoom,
                                                                name2,cnts,
                                                                maxAnalDL, # can be set to -1 to analyse everything
                                                                CNNmodel,labels,filtre_RL,
                                                                x_pix_max,y_pix_max,x_pix_min,y_pix_min
                                                                ,coef_filtre=coef_filtre_RQ)
    

        #"cv2.imwrite("Output_images/"+name+".jpg", imageRectangles)"
        cv2.imwrite("Output_images/findsquaress.jpg", imageRectangles)
    
    table_resultat=table.iloc[:,7:]
    arr_resultat=table_resultat.values
    arg_resulat=arr_resultat.argmax(axis=1)
    for i in range(len(table_resultat.columns)):
        print(table_resultat.columns[i], list(arg_resulat.flatten()).count(i) )
    print("nombre total d'imagettes : ", len(arg_resulat))
    cv2.imwrite(output_Images_path+name+".jpg", imageRectangles)

#On va selectionner le modèle z1.3 qui est le meilleur pour cette image et on vérifie 
#qu'on retombe sur ce résultat.
#array([5, 0, 5, 0, 5, 0, 0, 0, 5, 0, 5, 0, 0, 0, 5, 0, 5, 0, 5, 0, 5, 5,
#        5, 0, 5, 0])]




#Selectionne les images qui ont le plus de chance d'être des imagettes d'oiseaux 
#A integrer avant le RN avant de tourner sur le pi













#Afficher les résultats trouvés




#print("Temps d execution : %s secondes ---" % (time.time() - start_time))

if methode=="Khalid":
    print(table_resultat)

