#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:42:58 2020

@author: marcpozzo
"""

#This script wants to find the beset para of filter
#See the influence of reference picture choosen

from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
import cv2
import matplotlib.pyplot as plt
import os
from imutils import grab_contours
import pandas as pd
#import functions as fn
import numpy as np

path_images="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/"




name_ref1="/image_2019-04-30_17-41-18.jpg"  #shape (1024, 1280, 3)

#Pour ce couple il n'y a pas d'imagettes énorme comme avant et avec l'oiseaux ça semble fonctionner
name_ref2="image_2019-04-30_18-16-57.jpg"   
name_test1="image_2019-04-30_18-17-14.jpg"  #shape (720, 1280, 3)



#Pour ces deux couples ça va aussi 
#Effectivement quand on prend test2 qui est plus éloigné de ref1 on tendance à avoir des images plus grandes
#Si on compare test1 à ref1 précédent on a des différences très grandes 
name_ref1="image_2019-04-30_18-37-16.jpg"
name_test1="image_2019-04-30_18-37-33.jpg"

name_test2="image_2019-04-30_18-30-02.jpg"

#Maintenant on compare deux images qui ont des oiseaux cela fonctionne bien à part que la bache reflete des grandes diff (comme le cas précédant)

name_test1="image_2019-04-30_18-40-19.jpg"
name_ref1="image_2019-04-30_18-40-36.jpg"


#trouver un autre couple pour voir comment ça marche

name2="toto"
imageA=cv2.imread(path_images+name_ref1)
imageB=cv2.imread(path_images+name_test1)
plt.imshow(imageA)
plt.imshow(imageB)
cv2.imwrite("filter_test/imageA.jpg",imageA)
cv2.imwrite("filter_test/imageB.jpg",imageB)

imageSize=28

def filtre_light(n_image,contrast=-5,blockSize=51,blurFact = 15):
    
    
    
    imageA=cv2.imread(path_images+liste_name_ref[n_image])
    imageB=cv2.imread(path_images+liste_name_test[n_image])
    imageC=imageB.copy
    liste_table=[]
    batchImages=[]
    
    img2 = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    cv2.imwrite("filter_test/img2.jpg",img2)
    img1 = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
    cv2.imwrite("filter_test/img1.jpg",img1)
    
    absDiff2 = cv2.absdiff(img1, img2)
    cv2.imwrite("filter_test/absDiff2.jpg",absDiff2)
    diff = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
    th2 = cv2.adaptiveThreshold(src=diff,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
                                thresholdType=cv2.THRESH_BINARY,blockSize=blockSize,C=contrast) #c=-30 pour la cam de chasse adaptation de C à histogram de la photo ?
    cv2.imwrite("filter_test/th2.jpg",th2)
    th2Blur=cv2.GaussianBlur(th2,(blurFact,blurFact),sigmaX=0)
    cv2.imwrite("filter_test/th2Blur.jpg",th2Blur)
    th2BlurTh = cv2.adaptiveThreshold(src=th2Blur,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
            thresholdType=cv2.THRESH_BINARY,blockSize=blockSize,C=contrast) # adaptation de C à histogram de la photo ?
    cv2.imwrite("filter_test/th2BlurTh.jpg",th2BlurTh)
    threshS=th2BlurTh

        # defines corresponding regions of change
    cnts = cv2.findContours(threshS.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    
    
    
    
    
    #On va essayer de mettre maintenant les carrés
    
    
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        name = (os.path.split(name2)[-1]).split(".")[0]
        name = name + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   

        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif

        subI, o, d, imageRectangles = fn.GetSquareSubset(imageB,f,verbose=False)
        subI = fn.RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))
     
    imageRectangle=imageRectangles.copy()
    table_full = pd.DataFrame(np.vstack(liste_table))
    
    
    #Ce serait bien de rajouter rename ici !!! Si ça n'entraine pas de bug
    table_full = table_full.rename(columns={0: 'imagettename', 1: 'xmin', 2: 'xmax', 3: 'ymin', 4: 'ymax'})
    table_full.iloc[:,1:]=table_full.iloc[:,1:].astype(int)

    
    
    for i in range(len(table_full)):   
        xmin=int(table_full.iloc[i,1])
        xmax=int(table_full.iloc[i,2])
        ymin=int(table_full.iloc[i,3])
        ymax=int(table_full.iloc[i,4])

        cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2) 

    
    cv2.imwrite("filter_test/threshS.jpg",imageRectangles)  
    return len(cnts)

#imageB=cv2.imread(path_images+name_test1)
#filtre_light(imageA,imageB,contrast=-5,blockSize=51)
cnts=filtre_light(imageA,imageB,contrast=-5,blockSize=51,blurFact = 25)



import os
from os.path import basename, join
chdir("/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/")
liste_image_ref = []
# r=root, d=directories, f = files
for r, d, f in os.walk("/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/"):
    for file in f:
        if '.jpg' in file:
            liste_image_ref.append(basename(join(r, file)))
            
            

path_images="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/"
imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
imagettes_PI_0=imagettes[(imagettes["path"]=="./DonneesPI/timeLapsePhotos_Pi1_0") ]

#Les seules imagettes qui nous intéressent  sont celles des oiseaux pas celle de la terrer
imagettes_PI_0=imagettes_PI_0[imagettes_PI_0["classe"]!="ground"]

liste_name_test=list(imagettes_PI_0["filename"].unique())
liste_name_ref=[]
for name_test in liste_name_test[0:30]:
    index_of_ref=liste_image_ref.index(name_test)-1
    name_ref=liste_image_ref[index_of_ref]
    liste_name_ref.append(name_ref)


"""
liste_name_test               
          Nb_oiseaux_predits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
          Nb_oiseaux_predits=[1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1, 4, 5, 5, 5, 3]
Nombre_imagette_oiseau_match=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 6, 7, 6, 5, 4]

"""





#54 56 oiseaux et 32 trouvées 
name_test1="image_2019-04-30_18-40-19.jpg"
name_ref1="image_2019-04-30_18-40-36.jpg"


#trouver un autre couple pour voir comment ça marche

name2="toto"
imageA=cv2.imread(path_images+name_ref1)
imageB=cv2.imread(path_images+name_test1)

imageA=cv2.imread(path_images+liste_name_ref[0])
imageB=cv2.imread(path_images+liste_name_test[0])



###Porcédons au test ici

cnts=filtre_light(2,contrast=-5,blockSize=51,blurFact = 25)