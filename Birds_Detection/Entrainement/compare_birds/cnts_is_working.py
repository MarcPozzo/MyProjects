#!/usr/bin/env python3
# 
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:45:28 2020

@author: pi




"""
#Ce script propose de vérifier si les carrés d'annotations sont bien repérées par la différence puis après le ou les filtres sous forme de fonction avec un script sourçale et sunthétique

from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
#import functions as fn
exec(open("functions.py").read())
import cv2
import pandas as pd
import joblib
import matplotlib.pyplot as plt

output_Images_path="/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/"

#Paramètres à choisir

#Pour filtre quantile



filtre_choice="No_filtre" #"quantile_filtre"#"No_filtre" "RL_filtre"



#Autres Paramètres
path="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/6classes_zoom/"






#Avec les photos du débuts
name2 = "EK000228.JPG"
imageA = cv2.imread("testingInputs/EK000227.JPG")
imageB = cv2.imread("testingInputs/"+name2)
 
cnts=filtre_light(imageA,imageB)
print(len(cnts))


#Avec le même répertoire 



path_anotation="testingInputs/oiseau_lab_Alex.csv"

"""

a="/mnt/VegaSlowDataDisk/c3po_interface/bin/image_inputs/image_2019-04-30_18-17-14.jpg"
image_ref="/mnt/VegaSlowDataDisk/c3po_interface/bin/image_inputs/image_ref.jpg"
img1=cv2.imread(image_ref)
plt.imshow(img1)
img_test=cv2.imread(a)"""


imageA = cv2.imread("/mnt/VegaSlowDataDisk/c3po/Images_aquises/donneesSausse/2017_photos_grignon_non_protege/EK000226.JPG")
imageB = cv2.imread("/mnt/VegaSlowDataDisk/c3po/Images_aquises/donneesSausse/2017_photos_grignon_non_protege/EK000227.JPG")
cnts=filtre_light(imageA,imageB)
print(len(cnts))


place_generate_sqaure(imageB,cnts)

#Avec les photos du pi

#Dans le dossier1 du PI Ceux la moitié fonctionne


path_image_test="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-27-49.jpg"

path_image_ref="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-24-28.jpg"
cv2.imread(image_test)



imageA = cv2.imread(path_image_ref)
imageB = cv2.imread(path_image_test)

cv2.imread(image_ref)
cnts=filtre_light(imageA,imageB)
print(len(cnts))
place_generate_sqaure(imageB,cnts)

#Dans le dossier 3
path_image_test="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_3/image_2019-05-29_07-57-16.jpg"
path_image_ref="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_3/image_2019-05-29_07-56-59.jpg"

imageA = cv2.imread(path_image_ref)
imageB = cv2.imread(path_image_test)

cnts=filtre_light(imageA,imageB)
print(len(cnts))




place_generate_sqaure(imageB,cnts)









"""

path_imagettes="/mnt/VegaSlowDataDisk/c3po/DonneesPI/timeLapsePhotos_Pi1_0/"
name_test="image_2019-05-29_07-57-16.jpg"#"image_2019-05-29_07-57-16.jpg"#"image_2019-04-30_18-27-49.jpg"
#image_ref="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-24-28.jpg"
path_image_ref=path_imagettes+"image_2019-04-30_18-13-03.jpg"
path_image_test=path_imagettes+name_test
path_imaget_test=["/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_3/image_2019-05-29_07-57-16.jpg"]

"""

image_ref="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-24-28.jpg"
image_test="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-25-35.jpg"


imageA=cv2.imread(image_ref)
imageB=cv2.imread(image_test)


cnts=filtre_light(imageA,imageB)
print(len(cnts))




place_generate_sqaure(imageB,cnts)





imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")



donne=['./DonneesPI/timeLapsePhotos_Pi1_4',
       './DonneesPI/timeLapsePhotos_Pi1_3',
       './DonneesPI/timeLapsePhotos_Pi1_2',
       './DonneesPI/timeLapsePhotos_Pi1_1',
       './DonneesPI/timeLapsePhotos_Pi1_0']
    
a='./DonneesPI/timeLapsePhotos_Pi1_0' 
b='./DonneesPI/timeLapsePhotos_Pi1_1' 
c='./DonneesPI/timeLapsePhotos_Pi1_2' 
d='./DonneesPI/timeLapsePhotos_Pi1_3' 
e='./DonneesPI/timeLAapsePhotos_Pi1_4'

imagettes_PI_0=imagettes[(imagettes["path"]==a) ]
liste_image_PI_O=imagettes_PI_0["filename"]



imagettes1=imagettes_PI_0[imagettes_PI_0["filename"]=="image_2019-04-30_18-17-14.jpg"]
to_drop=['path', 'filename', 'width', 'height', 'classe', 'index']

im=imagettes1.drop(to_drop,axis=1)
col=list(im.columns)
col = col[-1:] + col[:-1]
im=im[col]

annontation_reduit=im

#On pourra eventuellement essayer de changer l'ordre des colonnes pour passer imagetteName en première position


path_to_image=["/mnt/VegaSlowDataDisk/c3po/Images_aquises//DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-17-14.jpg"]

 


plt.imshow(img_test)

#import functions as fn
cnts=filtre_light(img_ref,img_test)

imageRectangles=img_test.copy()
#L'objectif est de faire des carrés pour les annotation d'alex et pour les carrés générés pour voir si certains correspondent
for i in range(len(im)):
    xmin=annontation_reduit["xmin"].iloc[i]
    ymin=annontation_reduit["ymin"].iloc[i]
    xmax=annontation_reduit["xmax"].iloc[i]
    ymax=annontation_reduit["ymax"].iloc[i]
    cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 255,0), 2) 
    
cv2.imwrite("/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/test_annotation_serie_2",imageRectangles)




















def place_generate_sqaure(imageB,cnts):

    
    image_annote=imageB.copy()
    batchImages = []
    liste_table = []
    imageSize= 28
    #On récupère les coordonnées des pixels différent par différence
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        name = (os.path.split(name2)[-1]).split(".")[0]
        name = name + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   

        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif

        subI, o, d, imageRectangles = GetSquareSubset(image_annote,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))
     
    table_full = pd.DataFrame(np.vstack(liste_table))
    
    
    #Ce serait bien de rajouter rename ici !!! Si ça n'entraine pas de bug
    table_full = table_full.rename(columns={0: 'imagettename', 1: 'xmin', 2: 'xmax', 3: 'ymin', 4: 'ymax'})
    table_full.iloc[:,1:]=table_full.iloc[:,1:].astype(int)

    
    for i in range(len(table_full)):
        xmin=table_full["xmin"].iloc[i]
        ymin=table_full["ymin"].iloc[i]
        xmax=table_full["xmax"].iloc[i]
        ymax=table_full["ymax"].iloc[i]
        cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 255,0), 2) 

    cv2.imwrite("/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/annote.jpg",imageRectangles)
    cv2.imwrite("/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/vierge.jpg",imageB)




place_generate_sqaure(imageB,cnts)














imageA=img1
imageB=img2






name2="image_2019-04-30_18-17-14.jpg"


#birds_is_catched(neurone_features,imageA,imageB,filtre_choice,coef_filtre,path_anotation,name2,height=2448,width=3264)







