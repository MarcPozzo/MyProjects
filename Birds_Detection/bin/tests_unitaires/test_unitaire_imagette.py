#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:34:12 2019

@author: khalid
"""


#This files displays performances for different types of images, imagettes, zoom and neural networks 


from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
import functions as fn
import cv2
import pandas as pd
from keras.models import Model, load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
p="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/6classes_zoom/"
#parameters to choose
repertoire="zoom_images" #or zoom images#original_images

# parameters 
finalSize = 96 # side size of the square of the final "imagette"
folderOut = "Rec_images/"


#Le zoom n'est pas en bas
#Rajouter ces deux fonctions dans functions
def aggrandissemet(xmax,xmin):
    distance=xmax-xmin
    centre=(xmax+xmin)/2
    xmax=centre+(distance*1.2)/2
    xmin=centre-(distance*1.2)/2
    
    return xmax,xmin



def retrecissement(xmax,xmin,zoom):
    distance=xmax-xmin
    centre=(xmax+xmin)/2
    xmax=centre+(distance*zoom)/2
    xmin=centre-(distance*zoom)/2
    
    return xmax,xmin



if repertoire=="original_images":
    imageA = cv2.imread("/mnt/VegaSlowDataDisk/c3po_interface/bin/testingInputs/EK000228.JPG")
    
if repertoire=="zoom_images":
    imageA = cv2.imread("/mnt/VegaSlowDataDisk/c3po_interface/bin/testingInputs/EK000228.JPG")
    
    
    
    

#We will compare the results for original imagette annoted by Alexandre and for wider imagette 
pathBase= "./"
folder_imagettes="/mnt/VegaSlowDataDisk/c3po/Images_aquises/"
#type_imagettes=["imagettes_bigger.csv","imagettes.csv"]
type_imagettes=["imagettes.csv"]
#paht_imagette=["/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes_bigger.csv","/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv"]
imageSize=28 
liste_resultats=[]

for imagette_type in type_imagettes:
    data = pd.read_csv(folder_imagettes+imagette_type)
    #Selecton only 1 imagette
    df=data[data["filename"]=="EK000228.JPG"]
  
    #ZOOM=[1,1.2]
    ZOOM=[1]
    for zoom in ZOOM:
        batchImages_unitaire = []
        print(" ")
        print("le zoom est de: ",zoom)
        print(" ")
        for i in range(len(df)):   
            h = df.iloc[i,]   
            h.xmax,h.xmin=retrecissement(h.xmax,h.xmin,zoom)
            h.ymax,h.ymin=retrecissement(h.ymax,h.ymin,zoom)

            h.xmax=int(round(h.xmax))
            h.xmin=int(round(h.xmin))
            h.ymax=int(round(h.ymax))
            h.ymin=int(round(h.ymin))    
            subI_unitaire, o, distance, img = fn.GetSquareSubset(imageA,h,verbose=False,xml=True)
            subI_unitaire = fn.RecenterImage(subI_unitaire,o)
            subI_unitaire = cv2.resize(subI_unitaire,(imageSize,imageSize))
            batchImages_unitaire.append(subI_unitaire)
            
        #dimension 728,28,3
        batchImages_array = np.vstack(batchImages_unitaire)
    
        #Neurone_files=["6c_rob","zoom_0.9:1.3_flip","zoom_1.3","drop_out.50","z1.3"]
        Neurone_files=["z1.3"]
        for n in Neurone_files:
            neurone_features=p+n
            print(" ")
            print(n)
            print(" ")
            model = load_model(neurone_features,compile=False)
            CNNmodel = Model(inputs=model.input, outputs=model.layers[-1].output)
            #features_unitaire=preprocess_input(np.array(batchImages_array))
            #features=features.reshape((features.shape[0], 28,28,3))
            #estimates = CNNmodel.predict(features)


            #subI = np.expand_dims(subI, axis=0) Inutile pour une image seule
            #batchImages.append(subI)
            estimates_unitaire = CNNmodel.predict(batchImages_array.reshape(-1,28,28,3))
            # on va comparer batch_reshape de test_unitaire avec celui dans find_square
            #batch_reshape=batchImages.reshape(-1,28,28,3)

            arg_resulat=estimates_unitaire.argmax(axis=1)
            print(arg_resulat)
            if (zoom==1) and (imagette_type=="imagettes.csv"):
                liste_resultats.append(arg_resulat)
#Les meilleurs résultats pour cette imagette est le modèle z1.3, on va donc tester
# findsquare avec ce modèle.


#Le resultat sans zoom peut confondre pigeon avec faisan ou corbeau mais aussi confondre autre et pigeon




#df.to_csv("/mnt/VegaSlowDataDisk/c3po_interface/bin/testingInputs/oiseau_lab_Alex.csv")

"""
arr_resultat=table_resultat.values




#Afficher les résultats trouvés
for i in range(len(table_resultat.columns)):
    print(table_resultat.columns[i], list(arg_resulat.flatten()).count(i) )
print("nombre total d'imagettes : ", len(arg_resulat))
"""



"""   #     Resize image
    subI = cv2.resize(subI,(finalSize,finalSize))
    imagetteName = h.filename[:-4]+"_"+h.classe+"_"+str(i)+".JPG"       
    cv2.imwrite(folderOut+imagetteName,subI)
    data["imagetteName"][i] = imagetteName
    
data.to_csv(pathBase+"imagettes.csv", index=None)
print("Done")"""

#Maintenant il faut faire pour au moins deux images

