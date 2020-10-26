#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:24:09 2020

@author: marcpozzo
"""
import common
import pandas as pd
from sklearn.model_selection import train_test_split

#On va faire un nouveau tableau d'imagettes
#On compte le nombre d'élements à identifier
#On propose l'oiseau ou l'ois le moins fréquent si il y en a plusieurs 
#On peut faire aussi selon le dossier y aura pas de confit par rapport à ça.


imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon"]
imagettes=common.to_reference_labels (imagettes,"classe")
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)]
folder_to_keep= ['./DonneesPI/timeLapsePhotos_Pi1_4',
       './DonneesPI/timeLapsePhotos_Pi1_3',
       './DonneesPI/timeLapsePhotos_Pi1_2',
       './DonneesPI/timeLapsePhotos_Pi1_1',
       './DonneesPI/timeLapsePhotos_Pi1_0']   
imagettes=imagettes[imagettes["path"].isin(folder_to_keep)]
imagettes['freq'] = imagettes.groupby('filename')['filename'].transform('count')
imagettes.groupby('classe')['classe'].transform('count')







imagettes["cat_maj"]=0
liste_name_test=imagettes["filename"].unique()
for name_test in liste_name_test:
    list_cat=list(imagettes["classe"][imagettes["filename"]==name_test].unique())
    cat_maj=list_cat[0]
    imagettes["cat_maj"][imagettes["filename"]==name_test]=cat_maj

dataframe =imagettes.sort_values('filename').drop_duplicates(subset=['filename'])

fn_train,fn_test=train_test_split(dataframe["filename"],stratify=dataframe[['path', 'classe']],random_state=42,test_size=0.2)

"""
index_train,index_test=train_test_split(imagettes.index,stratify=imagettes[['path', 'classe']],random_state=42)
index_train_reduit=index_test[:25]
images, labels, labels2=common.read_imagettes(imagettes.loc[index_train_reduit])
"""

#Inquire the prior categories
def select_one_category(list_cat):
    if len(list_cat)==1:
        category=list_cat[0]
        
    if len(list_cat)>1.1:
        if "faisan" in list_cat:
            category="faisan"
        elif "pigeon" in list_cat:
            category="pigeon"  
        elif "corneille" in list_cat:
            category="corneille"
        elif "lapin" in list_cat:
            category="lapin"             
        elif "chevreuil" in list_cat:
            category="chevreuil" 
    return category




###V2
    
dic_name_test={}
for name_test in liste_name_test:
    liste_animals=list(imagettes["classe"][imagettes["filename"]==name_test].values)
    dic_name_test[name_test]=select_one_category(liste_animals)

ind=imagettes["index"].iloc[0]

for ind in imagettes.index:
    imagettes["cat_maj"].loc[ind]=dic_name_test[imagettes["filename"].loc[ind]]
    
fn_train,fn_test=train_test_split(dataframe["filename"],stratify=dataframe[['path', 'cat_maj']],random_state=42,test_size=0.2)

#Split DataCategories with adaptapted stratify
def split(imagettes):
    #get dic name_test less birds vs represented categories
    dic_name_test={}
    for name_test in liste_name_test:
        liste_animals=list(imagettes["classe"][imagettes["filename"]==name_test].values)
        dic_name_test[name_test]=select_one_category(liste_animals)

    #fill df with dic
    for ind in imagettes.index:
        imagettes["cat_maj"].loc[ind]=dic_name_test[imagettes["filename"].loc[ind]]
    
    #Split the DataSet
    fn_train,fn_test=train_test_split(dataframe["filename"],stratify=dataframe[['path', 'cat_maj']],random_state=42,test_size=0.2)
    
    return fn_train,fn_test




