#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:28:26 2020

@author: marcpozzo
"""
import numpy as np 
import functions_4C as fn
import pickle
import os
from sklearn.model_selection import train_test_split
from numpy import save
import pickle

#path
Mat_path="../../Materiels/"
pic_4C = Mat_path+"4D_Pictures/"
fp_path=pic_4C+"FP/Neurone_name/dossier0/"
path_list_4Canimals=pic_4C+"animals_match/dossier0/"
path_to_load=pic_4C+'Alex_db/HSV_Pictures/'

#initialized parameters
corbeaux_liste,pigeons_liste,faisans_liste,lapins_liste,chevreuils_liste=([] for i in range(5))
dic_an_list={"corbeau.txt":corbeaux_liste,"pigeon.txt":pigeons_liste,"faisan.txt":faisans_liste,"lapin.txt":lapins_liste,"chevreuil.txt":chevreuils_liste}
test_size=0.2


#Get 4C chanels from annoted images
X_test=np.load(path_to_load+'X_test_HSV.dat')
X_train=np.load(path_to_load+'X_train_HSV.dat')
with open(path_to_load+"Y_test_Alex_db.txt", "rb") as fp:   # Unpickling
    Y_test = pickle.load(fp)

with open(path_to_load+"Y_train_Alex_db.txt", "rb") as fp:   # Unpickling
    Y_train = pickle.load(fp)
X_im_anote=np.concatenate((X_train,X_test),axis=0)
Y_im_anote=Y_train+Y_test



#Get 4C chanels from false positive images

list_of_list_fp=os.listdir(fp_path)
liste_fp=[]
for fp in list_of_list_fp:
    with open(fp_path+fp, "rb") as fp:   # Unpickling
        liste1type_fp = pickle.load(fp)
    liste_fp=liste_fp+liste1type_fp
array_fp=np.array(liste_fp)
X_im_anote_fp=np.concatenate((array_fp, X_im_anote), axis=0)
y_to_add = [0] * len(array_fp)
Y_im_anote_fp=y_to_add+Y_im_anote




#Get 4C chanels from animals matched pictures
liste_animals=os.listdir(path_list_4Canimals)                         
for image in liste_animals:
    an=image.split("_")[-1]
    dic_an_list[an].append(image)
liste_array_corbeaux=fn.add_list(corbeaux_liste,path_list_4Canimals)
liste_array_pigeons=fn.add_list(pigeons_liste,path_list_4Canimals)
liste_array_faisans=fn.add_list(faisans_liste,path_list_4Canimals)
liste_array_lapins=fn.add_list(lapins_liste,path_list_4Canimals)
liste_array_chevreuils=fn.add_list(chevreuils_liste,path_list_4Canimals)




map_listes_names={"pigeon":[liste_array_pigeons,5],"corbeau":[liste_array_corbeaux,2],"chevreuil":[liste_array_chevreuils,1],
                     "faisan":[liste_array_faisans,3],"lapin":[liste_array_lapins,4]}

for animal in map_listes_names.keys():
    category=map_listes_names[animal][1]
    liste_array=map_listes_names[animal][0]
    X,Y=fn.concatenate_X_Y(X_im_anote_fp,Y_im_anote_fp,category,liste_array)
    
#Get train and test subset    
indices=list(range(len(Y)))
Y_train,Y_test,indices_train,indices_test=train_test_split(Y,indices,stratify=Y,test_size=test_size,random_state=42)
X_train=[X[i] for i in indices_train]
X_test=[X[i] for i in indices_test]

X_train=np.array(X_train)
X_test=np.array(X_test)

save('imagettes_train_all_types.npy', X_train)
save('imagettes_test_all_types.npy', X_test)


with open("labels_train_all_types.txt", "wb") as fp:   #Pickling
    pickle.dump(Y_train, fp)

with open("labels_test_all_types.txt", "wb") as fp:   #Pickling
    pickle.dump(Y_test, fp)


