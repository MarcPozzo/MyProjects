#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:07:35 2020

@author: marcpozzo
"""


import ast
import os
from os import chdir
from os.path import basename, join
import cv2
import pandas as pd
from keras.layers import Dropout
import numpy as np
import gc

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D,Flatten
from keras.layers.convolutional import Conv2D
import pickle

mat_path="../../../Materiels/"
fichierClasses= mat_path+"Table_Labels_to_Class.csv" # overwritten by --classes myFile
frame=pd.read_csv(fichierClasses,index_col=False)


def eliminate_small_categories(df,Minimum_Number_Class):
    numerous_labels_=[]
    all_labels_=df["classe"].unique()
    print("This is the list of different labels (acccording to the DataFrame) :",df["classe"].unique())
    print("Images are now deleting from data base if the population are below ",Minimum_Number_Class)  
    for i in df["classe"].unique():
        if df["classe"][df["classe"]==i].count()>Minimum_Number_Class:
            numerous_labels_.append(i)

    less_numerous_labels_=set(all_labels_)-set(numerous_labels_)
    #print(Non_Utilisable,"Non_Utilisable")
    for i in less_numerous_labels_:
        df=df[df["classe"]!=i] 

    print("This is the list of labels keep:", df["classe"].unique())
    return df



def cc(image,changement):
    if changement=="HSV":
        image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if changement=="BGRGRAY":
        image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    if changement=="BGR":
        image=cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        
    return image



def nn(dropout_rate=0.2):    
    
    lenet = Sequential()

    conv_1 = Conv2D(filters = 30,                     # Nombre de filtres
                kernel_size = (5, 5),            # Dimensions du noyau
                padding = 'valid',               # Mode de Dépassement
                input_shape = (28, 28, 4),       # Dimensions de l'image en entrée
                activation = 'relu')             # Fonction d'activation

    max_pool_1 = MaxPooling2D(pool_size = (2, 2))

    conv_2 = Conv2D(filters = 16,                    
                kernel_size = (3, 3),          
                padding = 'valid',             
                activation = 'relu')

    max_pool_2 = MaxPooling2D(pool_size = (2, 2))

    flatten = Flatten()

    dropout = Dropout(rate = dropout_rate)

    dense_1 = Dense(units = 128,
                activation = 'relu')

    dense_2 = Dense(units = 6,
                activation = 'softmax')

    lenet.add(conv_1)
    lenet.add(max_pool_1)
    lenet.add(conv_2)
    lenet.add(max_pool_2)
    lenet.add(dropout)
    lenet.add(flatten)
    lenet.add(dense_1)
    lenet.add(dense_2)



    
    
    
    lenet.compile(loss='sparse_categorical_crossentropy',  # fonction de perte
              optimizer='adam',                 # algorithme de descente de gradient
              metrics=['accuracy'])             # métrique d'évaluation
    
    return lenet


def convert_imagette(X):
    X_img=[]
    for image in X:
        # Load image
        img=cv2.imread(image)
        # Resize image
        
        X_img.append(img[...,::-1])
    return np.array(X_img)










def add_chanel(image,ar_in_liste_diff):

    b_channel, g_channel, r_channel = cv2.split(image)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, ar_in_liste_diff))
    
    return img_BGRA








def to_imagette4C(base,liste4D_diff):
    
    if len(base)!=len(liste4D_diff):
        print("la table de données et la liste ne font pas la même taille")
    imagette4C_liste4=[]
    

    for i in range(len(liste4D_diff)):
        image_4C_entire=liste4D_diff[i]
        x_min=base["xmin"].iloc[i]
        x_max=base["xmax"].iloc[i]
        y_min=base["ymin"].iloc[i]
        y_max=base["ymax"].iloc[i]

        imagette=image_4C_entire[y_min:y_max,x_min:x_max]
        imagette=cv2.resize(imagette,(28,28))
        imagette4C_liste4.append(imagette)
        
    return imagette4C_liste4


def get_Y(base):
    #data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
    data_generator = ImageDataGenerator()
    #      preprocessing_function = preprocess_input


    generator = data_generator.flow_from_dataframe(dataframe=base,
                                                              directory="",
                                                               x_col = "filename",
                                                               y_col="classe",
                                                               class_mode ="sparse",
                                                              target_size = (28 , 28), 
                                                              batch_size = len(base))

    Y=generator.classes
    
    return Y


def make_image_difference(base,liste_image_ref,liste_name_test):
    Gray_Diff_from_HSV=[]
    Gray_Diff_from_BGR=[]


    compteur=0
    for i in range(len(base)):
        compteur+=1
        #print("valeur du compteur",compteur)
        #name_test=base["filename"].iloc[i]
        name_test=liste_name_test[i]
        #print("name_test",name_test)
        imageA=cv2.imread(name_test)

        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        imageB=cv2.imread(name_ref)

        BGR_Diff = cv2.absdiff(imageA, imageB)
        BGR_Diff=cc(BGR_Diff,"BGRGRAY")
        Gray_Diff_from_BGR.append(BGR_Diff)

        imgAHSV=cc(imageA,"HSV")
        imgBHSV=cc(imageB,"HSV")

        HSV = cv2.absdiff(imgAHSV, imgBHSV)
        HSV_Diff = cc(HSV,"BGRGRAY")
        Gray_Diff_from_HSV.append(HSV_Diff)
        
    return Gray_Diff_from_BGR,Gray_Diff_from_HSV




def to_reference_labels (df,class_colum,frame=frame):

    #flatten list in Labels_File
    cat=[]
    for i in range(len(frame["categories"]) ):
        cat.append( frame["categories"][i] )

    liste = [ast.literal_eval(item) for item in cat]

    # set nouvelle_classe to be the "unified" class name
    for j in range(len(frame["categories"])):
        #classesToReplace = frame["categories"][j].split(",")[0][2:-1]
        className = frame["categories"][j].split(",")[0][2:-1]
        #df["nouvelle_classe"]=df["classe"].replace(classesToReplace,className)
        df[class_colum]=df[class_colum].replace(liste[j],className)

    return df


def open_table(df,liste_folders):
    #liste_folders=['./DonneesPI/timeLapsePhotos_Pi1_0','./DonneesPI/timeLapsePhotos_Pi1_1','./DonneesPI/timeLapsePhotos_Pi1_2','./DonneesPI/timeLapsePhotos_Pi1_3','./DonneesPI/timeLapsePhotos_Pi1_4']
    df=df[df["path"].isin(liste_folders)]

    df=to_reference_labels (df,"classe",frame=frame)

    categories_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","autre"]
    df=df[df["classe"].isin(categories_to_keep)]
    return df


def get_liste_name_test(base):
    
    
    absolute_path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    liste_name_test=[]
    filename_liste=[]
    for i in range(len(base)):
        folder_path=base["path"].iloc[i][1:]
        file_name=base["filename"].iloc[i]
        filename_liste.append(file_name)
        image_path=absolute_path+folder_path+"/"+file_name
        liste_name_test.append(image_path)
    
    return liste_name_test,filename_liste
"""
def get_liste_image_ref():

    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises/"
    path='../../../../../DonneesPI/'
    chdir(path)
    folders=['timeLapsePhotos_Pi1_0','timeLapsePhotos_Pi1_1','timeLapsePhotos_Pi1_2','timeLapsePhotos_Pi1_3','timeLapsePhotos_Pi1_4']
    liste_image_ref = []
    liste_name=[]
    for folder_choose in folders:
        print(len(liste_image_ref))
        folder=folder_choose
        pre_path=path+folder
        chdir(pre_path)
        for r, d, f in os.walk(path+folder):
            for file in f:
                if '.jpg' in file:
                    name=basename(join(r, file))
                    liste_name.append(name)
                    picture_path=pre_path+"/"+name
                    liste_image_ref.append(picture_path)
    return liste_image_ref,liste_name
"""
                    
"""

def get_liste_image_ref(path):

    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises/"
    path='../../../../../DonneesPI/'
    #chdir(path)
    folders=['timeLapsePhotos_Pi1_0']
    liste_image_ref = []
    liste_name=[]
    for folder_choose in folders:
        print(len(liste_image_ref))
        folder=folder_choose
        pre_path=path+folder
        chdir(pre_path)
        for r, d, f in os.walk(path+folder):
            for file in f:
                if '.jpg' in file:
                    name=basename(join(r, file))
                    liste_name.append(name)
                    picture_path=pre_path+"/"+name
                    liste_image_ref.append(picture_path)
    return liste_image_ref,liste_name
"""

def get_liste_image_ref(path):
        liste_image_ref=[]
        image_path='../../../../../Pic_dataset/'
        #chdir(image_path)
        for r, d, f in os.walk(image_path):
            for file in f:
                if '.jpg' in file:
                    name=basename(join(r, file))
                    #liste_name.append(name)
                    picture_path=image_path+name
                    liste_image_ref.append(picture_path)

        return liste_image_ref
    
def get_X_Y(base):
    
    path='../../../../../Pic_dataset/'
    #Get ref 
    #liste_name_test,filename_liste=get_liste_name_test(base)
    liste_name_test=list(base["filename"].unique())
    liste_name_test=[path+name for name in liste_name_test]
    #liste_image_ref,liste_name=get_liste_image_ref(path)
    liste_image_ref=get_liste_image_ref(path)
    X_img=convert_imagette(liste_name_test) 
    
    GBR_Diff,HSV_Diff,=make_image_difference(base,liste_image_ref,liste_name_test)

    #Cette étape prend du temps
    image_4C_liste4=list(map(add_chanel,X_img , GBR_Diff))
    del GBR_Diff,HSV_Diff
    del X_img
    gc.collect()

    
    #Ici on peut optimiser en créant un dico pour éviter de multiplier les images diff
    #Il suffirait d'aller voir dans name_test, vérifier que les images ne sont pas en double
    imagette4C_liste4=to_imagette4C(base,image_4C_liste4)
    del image_4C_liste4
    gc.collect()
    
    #C'est le X
    X=np.array(imagette4C_liste4)
    del imagette4C_liste4
    gc.collect()
    
    base["filename"]=liste_name_test
    Y=get_Y(base)
    
    return X,Y,base


def concatenate_X_Y(X,Y,category,liste_array):
    array=np.array(liste_array)
    if len(array)!=0:
        X=np.concatenate((array, X), axis=0)

        liste_fp = [category] * len(array)
        Y=liste_fp+Y
        
    return X,Y

def add_list(specific_animal_liste,animals_paths):
    liste_array=[]
    for image in specific_animal_liste:
        with open(animals_paths+image, "rb") as fp:   # Unpickling
            liste_image = pickle.load(fp)
        liste_array=liste_array+liste_image
    return liste_array


