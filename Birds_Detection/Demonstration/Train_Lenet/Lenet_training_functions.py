#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:38:59 2020

@author: marcpozzo
"""






from keras.models import Sequential # Pour construire un réseau de neurones
from keras.layers import   MaxPooling2D,Dense, Conv2D,Dropout,Flatten# Pour instancier une couche dense
import cv2
import numpy as np







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


def Lenet_archi(NB_CLASSES,dropout_rate):
    lenet = Sequential()

    conv_1 = Conv2D(filters = 30,                     # Nombre de filtres
                    kernel_size = (5, 5),            # Dimensions du noyau
                    padding = 'valid',               # Mode de Dépassement
                    input_shape = (28, 28, 3),       # Dimensions de l'image en entrée
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
    
    return lenet


#Convert to 28,28 shape array 
def convert_image(X):
    X_img=[]
    for image in X:
        # Load image
        img=cv2.imread(image)
        # Resize image
        img=cv2.resize(img,(28,28))
        # for the black and white image
        if img.shape==(28, 28):
            img=img.reshape([28,28,1])
            img=np.concatenate([img,img,img],axis=2)
        # cv2 load the image BGR sequence color (not RGB)
        X_img.append(img[...,::-1])
    return np.array(X_img)


#Creation of dictionnary according to generator classes to convert the number labels (0,1,2,3,4,5) to tring labels ("autre","corbeau",....)
def map_prediction(arg_predict,generator):
    dictionnaire=generator.class_indices
    dictionnaire_inv = {v: k for k, v in dictionnaire.items()}
    
    Keys=[]
    Values=[]

    for i in range(len(arg_predict)) :
        Keys.append(arg_predict[i])
        Values.append(dictionnaire_inv[arg_predict[i]])

    #print(metrics.classification_report(Y_test, Values))
    return Values

#Take a wider image than the annotation
def dezoom_image(xmin,ymin,xmax,ymax,coef_raise,image):
    limit_widht=image.shape[0]
    limit_height=image.shape[1]
    width=xmax-xmin
    dev_width=int((coef_raise-1)*(width/2)) #take width deviation
    length=ymax-ymin
    dev_length=int(((coef_raise-1)*length/2))
    xmin-=dev_width
    xmin=max(0,xmin)
    xmax+=dev_width
    xmax=min(xmax,limit_height)
    ymin-=dev_length
    ymin=max(0,ymin)
    ymax+=dev_length
    ymax=min(ymax,limit_widht)
    return xmin,ymin,xmax,ymax

