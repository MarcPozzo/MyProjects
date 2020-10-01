#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:40:37 2020

@author: marcpozzo
"""

#libraries importation
import pandas as pd
import os
import numpy as np
import cv2
import tensorflow
from numpy.random import seed
from keras.models import Model
from sklearn import metrics 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
#from sklearn import linear_model, preprocessing
#from keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
import pickle
from keras.models import load_model



#Paramètres
base_img_paths="/home/marcpozzo/Desktop/c3po/Images_aquises/"
generateur_path='/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/generateur.csv'
neurone_features='model.h8'
#lg_model='dt_clf_model.sav'
lg_model='finalized_model.sav'

test_size=0.2
epochs=200
batch_size = 400
zoom_range = 1
horizontal_flip = False
Minimum_Number_Class=100
dropout_rate=0


#Files importation
df=pd.read_csv(generateur_path)
df.drop('labels',inplace=True,axis=1)
classes=df["class"]

#Select Labels according to the size
All_Unique=df["class"].unique()
Utilisable=[]
for i in df["class"].unique():
    if df["class"][df["class"]==i].count()>Minimum_Number_Class:
        Utilisable.append(i)
Utilisable
Non_Utilisable=set(All_Unique)-set(Utilisable)
Non_Utilisable
for i in Non_Utilisable:
    df=df[df["class"]!=i]
df=df[df["class"]!="oiseau"]  
df["class"].unique()


#Adapte the generator on the new paths
for i in range(len(df["class"])):
    image_name=df["img_paths"].iloc[i]
    df["img_paths"].iloc[i]=os.path.join(base_img_paths,image_name)
    

#seed(1)
#tensorflow.random.set_seed(2)

data_train,data_test= train_test_split(df,stratify=df["class"], test_size=test_size,random_state=42)
train_data_generator = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        # data augmentation
        #rotation_range = 10,
        zoom_range = zoom_range,
        horizontal_flip = horizontal_flip
        )

test_data_generator = ImageDataGenerator(
    preprocessing_function = preprocess_input)




train_generator = train_data_generator.flow_from_dataframe(dataframe=data_train,
                                                          directory="",
                                                           x_col = "img_paths",
                                                           class_mode ="sparse",
                                                          target_size = (28 , 28), 
                                                          batch_size = batch_size)


test_generator = test_data_generator.flow_from_dataframe(dataframe=data_test,
                                                          directory="",
                                                           x_col = "img_paths",
                                                           class_mode ="sparse",
                                                          target_size = (28 , 28), 
                                                          batch_size = batch_size)


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

# Load the images train
X_train_img = convert_image(data_train.img_paths)
Y_train = data_train['class']

# Load the images test
X_test_img = convert_image(data_test.img_paths)
Y_test = data_test['class']


model = load_model(neurone_features)
intermediate_layer_model = Model(input=model.input, output=model.layers[-2].output)
X_test_features = intermediate_layer_model.predict(preprocess_input(X_test_img))




loaded_model = pickle.load(open(lg_model, 'rb'))
y_predict=loaded_model.predict(X_test_features)
y_test=data_test["class"]
print(metrics.classification_report(y_test, y_predict))