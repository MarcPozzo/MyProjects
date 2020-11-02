#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:34:08 2020

@author: marcpozzo
"""



import pandas as pd # Pour manipuler des DataFrames pandas
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input
import Lenet_training_functions as fn


#Paramètres

test_size=0.2
epochs=200
batch_size = 600
zoom_range = 1.25
horizontal_flip = True
Minimum_Number_Class=100
dropout_rate=0.1
validation_steps=1

image_path="../../Materiels/images.csv"
Images=pd.read_csv(image_path)
Images["classe"].unique()


Minimum_Number_Class=100

    
df=fn.eliminate_small_categories(Images,Minimum_Number_Class)


test_size=0.2
data_train,data_test= train_test_split(df,stratify=df["classe"], test_size=test_size,random_state=42)

zoom_range=1.1
horizontal_flip=False


train_data_generator = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range = 10,
        zoom_range = zoom_range,
        horizontal_flip = horizontal_flip)


test_data_generator = ImageDataGenerator(
    preprocessing_function = preprocess_input)




train_generator = train_data_generator.flow_from_dataframe(dataframe=data_train,
                                                          directory="../../../../Pic_dataset",
                                                           x_col = "filename",
                                                           y_col = "classe",
                                                           class_mode ="sparse",
                                                          target_size = (28 , 28), 
                                                          batch_size = batch_size)


test_generator = test_data_generator.flow_from_dataframe(dataframe=data_test,
                                                          directory="../../../../Pic_dataset",
                                                           x_col = "filename",
                                                            y_col = "classe",
                                                           class_mode ="sparse",
                                                          target_size = (28 , 28), 
                                                          batch_size = batch_size)



NB_CLASSES=len(set(train_generator.classes))

dropout_rate=0.5
nn=fn.Lenet_archi(NB_CLASSES,dropout_rate)

# Compilation
nn.compile(loss='sparse_categorical_crossentropy',  # fonction de perte
              optimizer='adam',                 # algorithme de descente de gradient
              metrics=['accuracy'])             # métrique d'évaluation

epochs=2
history=nn.fit_generator( train_generator,
                           steps_per_epoch=len(data_train)//batch_size,
                           epochs=epochs,
                           workers=-1,
                           validation_data=test_generator,
                           validation_steps=len(data_test)//batch_size)