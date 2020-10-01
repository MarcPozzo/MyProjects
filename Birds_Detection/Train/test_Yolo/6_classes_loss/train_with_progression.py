import os
from os import chdir
chdir("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po_interface_mark/test_Yolo/6_classes_loss/")
import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import common
import config
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import common_2
import shutil
batch_size=16


path_model_saved="/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po_interface_mark/Materiels/Models/Yolo_models/"


imagettes=pd.read_csv("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises/imagettes.csv")
imagettes=common.to_reference_labels (imagettes,"classe")
index_train,index_test=common.split(imagettes)
imagettes_train=imagettes[imagettes["filename"].isin(index_train)]
name_train="Nom_a renseigner"
string=path_model_saved+name_train
nbr_entrainement=1


nbr=5
coef=1.3
flip=False




imagettes_train_copy=imagettes_train.copy()
images_2, labels_2, labels2_2=common_2.read_imagettes(imagettes_train_copy,nbr=nbr,coef=coef,flip=flip)
images_2=np.array(images_2, dtype=np.float32)/255
labels_2=np.array(labels_2, dtype=np.float32)
train_ds_2=tf.data.Dataset.from_tensor_slices((images_2, labels_2)).batch(batch_size)
Model=model.model(config.nbr_classes, config.nbr_boxes, config.cellule_y, config.cellule_x)



imagettes_train_2=imagettes[imagettes["filename"].isin(index_test)]
imagettes_train_2_copy=imagettes_train_2.copy()
images_2_2, labels_2_2, labels2_2_2=common_2.read_imagettes(imagettes_train_2_copy,nbr=nbr,coef=coef,flip=flip)
images_2_2=np.array(images_2_2, dtype=np.float32)/255
labels_2_2=np.array(labels_2_2, dtype=np.float32)
train_ds_2_2=tf.data.Dataset.from_tensor_slices((images_2_2, labels_2_2)).batch(batch_size)



chdir(path_model_saved)

optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4)
#checkpoint=tf.train.Checkpoint(model=Model)
train_loss=tf.keras.metrics.Mean()   
test_loss=tf.keras.metrics.Mean() 

pack_train=(labels2_2,train_ds_2)

def train_get_loss(imagettes_train,nbr_entrainement,string,pack_train,train_loss,test_loss):


    (labels2_2,train_ds_2)=pack_train
    checkpoint=tf.train.Checkpoint(model=Model)
    checkpoint.restore(tf.train.latest_checkpoint(string))    
    LOSS,LOSS_test=common_2.train_progression_losses(train_ds_2, nbr_entrainement,string,labels2_2,optimizer,Model,train_loss,test_loss,checkpoint,train_ds_2,labels2_2)
    checkpoint.save(file_prefix=string)

    return LOSS,LOSS_test


nbr_entrainement=1
LOSS,LOSS_test=train_get_loss(imagettes_train,nbr_entrainement,string,pack_train,train_loss,test_loss)
print(LOSS,LOSS_test)
#name_train="Test_Test"
#LOSS=train_get_loss(imagettes_train,nbr_entrainement,name_train,nbr=5,coef=1.3,flip=True)

plt.plot(LOSS)
plt.plot(LOSS_test)

