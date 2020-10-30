import os
from os import chdir
#chdir("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po_interface_mark/test_Yolo/6_classes_loss/")
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



tf_version=tf.__version__
if tf_version[0]=="1":
    tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None)
elif tf_version[0]=="2":
    print("Warning your are in tensorflow 2, it the programm doesn't work please try with tensorflow 1")
    tf.enable_eager_execution




#Paramètres par défaut
data_path='../../../../Pic_dataset/'
Mat_path="../../Materiels/"
#neurone_feature=Mat_path+"Models/drop_out.50"
#CNNmodel  = load_model(neurone_feature,compile=False)
Images=pd.read_csv(Mat_path+"images.csv")
imagettes=pd.read_csv(Mat_path+"imagettes.csv")


path_model_saved=Mat_path

"""
#imagettes=pd.read_csv("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises/imagettes.csv")
imagettes=common.to_reference_labels (imagettes,"classe")

index_train,index_test=common_2.split(imagettes)
imagettes_train=imagettes[imagettes["filename"].isin(index_train[:5])]
"""

index_train,index_test=common_2.split(Images)
Images_train=Images[Images["filename"].isin(index_train[:5])]
name_train="Nom_a renseigner"
string=path_model_saved+name_train
nbr_entrainement=1

batch_size=5
nbr=5
coef=1.3
flip=False




Images_2, labels_2, labels2_2=common_2.read_imagettes(Images_train,nbr=nbr,coef=coef,flip=flip)
Images_2=np.array(Images_2, dtype=np.float32)/255
labels_2=np.array(labels_2, dtype=np.float32)
train_ds_2=tf.data.Dataset.from_tensor_slices((Images_2, labels_2)).batch(batch_size)
Model=model.model(config.nbr_classes, config.nbr_boxes, config.cellule_y, config.cellule_x)


images_train_2=Images_2[Images_2["filename"].isin(index_train[:5])]
images_train_2_copy=images_train_2.copy()
images_2_2, labels_2_2, labels2_2_2=common_2.read_imagettes(images_train_2_copy,nbr=nbr,coef=coef,flip=flip)
images_2_2=np.array(images_2_2, dtype=np.float32)/255
labels_2_2=np.array(labels_2_2, dtype=np.float32)
train_ds_2_2=tf.data.Dataset.from_tensor_slices((images_2_2, labels_2_2)).batch(batch_size)



#chdir(path_model_saved)

optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4)
optimizer=tf.keras.optimizers.Adam()
#checkpoint=tf.train.Checkpoint(model=Model)
train_loss=tf.keras.metrics.Mean()   
test_loss=tf.keras.metrics.Mean() 

pack_train=(labels2_2,train_ds_2)

def train_get_loss(Images_train,nbr_entrainement,string,pack_train,train_loss,test_loss):


    (labels2_2,train_ds_2)=pack_train
    checkpoint=tf.train.Checkpoint(model=Model)
    checkpoint.restore(tf.train.latest_checkpoint(string))    
    LOSS,LOSS_test=common_2.train_progression_losses(train_ds_2, nbr_entrainement,string,labels2_2,optimizer,Model,train_loss,test_loss,checkpoint,train_ds_2,labels2_2)
    checkpoint.save(file_prefix=string)

    return LOSS,LOSS_test


nbr_entrainement=1
LOSS,LOSS_test=train_get_loss(Images_train,nbr_entrainement,string,pack_train,train_loss,test_loss)
print(LOSS,LOSS_test)
#name_train="Test_Test"
#LOSS=train_get_loss(imagettes_train,nbr_entrainement,name_train,nbr=5,coef=1.3,flip=True)

plt.plot(LOSS)
plt.plot(LOSS_test)

