from os import chdir
chdir("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po_interface_mark/test_Yolo/6_classes_loss/")
import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import math
import common
import config
import model
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

chdir("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po_interface_mark/test_Yolo/6_classes_loss/")

#Le problème est qu'on a pris l'indexation sur l'index et nom pas sur name_test
#Le mieux en réalité c'est d'appliquer le hasard sur liste_name_test 
#'image_2019-04-22_19-11-23.jpg' ne fonctionne pas bien
# Un problème avec /mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_3/image_2019-05-29_15-10-43.jpg
#Car les images sont trop proches l'une de l'autre



imagettes=pd.read_csv("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises/imagettes.csv")
imagettes=common.to_reference_labels (imagettes,"classe")
index_train,index_test=common.split(imagettes)

images, labels, labels2=common.read_imagettes(imagettes[imagettes["filename"].isin(index_train)])

images=np.array(images, dtype=np.float32)/255
labels=np.array(labels, dtype=np.float32)




Model=model.model(config.nbr_classes, config.nbr_boxes, config.cellule_y, config.cellule_x)

checkpoint=tf.train.Checkpoint(model=Model)

chdir("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po_interface_mark/test_Yolo/6_classes_loss/")
checkpoint=tf.train.Checkpoint(model=Model)
#checkpoint.restore(tf.train.latest_checkpoint("/mnt/VegaSlowDataDisk/c3po_interface_mark/test_Yolo/My_data/training/"))
checkpoint.restore(tf.train.latest_checkpoint("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po_interface_mark/Materiels/Models/Yolo_models/training_jeux_difficile"))

grid=np.meshgrid(np.arange(config.cellule_x, dtype=np.float32), np.arange(config.cellule_y, dtype=np.float32))
grid=np.expand_dims(np.stack(grid, axis=-1), axis=2)
grid=np.tile(grid, (1, 1, config.nbr_boxes, 1))

#for i in range(len(images)):
for i in range(len(images)):
  img=common.prepare_image(images[i], labels[i], False)
  img2=images[i].copy()
  predictions=Model(np.array([images[i]]))
  pred_boxes=predictions[0, :, :, :, 0:4]
  pred_conf=common.sigmoid(predictions[0, :, :, :, 4])
  pred_classes=common.softmax(predictions[0, :, :, :, 5:])
  ids=np.argmax(pred_classes, axis=-1)

  x_center=((grid[:, :, :, 0]+common.sigmoid(pred_boxes[:, :, :, 0]))*config.r_x)
  y_center=((grid[:, :, :, 1]+common.sigmoid(pred_boxes[:, :, :, 1]))*config.r_y)
  w=(np.exp(pred_boxes[:, :, :, 2])*config.anchors[:, 0]*config.r_x)
  h=(np.exp(pred_boxes[:, :, :, 3])*config.anchors[:, 1]*config.r_y)

  x_min=(x_center-w/2).astype(np.int32)
  y_min=(y_center-h/2).astype(np.int32)
  x_max=(x_center+w/2).astype(np.int32)
  y_max=(y_center+h/2).astype(np.int32)
  
  for y in range(config.cellule_y):
    for x in range(config.cellule_x):
      for b in range(config.nbr_boxes):
        if pred_conf[y, x, b]>0.10:
          color=list(config.dict.values())[ids[y, x, b]]
          cv2.circle(images[i], (x_center[y, x, b], y_center[y, x, b]), 1, color, 2)
          cv2.rectangle(images[i], (x_min[y, x, b], y_min[y, x, b]), (x_max[y, x, b], y_max[y, x, b]), color, 1)
          cv2.rectangle(images[i], (x_min[y, x, b], y_min[y, x, b]), (x_max[y, x, b], y_min[y, x, b]-15), color, cv2.FILLED)
          cv2.putText(images[i], "{:3.0%}".format(pred_conf[y, x, b]), (x_min[y, x, b], y_min[y, x, b]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (255, 255, 255), 1)
          
  tab_boxes=np.stack([y_min, x_min, y_max, x_max], axis=-1).reshape(-1, 4).astype(np.float32)
  pred_conf=pred_conf.reshape(-1)
  ids=ids.reshape(-1)
  tab_index=tf.image.non_max_suppression(tab_boxes, pred_conf, 5)
          
  for id in tab_index:
    if pred_conf[id]>0.10:
      x_min=tab_boxes[id, 1]
      y_min=tab_boxes[id, 0]
      x_max=tab_boxes[id, 3]
      y_max=tab_boxes[id, 2]

      color=list(config.dict.values())[ids[id]]
      cv2.rectangle(img2, (x_min, y_min), (x_max, y_max), color, 1)
      cv2.rectangle(img2, (x_min, y_min), (x_max, int(y_min-15)), color, cv2.FILLED)
      cv2.putText(img2, "{:3.0%}".format(pred_conf[id]), (x_min, int(y_min-5)), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (255, 255, 255), 1) # {%} ???
  
  
  
  
  cv2.imshow("Inference", images[i])
  cv2.resizeWindow("Inference", 12, 12)
  cv2.imshow("Bonne reponse", img)
  cv2.imshow("Non max suppression", img2)
    
  key=cv2.waitKey()&0xFF
  if key==ord('q'):
    quit()


def max_by_index(idx, arr):
    return (idx,) + np.unravel_index(np.argmax(arr[idx]), arr.shape[1:])

def max_by_index(idx, arr):
    return np.where(arr[idx] == np.max(arr[idx]))

"""

"""