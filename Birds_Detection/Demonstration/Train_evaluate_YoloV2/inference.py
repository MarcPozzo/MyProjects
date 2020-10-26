#In this script we try YoloV2 on 3 images containing birds. 
#For each of them a couple of image are displayed the first one with the right answer the other one with birds find by Yolo

#Load Libraries
import tensorflow as tf
import cv2
import numpy as np
import common as common 
import config
import model
import pandas as pd

#Set Parameters
Mat_path="../../Materiels/"
neurone="Models/training_jeux_difficile"
string=Mat_path+neurone
imagettes_to_keep=['image_2019-06-14_15-46-54.jpg', 'image_2019-06-14_15-47-11.jpg','image_2019-06-14_15-47-28.jpg']


tf_version=tf.__version__
if tf_version[0]=="1":
    tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None)
elif tf_version[0]=="2":
    print("Warning your are in tensorflow 2, it the programm doesn't work please try with tensorflow 1")
    tf.enable_eager_execution



#Gather images in a dataset object
imagettes=pd.read_csv(Mat_path+"imagettes.csv")
imagettes=imagettes[imagettes["filename"].isin(imagettes_to_keep)]
imagettes=common.to_reference_labels (imagettes,"classe")
images, labels, labels2=common.read_imagettes(imagettes)
images=np.array(images, dtype=np.float32)/255
labels=np.array(labels, dtype=np.float32)
dataset=tf.data.Dataset.from_tensor_slices((images, labels)).batch(config.batch_size)


#Load the model
Model=model.model(config.nbr_classes, config.nbr_boxes, config.cellule_y, config.cellule_x)
print("the next line shoudl return something like <tensorflow.python.training.checkpointable.util.CheckpointLoadStatus at .....>" )
checkpoint=tf.train.Checkpoint(model=Model)
checkpoint.restore(tf.train.latest_checkpoint(string))



#Set a grid
grid=np.meshgrid(np.arange(config.cellule_x, dtype=np.float32), np.arange(config.cellule_y, dtype=np.float32))
grid=np.expand_dims(np.stack(grid, axis=-1), axis=2)
grid=np.tile(grid, (1, 1, config.nbr_boxes, 1))


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
  
  
  
  
  cv2.imshow("Bonne reponse", img)
  cv2.imshow("Inference", img2)
    
  key=cv2.waitKey()&0xFF
  if key==ord('q'):
    quit()


