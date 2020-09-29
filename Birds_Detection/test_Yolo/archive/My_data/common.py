from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface_mark/test_Yolo/My_data/")
import tensorflow as tf
from tensorflow.keras import layers, models
import json
import random
import cv2
import numpy as np
import math
import config
import pandas as pd
from pandas.core.common import flatten
import ast

fichierClasses= "/mnt/VegaSlowDataDisk/c3po/Images_aquises/Table_Labels_to_Class.csv" # overwritten by --classes myFile
frame=pd.read_csv(fichierClasses,index_col=False)

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

def sigmoid(x):
  x=np.clip(x, -50, 50)
  return 1/(1+np.exp(-x))

def softmax(x):
  e=np.exp(x)
  e_sum=np.sum(e)
  return e/e_sum

def prepare_image(image, labels, grille=True):
  img=image.copy()
  
  if grille is True:
    for x in range(config.r_x, config.largeur+config.r_x, config.r_x):
      for y in range(config.r_y, config.hauteur+config.r_y, config.r_y):
        cv2.line(img, (0, y), (x, y), (0, 0, 0), 1)
        cv2.line(img, (x, 0), (x, y), (0, 0, 0), 1)

  for y in range(config.cellule_y):
    for x in range(config.cellule_x):
      for box in range(config.nbr_boxes):
        if labels[y, x, box, 4]:
          ids=np.argmax(labels[y, x, box, 5:])
          x_center=int(labels[y, x, box, 0]*config.r_x)
          y_center=int(labels[y, x, box, 1]*config.r_y)
          w_2=int(labels[y, x, box, 2]*config.r_x/2)
          h_2=int(labels[y, x, box, 3]*config.r_y/2)
          x_min=x_center-w_2
          y_min=y_center-h_2
          x_max=x_center+w_2
          y_max=y_center+h_2
          cv2.rectangle(img, (x_min, y_min), (x_max, y_max), list(config.dict.values())[ids], 1)
          cv2.circle(img, (x_center, y_center), 1, list(config.dict.values())[ids], 2)
            
  return img

def bruit(image):
  h, w, c=image.shape
  n=np.random.randn(h, w, c)*random.randint(5, 30)
  return np.clip(image+n, 0, 255).astype(np.uint8)

def gamma(image, alpha=1.0, beta=0.0):
  return np.clip(alpha*image+beta, 0, 255).astype(np.uint8)

def intersection_over_union(boxA, boxB):
  xA=np.maximum(boxA[0], boxB[0])
  yA=np.maximum(boxA[1], boxB[1])
  xB=np.minimum(boxA[2], boxB[2])
  yB=np.minimum(boxA[3], boxB[3])
  interArea=np.maximum(0, xB-xA)*np.maximum(0, yB-yA)
  boxAArea=(boxA[2]-boxA[0])*(boxA[3]-boxA[1])
  boxBArea=(boxB[2]-boxB[0])*(boxB[3]-boxB[1])
  return interArea/(boxAArea+boxBArea-interArea)

def prepare_labels(fichier_image, objects, coeff=None):
    image=cv2.imread(fichier_image)

    ######################    
    trophozoite=0
    for o in objects:
      if config.dict2.index(o['category'])==4:
        trophozoite=1
        break
    if trophozoite==0:
      return None, None, None
    ######################
      
    if coeff is None:
      coeff=random.uniform(1.1, 2.5)
    image_r=cv2.resize(image, (int(coeff*config.largeur), int(coeff*config.hauteur)))
    image_r=gamma(image_r, random.uniform(0.7, 1.3), np.random.randint(60)-30)
    image_r=bruit(image_r)
    
    if coeff==1:
      shift_x=0
      shift_y=0
    else:
      shift_x=np.random.randint(image_r.shape[1]-config.largeur)
      shift_y=np.random.randint(image_r.shape[0]-config.hauteur)

    ratio_x=coeff*config.largeur/image.shape[1]
    ratio_y=coeff*config.hauteur/image.shape[0]

    flip=np.random.randint(4)
    if flip!=3:
      image_r=cv2.flip(image_r, flip-1)

    label =np.zeros((config.cellule_y, config.cellule_x, config.nbr_boxes, 5+config.nbr_classes), dtype=np.float32)
    label2=np.zeros((config.max_objet, 7), dtype=np.float32)

    nbr_objet=0
    for o in objects:
        id_class=config.dict2.index(o['category'])
        box=o['bounding_box']

        if flip==3:
          x_min=int(box['minimum']['c']*ratio_x)
          y_min=int(box['minimum']['r']*ratio_y)
          x_max=int(box['maximum']['c']*ratio_x)
          y_max=int(box['maximum']['r']*ratio_y)
        if flip==2:
          x_min=int((image.shape[1]-box['maximum']['c'])*ratio_x)
          y_min=int(box['minimum']['r']*ratio_y)
          x_max=int((image.shape[1]-box['minimum']['c'])*ratio_x)
          y_max=int(box['maximum']['r']*ratio_y)
        if flip==1:
          x_min=int(box['minimum']['c']*ratio_x)
          y_min=int((image.shape[0]-box['maximum']['r'])*ratio_y)
          x_max=int(box['maximum']['c']*ratio_x)
          y_max=int((image.shape[0]-box['minimum']['r'])*ratio_y)
        if flip==0:
          x_min=int((image.shape[1]-box['maximum']['c'])*ratio_x)
          y_min=int((image.shape[0]-box['maximum']['r'])*ratio_y)
          x_max=int((image.shape[1]-box['minimum']['c'])*ratio_x)
          y_max=int((image.shape[0]-box['minimum']['r'])*ratio_y)

        if x_min<shift_x or y_min<shift_y or x_max>(shift_x+config.largeur) or y_max>(shift_y+config.hauteur):
          continue
        x_min=(x_min-shift_x)/config.r_x
        y_min=(y_min-shift_y)/config.r_y
        x_max=(x_max-shift_x)/config.r_x
        y_max=(y_max-shift_y)/config.r_y

        area=(x_max-x_min)*(y_max-y_min)
        label2[nbr_objet]=[x_min, y_min, x_max, y_max, area, 1, id_class]
        
        x_centre=int(x_min+(x_max-x_min)/2)
        y_centre=int(y_min+(y_max-y_min)/2)
        x_cell=int(x_centre)
        y_cell=int(y_centre)

        a_x_min=x_centre-config.anchors[:, 0]/2
        a_y_min=y_centre-config.anchors[:, 1]/2
        a_x_max=x_centre+config.anchors[:, 0]/2
        a_y_max=y_centre+config.anchors[:, 1]/2

        id_a=0
        best_iou=0
        for i in range(len(config.anchors)):
          iou=intersection_over_union([x_min, y_min, x_max, y_max], [a_x_min[i], a_y_min[i], a_x_max[i], a_y_max[i]])
          if iou>best_iou:
            best_iou=iou
            id_a=i

        label[y_cell, x_cell, id_a, 0]=(x_max+x_min)/2
        label[y_cell, x_cell, id_a, 1]=(y_max+y_min)/2
        label[y_cell, x_cell, id_a, 2]=x_max-x_min
        label[y_cell, x_cell, id_a, 3]=y_max-y_min
        label[y_cell, x_cell, id_a, 4]=1.
        label[y_cell, x_cell, id_a, 5+id_class]=1.
        
        nbr_objet=nbr_objet+1
        if nbr_objet==config.max_objet:
          print("Nbr objet max atteind !!!!!")
          break

    ######################
    trophozoite=0
    for y in range(config.cellule_y):
      for x in range(config.cellule_x):
        for b in range(config.nbr_boxes):
          if np.argmax(label[y, x, b, 5:])==4:
            trophozoite=1
    if not trophozoite:
      return None, None, None
    ######################
      
    return image_r[shift_y:shift_y+config.hauteur, shift_x:shift_x+config.largeur], label, label2

def read_json(file, nbr=1, nbr_fichier=None):
  images=[]
  labels=[]
  labels2=[]
  with open(file) as json_file:        
    data=json.load(json_file)
    id=0
    for p in data:
      print(id, p['image']['pathname'])
      id+=1
      for i in range(nbr):
        image, label, label2=prepare_labels("./{}".format(p['image']['pathname']), p['objects'])
        if image is not None:
          images.append(image)
          labels.append(label)
          labels2.append(label2)
      if nbr_fichier is not None:
        if id==nbr_fichier:
          break
  images=np.array(images)
  labels=np.array(labels)
  labels2=np.array(labels2)
  return images, labels, labels2

imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
def prepare_marc(name_test):
  #s'assurer qu'on a bien repéré un oiseau, 
  #il faut changer de format
  images=[]
  labels=[]
  labels2=[]
  #file=imagette.iloc[line]
  #imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
  #file='image_2019-04-30_18-55-20.jpg'
  #name_test=file
  One_image=imagettes[imagettes["filename"]==name_test]
  path="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/"
  #big_image_path="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-55-20.jpg"
  big_image_path=path+name_test
  big_image=cv2.imread(big_image_path)
  for i in range(len(One_image)):
      One_imagette=One_image.iloc[i]
      xmin,xmax,ymin,ymax=One_imagette[["xmin","xmax","ymin","ymax"]]
      label=One_imagette["classe"]
      #small_image=big_image[xmin:xmax,ymin:ymax]
      small_image=big_image[ymin:ymax,xmin:xmax]
      images.append(small_image)
      labels.append(label)
      labels2.append(label)
      
  return images, labels, labels2

imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
"""
def read_marc(file,imagettes):
    #regarder dans prepare label quel est le format des images en sortie, est-il toujours le même ?
    #résoudre probleme des lables qui sont censé être des arrays
    images=[]
    labels=[]
    labels2=[]
    
    liste_name_test=list(imagettes["filename"][imagettes["path"]=='./DonneesPI/timeLapsePhotos_Pi1_0'].unique())
    
    for name_test in liste_name_test[:10]:
        Images, Labels, Labels2=prepare_labels_marc(name_test)
        images.append(Images)
        labels.append(Labels)
        labels.append(Labels2)
        
    flat_list = [item for sublist in images for item in sublist]
"""    
    
    
#  
def read_imagettes():
  
  imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
  images=[]
  labels=[]
  labels2=[]
  liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
  imagettes=to_reference_labels (imagettes,"classe")
  imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)]    
  liste_name_test=list(imagettes["filename"][imagettes["path"]=='./DonneesPI/timeLapsePhotos_Pi1_0'].unique())

  for name_test in liste_name_test:
      image, label, label2=prepare_labels_marc(name_test)
      if image is not None:
          images.append(image)
          labels.append(label)
          labels2.append(label2)
 
    
  images=np.array(images)
  labels=np.array(labels)
  labels2=np.array(labels2)
  return images, labels, labels2    
    
    
def read_imagettes_2():
  
  imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
  images=[]
  labels=[]
  labels2=[]
  liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
  imagettes=to_reference_labels (imagettes,"classe")
  imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)]    
  liste_name_test=list(imagettes["filename"][imagettes["path"]=='./DonneesPI/timeLapsePhotos_Pi1_3'].unique())

  for name_test in liste_name_test:
      image, label, label2=prepare_labels_marc(name_test)
      if image is not None:
          images.append(image)
          labels.append(label)
          labels2.append(label2)
 
    
  images=np.array(images)
  labels=np.array(labels)
  labels2=np.array(labels2)
  return images, labels, labels2  
 
    
def prepare_labels_marc(name_test):
  #image=cv2.imread(fichier_image)


    #file=imagette.iloc[line]
    #imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    #file='image_2019-04-30_18-55-20.jpg'
    #name_test=file
    
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
    imagettes=to_reference_labels (imagettes,"classe")
    imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)]  
    One_image=imagettes[imagettes["filename"]==name_test]
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_3/"
    #big_image_path="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-55-20.jpg"
    big_image_path=path+name_test
    big_image=cv2.imread(big_image_path)
      
    coeff=1
    image_r=cv2.resize(big_image, (int(coeff*config.largeur), int(coeff*config.hauteur)))

    shift_x=0
    shift_y=0
    
    ratio_x=coeff*config.largeur/big_image.shape[1]
    ratio_y=coeff*config.hauteur/big_image.shape[0]


    label =np.zeros((config.cellule_y, config.cellule_x, config.nbr_boxes, 5+config.nbr_classes), dtype=np.float32)
    label2=np.zeros((config.max_objet, 7), dtype=np.float32)

    nbr_objet=0
    for i in range(len(One_image)):
        One_imagette=One_image.iloc[i]
        classe=One_imagette["classe"]
        id_class=config.dict2.index(classe)
        (x_min,x_max,y_min,y_max)=One_imagette[["xmin","xmax","ymin","ymax"]]
        x_min=int(x_min*ratio_x)
        x_max=int(x_max*ratio_x)
        y_min=int(y_min*ratio_y)
        y_max=int(y_max*ratio_y)



        x_min=(x_min-shift_x)/config.r_x
        y_min=(y_min-shift_y)/config.r_y
        x_max=(x_max-shift_x)/config.r_x
        y_max=(y_max-shift_y)/config.r_y

        area=(x_max-x_min)*(y_max-y_min)
        label2[nbr_objet]=[x_min, y_min, x_max, y_max, area, 1, id_class]
        
        x_centre=int(x_min+(x_max-x_min)/2)
        y_centre=int(y_min+(y_max-y_min)/2)
        x_cell=int(x_centre)
        y_cell=int(y_centre)

        a_x_min=x_centre-config.anchors[:, 0]/2
        a_y_min=y_centre-config.anchors[:, 1]/2
        a_x_max=x_centre+config.anchors[:, 0]/2
        a_y_max=y_centre+config.anchors[:, 1]/2

        id_a=0
        best_iou=0
        for i in range(len(config.anchors)):
          iou=intersection_over_union([x_min, y_min, x_max, y_max], [a_x_min[i], a_y_min[i], a_x_max[i], a_y_max[i]])
          if iou>best_iou:
            best_iou=iou
            id_a=i

        label[y_cell, x_cell, id_a, 0]=(x_max+x_min)/2
        label[y_cell, x_cell, id_a, 1]=(y_max+y_min)/2
        label[y_cell, x_cell, id_a, 2]=x_max-x_min
        label[y_cell, x_cell, id_a, 3]=y_max-y_min
        label[y_cell, x_cell, id_a, 4]=1.
        label[y_cell, x_cell, id_a, 5+id_class]=1.
        
        nbr_objet=nbr_objet+1
        if nbr_objet==config.max_objet:
          print("Nbr objet max atteind !!!!!")
          break

    return image_r,label,label2