#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:15:20 2020

@author: marcpozzo
"""















import ast 
import pandas as pd
import math
from matplotlib import pyplot as plt
from scipy.stats.mstats import mquantiles
from scipy.optimize import leastsq
import pylab
import imutils
import cv2
import numpy as np
from keras.applications.vgg16 import preprocess_input
from imutils import grab_contours
from shapely.geometry import Polygon
import os
from keras.models import Model, load_model
#Pour faire fonctionner le code Khalid
import joblib
from keras.applications import  imagenet_utils
from keras.applications import VGG16
model = VGG16(weights="imagenet", include_top=False)
c3poFolder="/mnt/VegaSlowDataDisk/c3po_interface/"
filtre_RL = joblib.load(c3poFolder+"bin/output/RL_annotation_model")
Model1 = joblib.load(c3poFolder+"bin/output/model.cpickle")
from os import chdir





fichierClasses= "/mnt/VegaSlowDataDisk/c3po/Images_aquises/Table_Labels_to_Class.csv" # overwritten by --classes myFile
frame=pd.read_csv(fichierClasses,index_col=False)













# allow easy plotting
def disp(img): 
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img);plt.show()

def hist(img): #    Trace le histogramme 
    plt.hist(img.ravel(),np.arange(-0.5,256.5,1.));plt.show()

def plot(x,y):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x,y)
    fig.show()

# histogram fitting and plot

fitfunc  = lambda p, x: p[0]*pylab.exp(-0.5*((x-p[1])/p[2])**2)+p[3]
errfunc  = lambda p, x, y: (y - fitfunc(p, x))

def fitHistNorm(xdata,ydata,center,plotIt=False):
    init  = [1.0, center, 10, 0]
    out   = leastsq( errfunc, init, args=(xdata, ydata))
    c = out[0]
    if(plotIt):
        pylab.plot(xdata, fitfunc(c, xdata),label="fit")
        pylab.plot(xdata, ydata,label="data")
        pylab.legend(loc='upper right')
        pylab.title(r'$A = %.3f\  \mu = %.3f\  \sigma = %.3f\ k = %.3f $' %(c[0],c[1],abs(c[2]),c[3]));
        pylab.show()
    return(c)

def NormThreshold(layer,center=128,cutFact=1.96):
    binsHist = np.arange(0,257)-0.5
    x = np.arange(0,256)
    layer = layer + center
    diffR=plt.hist(layer.ravel(),bins=binsHist)
    diffR[0][center] = (diffR[0][center+1] + diffR[0][center-1])/2
    # plot(x,diffR[0])

    # fit the histogram with a gaussian
    c = fitHistNorm(x,diffR[0],center)
    print(c)
    meanNoise = c[1]
    sdNoise = abs(c[2])
    lowThr = round(meanNoise-cutFact*sdNoise).astype(int)
    highThr = round(meanNoise+cutFact*sdNoise).astype(int)
    print("lowThr:",lowThr,"; highThr:",highThr,"\n")
    diffl = (((layer> highThr) | (layer<lowThr))*255).astype("uint8")

    return(diffl)

# easy execution
def source(fileName) :
    exec(open(fileName).read())


# picture treatment

def maskandblur(img,landscapeMask,blurFact) :  ## Délimite l'espace de recherche et floutte les capture pour une meilleurs annalyse 

    imageA = cv2.bitwise_and(img,img,mask=landscapeMask)
    a=cv2.GaussianBlur(imageA,(blurFact,blurFact),sigmaX=0)

    return a,imageA;

def FindDiffs (a,b,kernel,prob=[0.001]):
    ## permet de repérer les différences entre deux photos t et t+1

    (score, diff) = compare_ssim(a, b, full=True,multichannel=True)
    diff = (diff * 255).astype("uint8")
    blur = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(src=blur, thresh=mquantiles(diff.ravel(),prob=prob), 
            maxval=255,type=cv2.THRESH_BINARY_INV)[1] 
    thresh = cv2.dilate(thresh,kernel)  # to be replaced by "closeIn" 
    thresh = cv2.erode(thresh,kernel,iterations=1)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts) # could be replaced by "findContours directement"

    # additional filtering
    cntsOut = [];
    for ic in range(1,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        if w>blurFact/2 or h>blurFact/2: # il faudrait ajouter la distance et pondérer ça
            cntsOut.append(cnts[ic]);

    return cntsOut;



def FindDiffs (a,b,kernel):
    ## permet de repérer les différences entre deux photos t et t+1

    (score, diff) = compare_ssim(a, b, full=True,multichannel=True)
    diff = (diff * 255).astype("uint8")
    blur = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(subImagesDir2+"ablur-"+timeStamp+".JPG",blur)
    thresh = cv2.threshold(src=blur, thresh= 210, 
            maxval=255,type=cv2.THRESH_BINARY_INV)[1] 
    # additional filtering
#    cntsOut = [];
#    for ic in range(1,len(cnts)):
#        (x, y, w, h) = cv2.boundingRect(cnts[ic])
#        if w>blurFact/2 or h>blurFact/2: # il faudrait ajouter la distance et pondérer ça
#            cntsOut.append(cnts[ic]);

    return cnts,thresh;






def SaveDrawPi (imageA,imageB,kernel,subImagesDir1,subImagesDir2,thresh,name1,name2,cnts,maxAnalDL) : 
    ## Permet d'enregistrer et de tracer les contours repérés plus haut
    ## les images sont stockées dans le fichier "XXX.JPG_subImages"
#    cv2.imwrite(subImagesDir2+"diffsTrRaw"+timeStamp2+".JPG",thresh)



    # open the output CSV file for writing
    
    iSave = 1
    
    batchImages = []
    table = []
#    np.empty((1,5), dtype = "int")  
    
    for ic in range(0,len(cnts)):
        
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        name = (os.path.split(name2)[-1]).split(".")[0]
        name = name + "_" + str(ic) + ".JPG"
        f = pd.Series()
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        if( (f.xmax-f.xmin) < 500 and  (f.ymax-f.ymin) < 350): # birds should less than 500 pixels wide and 350 high
            subI, o, d, imageRectanglesB = GetSquareSubset(imageB,f,verbose=True)
            subI = RecenterImage(subI,o)
            subI = cv2.resize(subI,(224,224))
            subI = np.expand_dims(subI, axis=0)
            subI = imagenet_utils.preprocess_input(subI)
            batchImages.append(subI)
            table.append(np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5)))
#            table1 = np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5))
            
    table = pd.DataFrame(np.vstack(table))
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    max_probas = np.amax(filtre_RL.predict_proba(np.array(table.iloc[:,1:], 
                                                          dtype = "float64")), axis = 1)
    max_probas = np.argsort(-max_probas)
    
    # todo : add max_probas to out table in addition to results from DL
    #        and keep all "imagettes" in table
    table = table.iloc[list(max_probas)[:min(len(cnts),maxAnalDL)],:]
    
    
    batchImages = [batchImages[i] for i in list(max_probas)[:min(len(cnts),maxAnalDL)]]
    
    batchImages = np.vstack(batchImages)
    print("features extraction")
    features = model.predict(batchImages, batch_size=4) # why 5 if only 4 proc on the pi...
    features = features.reshape((features.shape[0], 7 * 7 * 512))
    predictions = pd.DataFrame(Model1.predict_proba(features)).max(axis = 1)
    predictions = list(predictions)
    
    table["max_proba"] = predictions
    
    return table;

#permet d'eliminer les images qui ne sont pas oiseaux en se basant sur la taille et la distance des imagettes
def filtre_quantile(table,coef_filtre,height=2448,width=3264):
    ###Implemente ici avant de mettre dans fonction
    

    #definir les coefficients
    a_min=coef_filtre.iloc[0,1]
    b_min=coef_filtre.iloc[1,1]
    c_min=coef_filtre.iloc[2,1]
    c_min=round(c_min,5)
    d_min=coef_filtre.iloc[3,1]

    a_max=coef_filtre.iloc[0,2]
    b_max=coef_filtre.iloc[1,2]
    c_max=coef_filtre.iloc[2,2]
    d_max=coef_filtre.iloc[3,2]


    table=table.rename(columns={0: "imagetteName", 1: "xmin", 2 : "xmax", 3 : "ymin" , 4 : "ymax" })
    table["height"]=height
    table["width"]=width
    table.iloc[:,1:]=table.iloc[:,1:].astype(int)
    table["ycenter"]=(table["ymin"]+table["ymax"])/2
    table["ystd"]=table["ycenter"]/table["height"]
    ystd=table["ystd"]
    table["heightImagette"]=table["ymax"]-table["ymin"]
    table["heightImagetteStd"]=table["heightImagette"]/table["height"]
    table["logHIS"]=np.log(table["heightImagetteStd"])

    table["logHIS_pred_low"]= a_min+b_min*ystd+c_min*ystd**2+d_min*ystd**3
    table["logHIS_pred_high"]= a_max+b_max*ystd+c_max*ystd**2+d_max*ystd**3
    table["PossibleBird"]=True



    liste_possible_birds=[]
    for i in range(len(table)):
        logHIS=table["logHIS"].iloc[i]
        logHIS_high=table["logHIS_pred_high"].iloc[i]
        logHIS_low=table["logHIS_pred_low"].iloc[i]
        #possiblebird
        possible_birds=( logHIS< 0.8*logHIS_high) and (logHIS > 1.2*logHIS_low)
        liste_possible_birds.append(possible_birds)
        #table["PossibleBird"].iloc[i]=( logHIS< 0.8*logHIS_high) and (logHIS > 1.2*logHIS_low)
    table["PossibleBird"]=liste_possible_birds
    
    
    
    
    

    table=table[table['PossibleBird']==True]

    to_drop=['height', 'width',
       'ycenter', 'ystd', 'heightImagette', 'heightImagetteStd', 'logHIS',
       'logHIS_pred_low', 'logHIS_pred_high', 'PossibleBird']

    table.drop(to_drop,axis=1,inplace=True)
    index_possible_birds=list(table.index)
    
    
    return table,index_possible_birds




def DirectAnalysis_filtre_quantile (imageB,
                    name2,cnts,
                    maxAnalDL, # can be set to -1 to analyse everything
                    CNNmodel,labels,filtre_RL,
                    x_pix_max,y_pix_max,x_pix_min,y_pix_min
                    ,coef_filtre,height=2448,width=3264) : 
    ## Permet d'enregistrer et de tracer les contours repérés plus haut
    ## les images sont stockées dans le fichier "XXX.JPG_subImages"
#    cv2.imwrite(subImagesDir2+"diffsTrRaw"+timeStamp2+".JPG",thresh)


    # open the output CSV file for writing
    
    batchImages = []
    table = []
    imageSize= 28
#    np.empty((1,5), dtype = "int")  
    
    imageRectangles = imageB.copy()
    for ic in range(0,len(cnts)):
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    
        #
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        name = (os.path.split(name2)[-1]).split(".")[0]
        name = name + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        if( (f.xmax-f.xmin)<x_pix_max and (f.ymax-f.ymin)<y_pix_max # birds should less than 500 pixels wide and 350 high
           and (f.xmax-f.xmin)>x_pix_min and (f.ymax-f.ymin)>y_pix_min): # according to distribution in annotations
            subI, o, d, imageRectanglesB = GetSquareSubset(imageB,f,verbose=False)
            subI = RecenterImage(subI,o)
            subI = cv2.resize(subI,(imageSize,imageSize))
            subI = np.expand_dims(subI, axis=0)
            # subI = preprocess_input(subI)
            batchImages.append(subI)
            table.append(np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5)))
#            table1 = np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5))

    table = pd.DataFrame(np.vstack(table))
    table,index_possible_birds=filtre_quantile(table,coef_filtre,height=2448,width=3264)
    
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    #on enleve le filtre qui ne fonctionne pas
 
    # todo : add max_probas to out table in addition to results from DL
    #        and keep all "imagettes" in table
    
    #table = table.iloc[list(iOrderProba)[:min(len(cnts),maxAnalDL)],:]
    #batchImages = [batchImages[i] for i in list(iOrderProba)[:min(len(cnts),maxAnalDL)]]
    
    batchImages_filtre = [batchImages[i] for i in index_possible_birds]
    
    batchImages_filtre = np.vstack(batchImages_filtre)
    #print("features extraction")
    
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    features=preprocess_input(np.array(batchImages_filtre))
    features=features.reshape((features.shape[0], 28,28,3))
    estimates = CNNmodel.predict(features)
    for i in labels:
        table[i]=0
        
        
        
    
    colonne=0    
    for categorie in labels:
        for l in range(len(table)):
            table[categorie].iloc[l]=estimates[l,colonne]
        colonne+=1
        
    for i in range(len(table)):   
        xmin=int(table.iloc[i,1])
        xmax=int(table.iloc[i,2])
        ymin=int(table.iloc[i,3])
        ymax=int(table.iloc[i,4])
        #cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2) 
        max_col=np.amax(table.iloc[i,5:],axis=0)
        #if table.iloc[i,5]>=max_col:
        if table.iloc[i,10]==max_col:
            cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2) 
        elif table.iloc[i,5]==max_col:
            cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 255, 255), 2) 
            
        else:
            cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 
    
    #cv2.imwrite("testingInputs/type_oiseau.jpg", imageRectangles)    
    

    #return table,estimates
    #max_probas = estimates.argmax(axis=1)
    return table,imageRectangles,batchImages_filtre,batchImages;   

def agrandissement(xmin,xmax,zoom):
    distance=xmax-xmin
    centre=(xmax+xmin)/2
    xmax=centre+(distance*zoom)/2
    xmin=centre-(distance*zoom)/2
    
    return xmin,xmax
    

def find_square (imageB,intervalle,zoom,
                    name2,cnts,
                    maxAnalDL, # can be set to -1 to analyse everything
                    CNNmodel,labels,filtre_RL,
                    x_pix_max,y_pix_max,x_pix_min,y_pix_min
                    ,coef_filtre,height=2448,width=3264,table_add=pd.read_csv("testingInputs/oiseau_lab_Alex.csv")) : 
    ## Permet d'enregistrer et de tracer les contours repérés plus haut
    ## les images sont stockées dans le fichier "XXX.JPG_subImages"
#    cv2.imwrite(subImagesDir2+"diffsTrRaw"+timeStamp2+".JPG",thresh)

    
    # open the output CSV file for writing
    annontation_reduit=(table_add.iloc[:,6:12]).drop("index",axis=1)
    
    batchImages = []
    liste_table = []
    imageSize= 28
#    np.empty((1,5), dtype = "int")  
    
    imageRectangles = imageB.copy()
    for ic in range(0,len(cnts)):
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    
        #
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        #name = (os.path.split(name2)[-1]).split(".")[0]
        #name = name + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")

   
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   

        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif
        if( (f.xmax-f.xmin)<x_pix_max and (f.ymax-f.ymin)<y_pix_max # birds should less than 500 pixels wide and 350 high
           and (f.xmax-f.xmin)>x_pix_min and (f.ymax-f.ymin)>y_pix_min): # according to distribution in annotations
            subI, o, d, imageRectanglesB = GetSquareSubset(imageB,f,verbose=False)
            subI = RecenterImage(subI,o)
            subI = cv2.resize(subI,(imageSize,imageSize))
            subI = np.expand_dims(subI, axis=0)
            # subI = preprocess_input(subI)
            batchImages.append(subI)
            
            liste_table.append(np.array([[f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))
            
            #table.append(np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5)))
#            table1 = np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5))
    
            #   cv2.rectangle(imageRectangles, (o.xmin,o.ymin), (o.xmax,o.ymax), (255, 0, 0), 2)     
        #écriture des images
#        cv2.imwrite("images_carre/"+h.filename[:-4]+".JPG",img1)
        
    
            
    table = pd.DataFrame(np.vstack(liste_table))
    table,index_possible_birds=filtre_quantile(table,coef_filtre,height=2448,width=3264)
    #table=table.rename(columns={0: "imagetteName", 1: "xmin", 2 : "xmax", 3 : "ymin" , 4 : "ymax" })
    
    #Affecter le max et le min  renommer les variables
    liste_xmax=[]
    liste_xmin=[]
    liste_ymax=[]
    liste_ymin=[]
    for i in range(len(table)):
        XMAX=table["xmax"].iloc[i]
        XMIN=table["xmin"].iloc[i]
        YMAX=table["ymax"].iloc[i]
        YMIN=table["ymin"].iloc[i] 
        
        #MAX=table["xmax"].iloc[i]=agrandissement(XMIN,XMAX,zoom)
        largeur_min,largeur_max=agrandissement(XMIN,XMAX,zoom)
        largeur_min=int(round(largeur_min))
        largeur_max=int(round(largeur_max))
        liste_xmin.append(largeur_min)
        liste_xmax.append(largeur_max)
        
        profondeur_min,profondeur_max=agrandissement(YMIN,YMAX,zoom)
        profondeur_min=int(round(profondeur_min))
        profondeur_max=int(round(profondeur_max))
        liste_ymin.append(profondeur_min)
        liste_ymax.append(profondeur_max)
        
        
    #On ajoute les éléments de la dataframe provenant des anotations de Alex
    liste_xmax=liste_xmax
    liste_xmin=liste_xmin
    liste_ymax=liste_ymax
    liste_ymin=liste_ymin
    
 
    
    table["xmax"]=liste_xmax
    table["xmin"]=liste_xmin
    table["ymax"]=liste_ymax
    table["ymin"]=liste_ymin
    
    table=pd.concat([table,annontation_reduit])
    
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    #on enleve le filtre qui ne fonctionne pas
 
    # todo : add max_probas to out table in addition to results from DL
    #        and keep all "imagettes" in table
    
    #table = table.iloc[list(iOrderProba)[:min(len(cnts),maxAnalDL)],:]
    #batchImages = [batchImages[i] for i in list(iOrderProba)[:min(len(cnts),maxAnalDL)]]
    
    batchImages_filtre = [batchImages[i] for i in (index_possible_birds+list(range(26)))]
    
    batchImages_filtre = np.vstack(batchImages_filtre)
    #print("features extraction")
    
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    features=preprocess_input(np.array(batchImages_filtre))
    features=features.reshape((features.shape[0], 28,28,3))
    estimates = CNNmodel.predict(features)
    for i in labels:
        table[i]=0
        
        
        
    
    colonne=0    
    for categorie in labels:
        for l in range(len(table)):
            table[categorie].iloc[l]=estimates[l,colonne]
        colonne+=1
        
    for i in range(len(table)):   
        xmin=int(table.iloc[i,0])
        xmax=int(table.iloc[i,1])
        ymin=int(table.iloc[i,2])
        ymax=int(table.iloc[i,3])
        #cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2) 
        max_col=np.amax(table.iloc[i,5:],axis=0)
        #if table.iloc[i,5]>=max_col:
        if (i>intervalle[0]) and (i<intervalle[1]):
            cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)
            if table.iloc[i,10]==max_col:
                cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2) 
            elif table.iloc[i,5]==max_col:
                cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 255, 255), 2) 
            
            else:
                cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 
            
    cv2.imwrite("/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/test.jpg",imageRectangles)
        
    return table,imageRectangles,batchImages_filtre,batchImages;
















def filtre_light(imageA,imageB,contrast=-5,blockSize=51,blurFact = 25):
    img2 = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    img1 = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
    blurFact = blurFact
    absDiff2 = cv2.absdiff(img1, img2)
    diff = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
    th2 = cv2.adaptiveThreshold(src=diff,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
                                thresholdType=cv2.THRESH_BINARY,blockSize=blockSize,C=contrast) #c=-30 pour la cam de chasse adaptation de C à histogram de la photo ?

    th2Blur=cv2.GaussianBlur(th2,(blurFact,blurFact),sigmaX=0)
    th2BlurTh = cv2.adaptiveThreshold(src=th2Blur,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
            thresholdType=cv2.THRESH_BINARY,blockSize=blockSize,C=contrast) # adaptation de C à histogram de la photo ?
    threshS=th2BlurTh

        # defines corresponding regions of change
    cnts = cv2.findContours(threshS.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    return cnts



























#DISPLAY SQAURE IN A INTERVAL
def find_square_inverse (imageB,intervalle,zoom,
                    name2,cnts,
                    maxAnalDL, # can be set to -1 to analyse everything
                    CNNmodel,labels,filtre_RL,
                    x_pix_max,y_pix_max,x_pix_min,y_pix_min
                    ,coef_filtre,height=2448,width=3264,table_add=pd.read_csv("testingInputs/oiseau_lab_Alex.csv")) : 
    ## Permet d'enregistrer et de tracer les contours repérés plus haut
    ## les images sont stockées dans le fichier "XXX.JPG_subImages"
#    cv2.imwrite(subImagesDir2+"diffsTrRaw"+timeStamp2+".JPG",thresh)

    #'xmin', 'ymin', 'xmax',
    #  'ymax'
    
    # open the output CSV file for writing
    annontation_reduit=(table_add.iloc[:,6:12]).drop("index",axis=1)
    
    batchImages = []
    liste_table = []
    imageSize= 28
#    np.empty((1,5), dtype = "int")  
    
    imageRectangles = imageB.copy()
    for ic in range(0,len(cnts)):
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    
        #
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        name = (os.path.split(name2)[-1]).split(".")[0]
        name = name + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")

   
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   

        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif
        if( (f.xmax-f.xmin)<x_pix_max and (f.ymax-f.ymin)<y_pix_max # birds should less than 500 pixels wide and 350 high
           and (f.xmax-f.xmin)>x_pix_min and (f.ymax-f.ymin)>y_pix_min): # according to distribution in annotations
            subI, o, d, imageRectanglesB = GetSquareSubset(imageB,f,verbose=False)
            subI = RecenterImage(subI,o)
            subI = cv2.resize(subI,(imageSize,imageSize))
            subI = np.expand_dims(subI, axis=0)
            # subI = preprocess_input(subI)
            batchImages.append(subI)
            
            liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))
            
           
        #Pour l'annotation
        #Il va devoir comprendre qu'on fait une itération
        """annontation_reduit
        f.xmin, f.xmax, f.ymin, f.ymax = annontation_reduit["xmin"], annontation_reduit["xmax"], annontation_reduit["ymin"], annontation_reduit["ymax"]
        subI, o, d, imageRectanglesB = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        subI = np.expand_dims(subI, axis=0)
        # subI = preprocess_input(subI)
        batchImages.append(subI)"""
            
         #####Ajouter ici les mêmes batchs mais pour les annotations avec la même procédure que ci-dessus.
         #Non il faut faire le concat tout en haut .... puisqu'ensuite on va sélectionner l'index pour possible birds et 
         #pour les oiseaux
         
            #table.append(np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5)))
#            table1 = np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5))
    
            #   cv2.rectangle(imageRectangles, (o.xmin,o.ymin), (o.xmax,o.ymax), (255, 0, 0), 2)     
        #écriture des images
#        cv2.imwrite("images_carre/"+h.filename[:-4]+".JPG",img1)
        
    
            
    table = pd.DataFrame(np.vstack(liste_table))
    table,index_possible_birds=filtre_quantile(table,coef_filtre,height=2448,width=3264)
    #table=table.rename(columns={0: "imagetteName", 1: "xmin", 2 : "xmax", 3 : "ymin" , 4 : "ymax" })
    
    #Affecter le max et le min  renommer les variables
    liste_xmax=[]
    liste_xmin=[]
    liste_ymax=[]
    liste_ymin=[]
    for i in range(len(table)):
        XMAX=table["xmax"].iloc[i]
        XMIN=table["xmin"].iloc[i]
        YMAX=table["ymax"].iloc[i]
        YMIN=table["ymin"].iloc[i] 
        
        #MAX=table["xmax"].iloc[i]=agrandissement(XMIN,XMAX,zoom)
        largeur_min,largeur_max=agrandissement(XMIN,XMAX,zoom)
        largeur_min=int(round(largeur_min))
        largeur_max=int(round(largeur_max))
        liste_xmin.append(largeur_min)
        liste_xmax.append(largeur_max)
        
        profondeur_min,profondeur_max=agrandissement(YMIN,YMAX,zoom)
        profondeur_min=int(round(profondeur_min))
        profondeur_max=int(round(profondeur_max))
        liste_ymin.append(profondeur_min)
        liste_ymax.append(profondeur_max)
        
        
    #On ajoute les éléments de la dataframe provenant des anotations de Alex
    liste_xmax=liste_xmax
    liste_xmin=liste_xmin
    liste_ymax=liste_ymax
    liste_ymin=liste_ymin
    
 
    
    table["xmax"]=liste_xmax
    table["xmin"]=liste_xmin
    table["ymax"]=liste_ymax
    table["ymin"]=liste_ymin
    
    #table=pd.concat([table,annontation_reduit])
    #table=annontation_reduit
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    #on enleve le filtre qui ne fonctionne pas
 
    # todo : add max_probas to out table in addition to results from DL
    #        and keep all "imagettes" in table
    
    #table = table.iloc[list(iOrderProba)[:min(len(cnts),maxAnalDL)],:]
    #batchImages = [batchImages[i] for i in list(iOrderProba)[:min(len(cnts),maxAnalDL)]]
    
    #batchImages_filtre = [batchImages[i] for i in (index_possible_birds+list(range(26)))]
    batchImages_filtre = [batchImages[i] for i in (index_possible_birds)]
    
    batchImages_filtre = np.vstack(batchImages_filtre)
    #print("features extraction")
    
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    features=preprocess_input(np.array(batchImages_filtre))
    features=features.reshape((features.shape[0], 28,28,3))
    estimates = CNNmodel.predict(features)
    for i in labels:
        table[i]=0
        
        
        
    
    colonne=0    
    for categorie in labels:
        for l in range(len(table)):
            table[categorie].iloc[l]=estimates[l,colonne]
        colonne+=1
        
    for i in range(len(table)):   
        xmin=int(table.iloc[i,1])
        xmax=int(table.iloc[i,2])
        ymin=int(table.iloc[i,3])
        ymax=int(table.iloc[i,4])
        #cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2) 
        max_col=np.amax(table.iloc[i,5:],axis=0)
        #if table.iloc[i,5]>=max_col:
        if (i>intervalle[0]) and (i<intervalle[1]):
            cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)
            if table.iloc[i,10]==max_col:
                cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2) 
            elif table.iloc[i,5]==max_col:
                cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 255, 255), 2) 
            
            else:
                cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 
        
    return table,imageRectangles,batchImages_filtre,batchImages;











def DirectAnalysis_double (imageB,
                    name2,cnts,
                    maxAnalDL, # can be set to -1 to analyse everything
                    CNNmodel,labels,filtre_RL,
                    x_pix_max,y_pix_max,x_pix_min,y_pix_min
                    ,coef_filtre,height=2448,width=3264) : 
    ## Permet d'enregistrer et de tracer les contours repérés plus haut
    ## les images sont stockées dans le fichier "XXX.JPG_subImages"
#    cv2.imwrite(subImagesDir2+"diffsTrRaw"+timeStamp2+".JPG",thresh)


    # open the output CSV file for writing
    
    batchImages = []
    table = []
    imageSize= 28
#    np.empty((1,5), dtype = "int")  
    
    imageRectangles = imageB.copy()
    for ic in range(0,len(cnts)):
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    
        #
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        name = (os.path.split(name2)[-1]).split(".")[0]
        name = name + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        if( (f.xmax-f.xmin)<x_pix_max and (f.ymax-f.ymin)<y_pix_max # birds should less than 500 pixels wide and 350 high
           and (f.xmax-f.xmin)>x_pix_min and (f.ymax-f.ymin)>y_pix_min): # according to distribution in annotations
            subI, o, d, imageRectanglesB = GetSquareSubset(imageB,f,verbose=False)
            subI = RecenterImage(subI,o)
            subI = cv2.resize(subI,(imageSize,imageSize))
            subI = np.expand_dims(subI, axis=0)
            subI = preprocess_input(subI)
            batchImages.append(subI)
            table.append(np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5)))
#            table1 = np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5))
    
            #   cv2.rectangle(imageRectangles, (o.xmin,o.ymin), (o.xmax,o.ymax), (255, 0, 0), 2)     
        #écriture des images
#        cv2.imwrite("images_carre/"+h.filename[:-4]+".JPG",img1)
        
    
            
    table = pd.DataFrame(np.vstack(table))
    table,index_possible_birds=filtre_quantile(table,coef_filtre,height=2448,width=3264)
    
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    #on enleve le filtre qui ne fonctionne pas
 
    # todo : add max_probas to out table in addition to results from DL
    #        and keep all "imagettes" in table
    
    #table = table.iloc[list(iOrderProba)[:min(len(cnts),maxAnalDL)],:]
    #batchImages = [batchImages[i] for i in list(iOrderProba)[:min(len(cnts),maxAnalDL)]]
    
    batchImages = [batchImages[i] for i in index_possible_birds]
    
    batchImages = np.vstack(batchImages)
    print("features extraction")
    
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    features=preprocess_input(np.array(batchImages))
    features=features.reshape((features.shape[0], 28,28,3))
    estimates = CNNmodel.predict(features)
    for i in labels:
        table[i]=0
        
        
        
    
    colonne=0    
    for categorie in labels:
        for l in range(len(table)):
            table[categorie].iloc[l]=estimates[l,colonne]
        colonne+=1
        
    for i in range(len(table)):   
        xmin=int(table.iloc[i,1])
        xmax=int(table.iloc[i,2])
        ymin=int(table.iloc[i,3])
        ymax=int(table.iloc[i,4])
        cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2) 
        max_col=np.amax(table.iloc[i,5:],axis=0)
        #if table.iloc[i,5]>=max_col:
        if table.iloc[i,6]==max_col:
            cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2) 
        elif table.iloc[i,5]==max_col:
            cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 255, 255), 2) 
            
    
    cv2.imwrite("/mnt/VegaSlowDataDisk/c3po_interface/bin/testingInputs/oiseau_pasOis.jpg", imageRectangles)    
    

    #return table,estimates
    #max_probas = estimates.argmax(axis=1)
    return table;           
    





def DirectAnalysis (imageB,
                    name2,cnts,
                    maxAnalDL, # can be set to -1 to analyse everything
                    CNNmodel,labels,filtre_RL,
                    x_pix_max,y_pix_max,x_pix_min,y_pix_min) : 
    ## Permet d'enregistrer et de tracer les contours repérés plus haut
    ## les images sont stockées dans le fichier "XXX.JPG_subImages"
#    cv2.imwrite(subImagesDir2+"diffsTrRaw"+timeStamp2+".JPG",thresh)


    # open the output CSV file for writing
    
    batchImages = []
    table = []
    imageSize= 28
#    np.empty((1,5), dtype = "int")  
    
    imageRectangles = imageB.copy()
    for ic in range(0,len(cnts)):
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        name = (os.path.split(name2)[-1]).split(".")[0]
        name = name + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        if( (f.xmax-f.xmin)<x_pix_max and (f.ymax-f.ymin)<y_pix_max # birds should less than 500 pixels wide and 350 high
           and (f.xmax-f.xmin)>x_pix_min and (f.ymax-f.ymin)>y_pix_min): # according to distribution in annotations
            subI, o, d, imageRectanglesB = GetSquareSubset(imageB,f,verbose=False)
            subI = RecenterImage(subI,o)
            subI = cv2.resize(subI,(imageSize,imageSize))
            subI = np.expand_dims(subI, axis=0)
            # subI = preprocess_input(subI)
            batchImages.append(subI)
            table.append(np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5)))
#            table1 = np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5))
    
            #   cv2.rectangle(imageRectangles, (o.xmin,o.ymin), (o.xmax,o.ymax), (255, 0, 0), 2)     
        #écriture des images
#        cv2.imwrite("images_carre/"+h.filename[:-4]+".JPG",img1)
        
    
            
    table = pd.DataFrame(np.vstack(table))
    
    
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    max_probas = np.amax(filtre_RL.predict_proba(np.array(table.iloc[:,1:], 
                                                          dtype = "float64")), axis = 1)
    iOrderProba = np.argsort(-max_probas)
    
    imageRectanglesPreSelect = imageB.copy()
    for i in range(0,len(table)) :
        f = pd.Series(dtype= "int64")
        f.xmin, f.xmax, f.ymin, f.ymax =  table.iloc[i,1:].astype(int)
        if max_probas[i]>0.7 :
            cv2.rectangle(imageRectanglesPreSelect, (f.xmin,f.ymin), (f.xmax,f.ymax), (255, 0, 0), 2)
        else:
            cv2.rectangle(imageRectanglesPreSelect, (f.xmin,f.ymin), (f.xmax,f.ymax), (0, 0, 255), 2)
    cv2.imwrite("testingInputs/imageRectanglesPreSelect.jpg", imageRectanglesPreSelect)
    
    # todo : add max_probas to out table in addition to results from DL
    #        and keep all "imagettes" in table
    table = table.iloc[list(iOrderProba)[:min(len(cnts),maxAnalDL)],:]
    batchImages = [batchImages[i] for i in list(iOrderProba)[:min(len(cnts),maxAnalDL)]]
    
    batchImages = np.vstack(batchImages)
    print(batchImages.shape,"batchshape")
    print("features extraction")
    
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    features=preprocess_input(np.array(batchImages))
    features=features.reshape((features.shape[0], 28,28,3))
    estimates = CNNmodel.predict(features)
    for i in labels:
        table[i]=0
    
    #return table,estimates
    #max_probas = estimates.argmax(axis=1)
       

    
    colonne=0    
    for categorie in labels:
        for l in range(len(table)):
            table[categorie].iloc[l]=estimates[l,colonne]
        colonne+=1
   
    """     
    for i in range(len(table)):   
        xmin=int(table.iloc[i,1])
        xmax=int(table.iloc[i,2])
        ymin=int(table.iloc[i,3])
        ymax=int(table.iloc[i,4])
        max_col=np.amax(table.iloc[:,5:][i],axis=0)[i]
        if out.iloc[i,10]<max_col[i]:
            cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2) 
        else:
            cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2) """
    
    cv2.imwrite("testingInputs/oiseau_filtre.jpg", imageRectangles)    
    return table,imageRectangles;



    
    # todo : add max_probas to out table in addition to results from DL
    #        and keep all "imagettes" in table
#    table = table.iloc[list(max_probas)[:min(len(cnts),maxAnalDL)],:]
#    
#    
#    batchImages = [batchImages[i] for i in list(max_probas)[:min(len(cnts),maxAnalDL)]]
#    
#    batchImages = np.vstack(batchImages)
#    print("features extraction")
#    features = CNNmodel.predict(batchImages, batch_size=4) # why 5 if only 4 proc on the pi...
#    features = features.reshape((features.shape[0], 7 * 7 * 512))
#    predictions = pd.DataFrame(Model1.predict_proba(features)).max(axis = 1)
#    predictions = list(predictions)
#    
#    table["max_proba"] = predictions
    




#np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2)).reshape((1,5)
#
#f = pd.Series([name, x,x+w, y, y+h]) 

#np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2)).reshape((1,5)  
def SaveDraw (imageA,imageB,kernel,subImagesDir1,subImagesDir2,thresh,name1,name2) : 
    ## Permet d'enregistrer et de tracer les contours repérés plus haut
    ## les images sont stockées dans le fichier "XXX.JPG_subImages"
#    cv2.imwrite(subImagesDir2+"diffsTrRaw"+timeStamp2+".JPG",thresh)

# aggregate small changes close from each others
    threshS = cv2.dilate(thresh,(3,3))
    threshS = cv2.erode(threshS,(3,3),iterations=1)
#    cv2.imwrite(subImagesDir2+"diffsTrAgg-"+timeStamp2+".JPG",threshS)

# defines corresponding regions of change
    cnts = cv2.findContours(threshS.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

# changing parts
#    chParts = cv2.bitwise_and(imageB,imageB,mask=thresh)
#    maskContours = np.zeros(imageB.shape, np.uint8)
    iSave = 1
#    imageRectanglesA = imageA.copy()
    imageRectanglesB = imageB.copy()
#    imageContours = imageB.copy()
    
    for ic in range(0,len(cnts)):
        
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        f = pd.Series()
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        
        ## Couper en trois zones 
        if(Filtre == "zones"):
            if( 1200 > f.ymax):
                if( (f.xmax-f.xmin) > 20 and  (f.ymax-f.ymin) > 14):
                    ## Sérvira à centrer les images 
                    subI, o, d, imageRectanglesB = GetSquareSubset(imageRectanglesB,f,verbose=True)
                    subI = RecenterImage(subI,o)
#                    cv2.drawContours(imageContours,[cnts[ic]],0, (0,255,0), 3)
                    cv2.imwrite(subImagesDir1+"diffs_"+str(iSave)+"_"+name2.split("/")[-1],subI)
                    
            if(1800 > f.ymax > 1200):
                if(imageB.shape[1]/3 > (f.xmax-f.xmin) > 55 and (f.ymax-f.ymin) > 50):
                    subI, o, d, imageRectanglesB = GetSquareSubset(imageRectanglesB,f,verbose=True)       
                    subI = RecenterImage(subI,o)
#                    cv2.drawContours(imageContours,[cnts[ic]],0, (0,255,0), 3)
                    cv2.imwrite(subImagesDir1+"diffs_"+str(iSave)+"_"+name2.split("/")[-1],subI)
                    
            if(f.ymax > 1800):
                if(imageB.shape[1]/2 > (f.xmax-f.xmin) > 200 and imageB.shape[0]/2 > (f.ymax-f.ymin) > 150):
                    subI, o, d, imageRectanglesB = GetSquareSubset(imageRectanglesB,f,verbose=True)
                    subI = RecenterImage(subI,o)
#                   cv2.drawContours(imageContours,[cnts[ic]],0, (0,255,0), 3)
                    cv2.imwrite(subImagesDir1+"diffs_"+str(iSave)+"_"+name2.split("/")[-1],subI)
        
    #        else:
    #            subI, o, d, imageRectanglesB = GetSquareSubset(imageRectanglesB,f,verbose=True)       
    #            subI = RecenterImage(subI,o)
    #            cv2.drawContours(imageContours,[cnts[ic]],0, (0,255,0), 3)
    #            cv2.imwrite(subImagesDir+"diffs"+str(iSave)+"_"+name2+"_"+timeStamp2+".JPG",subI)
    #    
            iSave = iSave+1
        
        ## Filtre avec un Model RandomForest entrainer sur les xmin, xmax, ymin, ymax 
        ## on rajoute aussi un filtre sur la taille des imagettes 
        if(Filtre == "RandomForest"):
            if(f.ymax > 1100):
                if(imageB.shape[1]/2 > (f.xmax-f.xmin) > 55 and imageB.shape[0]/2 > (f.ymax-f.ymin) > 50):
                    TestX = np.array([f.xmin,f.ymin,f.xmax,f.ymax]).reshape(1,4)
                    y_pred = Model.predict(TestX)
                    if(int(y_pred) == 0):
                        subI, o, d, imageRectanglesB = GetSquareSubset(imageRectanglesB,f,verbose=True)
                        subI = RecenterImage(subI,o)
#                        cv2.drawContours(imageContours,[cnts[ic]],0, (0,255,0), 3) cv2.imwrite(subImagesDir1+"diffs_"+str(iSave)+"_"+name2.split("/")[-1],subI)
                        cv2.imwrite(subImagesDir1+"diffs_"+str(iSave)+"_"+name2.split("/")[-1],subI)
                        iSave = iSave+1
            
            else:
                if(imageB.shape[1]/2 > (f.xmax-f.xmin) > 15 and imageB.shape[0]/2 > (f.ymax-f.ymin) > 15):
                    TestX = np.array([f.xmin,f.ymin,f.xmax,f.ymax]).reshape(1,4)
                    y_pred = Model.predict(TestX)
                    if(int(y_pred) == 0):
                        subI, o, d, imageRectanglesB = GetSquareSubset(imageRectanglesB,f,verbose=True)
                        subI = RecenterImage(subI,o)
#                        cv2.drawContours(imageContours,[cnts[ic]],0, (0,255,0), 3)
                        cv2.imwrite(subImagesDir1+"diffs_"+str(iSave)+"_"+name2.split("/")[-1],subI)
                        iSave = iSave+1
            
        if(Filtre == "NoFiltre"):
            if( (f.xmax-f.xmin) < 500 and  (f.ymax-f.ymin) < 350):
                subI, o, d, imageRectanglesB = GetSquareSubset(imageRectanglesB,f,verbose=True)
                subI = RecenterImage(subI,o)
#               cv2.drawContours(imageContours,[cnts[ic]],0, (0,255,0), 3)
                cv2.imwrite(subImagesDir1+"diffs_"+str(iSave)+"_"+name2.split("/")[-1],subI)
                iSave = iSave+1
                
    cv2.imwrite(subImagesDir2+"diffs_"+str(iSave)+"_"+name2.split("/")[-1],imageRectanglesB)
#    cv2.imwrite(subImagesDir2+"drawContours-"+name2+"_"+timeStamp2+".JPG",imageContours)

    return;

def CapturePi() :
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera() # starts increase of 70mA
    # allow the camera to warmup
    time.sleep(1)
    rawCapture = PiRGBArray(camera)

    # grab an image from the camera
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
    
    # enregistrement de l'image    
    camera.close()
    # Inverse l'image
#    image = imutils.rotate(image, 180)
    timeStamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    return image, timeStamp;

def CaptureLaptop() :
    # initialize the camera and grab a reference to the raw camera capture
    camera = cv2.VideoCapture(0) 
    # allow the camera to warmup
    time.sleep(0.1)
    
    _, image = camera.read()   
    
    # enregistrement de l'image
    # cv2.imwrite(subImagesDir+"image_"+time.strftime("%Y-%m-%d_%H-%M-%S")+".jpg",image)   
    camera.release()
#    image = imutils.rotate(image, 180)
    timeStamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    cv2.destroyAllWindows()
    
    return image, timeStamp;

# Verifie les bords de l'image pour la découpe    
def accountForBorder(val,maxval):
#    """ tests
#    >>> accountForBorder(-3,10)
#    (0, 3)
#    >>> accountForBorder(15,10)
#    (10, 5)
#    >>> accountForBorder(10,15)
#    (10, 0)
#    >>> accountForBorder(0,0)
#    (0, 0)
#    >>> accountForBorder(10,10)
#    (10, 0)
#    """ 
    if(val<0):
        cut = 0-val
        val = 0
    elif(val>maxval):
        cut = val - maxval
        val = maxval    
    else:
        cut = 0
    return val,cut

# découpe les subimages à éxtraire
def GetSquareSubset(img,h,verbose=True, xml = False):
    
   
    # détermine le plus grand côte du carré
    d = max(h.ymax-h.ymin,h.xmax-h.xmin)
      
    # détermine le centre du carré
    xcent = int(round((h.xmax-h.xmin)/2)) + h.xmin
    ycent = int(round((h.ymax-h.ymin)/2)) + h.ymin
    
    # new corners
    hd = int(math.ceil(d/2)) # half distance
    o = pd.Series(dtype='float64')
    o.xmin = xcent- hd
    o.xmax = xcent+ hd
    o.ymin = ycent- hd
    o.ymax = ycent+ hd
    """   
    print(o.xmin,o.xmax)
    print(o.ymin,o.ymax)
    """
    # check we are not further than the image borders
    # get the min/max accounting for image border + n cutted pixels: o.(x|y)(min|max)cut
    o.xmin,o.xmincut = accountForBorder(o.xmin,img.shape[1])
    o.xmax,o.xmaxcut = accountForBorder(o.xmax,img.shape[1])
    o.ymin,o.ymincut = accountForBorder(o.ymin,img.shape[0])
    o.ymax,o.ymaxcut = accountForBorder(o.ymax,img.shape[0])
    """
    print(o.xmin,o.xmax)
    print(o.ymin,o.ymax)
    """
    if(verbose):
        # déssine le carré
        cv2.rectangle(img, (o.xmin,o.ymin), (o.xmax,o.ymax), (255, 0, 0), 2)     
        #écriture des images
#        cv2.imwrite("images_carre/"+h.filename[:-4]+".JPG",img1)
        # add squares/numbers
    
    if(xml == True):
        # couper l'image
        subI = img[o.ymin:o.ymax,o.xmin:o.xmax]
    else:
        # couper l'image
        # subI = imageB[o.ymin:o.ymax,o.xmin:o.xmax]
        subI = img[o.ymin:o.ymax,o.xmin:o.xmax]
   
    
    return subI, o, d, img

# Recentrage des subimages en cas de coupe par le bord de l'image
# ajoute la couleur moyenne sur les côtés
def RecenterImage(subI,o):
    
    h,l,r=subI.shape #on sait que r=3 (3 channels)
    
    # add to the dimension the dimensions of the cuts (due to image borders)
    h = h + o.ymincut + o.ymaxcut
    l = l + o.xmincut + o.xmaxcut

    t= np.full((h, l, 3), fill_value=int(round(subI.mean())),dtype=np.uint8) # black image the size of the final thing

    t[o.ymincut:(h-o.ymaxcut),o.xmincut:(l-o.xmaxcut)] = subI
    
    return t

# Crée le bon format du data
def Rightdataform(path):
    #list des fichiers .JPG
    df = glob.glob(path)
    
    labels = np.array([], ndmin = 1)
    d = np.array([], ndmin = 3)

    for i in df:
        
        if("PasOiseau" in i):
            labels = np.append(labels,[0])
        else:
            labels = np.append(labels,[1])
            
        T = cv2.imread(i) 
        a = T.shape
        a = a[0]
        b = np.array(T[0,:,:],ndmin = 2)
        for j in range(a-1):
            b = np.concatenate((b,T[j+1,...]),axis=1)
        
        b = np.array(b, ndmin = 3)
        if(d.size != 0):
            d = np.concatenate((d,b))
        else:
            d = np.array(b, ndmin = 3)
    labels = np.array(labels, dtype ="uint8" )
    
    return d, labels, df

## Load data split for predict.py  and Train.py scripts
def load_data_split(splitPath):
	# initialize the data and labels
	data = []
	labels = []
	names = []

	# loop over the rows in the data split file
	for row in open(splitPath):
		# extract the class label and features from the row
		row = row.strip().split(",")
		label = row[1]
		name = row[0]
		features = np.array(row[2:], dtype="float")

		# update the data and label lists
		data.append(features)
		labels.append(label)
		names.append(name)

	# convert the data and labels to NumPy arrays
	data = np.array(data)
	labels = np.array(labels)

	# return a tuple of the data and labels
	return (data, labels, names)


def New_load_data_split(splitPath):
	# initialize the data and labels
	data = []
	names = []

	# loop over the rows in the data split file
	for row in open(splitPath):
		# extract the class label and features from the row
		row = row.strip().split(",")
		name = row[0]
		features = np.array(row[1:], dtype="float")

		# update the data and label lists
		data.append(features)
		names.append(name)

	# convert the data and labels to NumPy arrays
	data = np.array(data)

	# return a tuple of the data and labels
	return (data, names)


## Button for annotation interface 
    
def newlabels_Oiseau():
    NewLabels.append(0)
    root.destroy()
    return 

def newlabels_PasOiseau():
    NewLabels.append(1)
    root.destroy()
    return 
def Undo():
    NewLabels.pop()
    root.destroy()
    return 

#
#if __name__ == "__main__":
#    import doctest
#    doctest.testmod()


## problème à résoudre avec classifier dans l'argument de fonction 
def VGG_glm(image,modelVGG,classifier,preprocess,batchSize):
    # ajout d'une vérification des dimensions conditionnant np.expand 
    image = np.expand_dims(image, axis=0) # fait un vecteur d'images
    image = preprocess(image)  # mise en forme pour par exemple VGG imagenet
    features = modelVGG.predict(image, batch_size=batchSize) # 
    features = features.reshape((features.shape[0], 7 * 7 * 512))
    
    # vec = ",".join([str(v) for v in features[0]])
    # csvTemp = "/tmp/VGG_glm.csv" # ajouter un timestamp a l'avenir
    # csv = open(csvTemp, "w") 
    # csv.write("{}\n".format(vec))
    
    pred = classifier(features)
    
    return pred

# ex: 
#image = load_img('diffs133_img_7786_2018-05-22_15-30-18.JPG', target_size=(224, 224))
#image = img_to_array(image)
#def bidon(argumentBidon):
#    return 1
#out = VGG_glm(image,model,bidon,imagenet_utils.preprocess_input,config.BATCH_SIZE)



def to_polygon(line):
    xmin=line["xmin"]
    xmax=line["xmax"]
    ymin=line["ymin"]
    ymax=line["ymax"]
    polygon=Polygon( [(xmin,ymin ),(xmin,ymax),(xmax,ymax),(xmax,ymin)])
    return polygon



"""
neurone_features="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/z1.3"
coef_filtre=pd.read_csv("testingInputs/coefs_filtre_RQ.csv")
"""




#This function place the square generate equal to the annotation
def place_generate_sqaure(imageB,cnts):

    
    image_annote=imageB.copy()
    batchImages = []
    liste_table = []
    imageSize= 28
    #On récupère les coordonnées des pixels différent par différence
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        #name = (os.path.split(name2)[-1]).split(".")[0]
        #name = name + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   
        #
        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif

        subI, o, d, imageRectangles = GetSquareSubset(image_annote,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        #liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))
        liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))
        
    table_full = pd.DataFrame(np.vstack(liste_table))
    
    
    #Ce serait bien de rajouter rename ici !!! Si ça n'entraine pas de bug
    #table_full = table_full.rename(columns={0: 'imagettename', 1: 'xmin', 2: 'xmax', 3: 'ymin', 4: 'ymax'})
    table_full = table_full.rename(columns={ 0: 'xmin', 1: 'xmax', 2: 'ymin', 3: 'ymax'})
    #table_full.iloc[:,1:]=table_full.iloc[:,1:].astype(int)
    table_full=table_full.astype(int)


    
    for i in range(len(table_full)):
        xmin=table_full["xmin"].iloc[i]
        ymin=table_full["ymin"].iloc[i]
        xmax=table_full["xmax"].iloc[i]
        ymax=table_full["ymax"].iloc[i]
        cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 255,0), 2) 

    cv2.imwrite("/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/annote.jpg",imageRectangles)
    cv2.imwrite("/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/vierge.jpg",imageB)












    

    


#Cette fonction permet les mêmes résultats que celle ci_dessous mais sur une boucle en l'occurence dans le dossier 0 du Pi
#Dans cette fonction est prévue des options encore non déployés faire des carrés pour les différences générées
#Les estimations pour chaque différence    
#Les deux premières lignes ne sont pas vouées à rester

#neurone_features="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/z1.3"
#coef_filtre=pd.read_csv("/mnt/VegaSlowDataDisk/c3po_interface/bin/testingInputs/coefs_filtre_RQ.csv")



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


coef_filtre=pd.read_csv("/mnt/VegaSlowDataDisk/c3po_interface/bin/testingInputs/coefs_filtre_RQ.csv")


    
  
    










def area_square(x_min,x_max,y_min,y_max):
    

    
    profondeur=y_max-y_min
    largeur=x_max-x_min
    surface=largeur*profondeur
    
    return surface




def area_intersection(x_min_gen,x_max_gen,y_min_gen,y_max_gen,  x_min_anote,x_max_anote,y_min_anote,y_max_anote   ):
    
    min_xmax=min(x_max_gen,x_max_anote)
    max_xmin=max(x_min_gen,x_min_anote)
    min_ymax=min(y_max_gen,y_max_anote) 
    max_ymin=max(y_min_gen,y_min_anote)    
    
    largeur=max(0,min_xmax-max_xmin)
    profondeur=max(0,min_ymax-max_ymin)
    area_intersection=largeur*profondeur
    return area_intersection


















def birds_square_light(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=19,blurFact=15,filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
    debut = time.time()
    
    dictionnaire_conversion={}
    dictionnaire_conversion[0]="autre"
    dictionnaire_conversion[1]="chevreuil"
    dictionnaire_conversion[2]="corneille"
    dictionnaire_conversion[3]="faisan"
    dictionnaire_conversion[4]="lapin"
    dictionnaire_conversion[5]="pigeon"
    
    
    
    
    
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    
    
    
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    imagettes=to_reference_labels (imagettes,"classe")
    
    imagettes=imagettes[ (imagettes["classe"]!="oiseau") & (imagettes["classe"]!="autre") & (imagettes["classe"]!="pie") 
                        & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") & (imagettes["classe"]!="sanglier") 
                        & (imagettes["classe"]!="cheval") ]
    folder_choosen="."+ folder
    imagettes_PI_0=imagettes[(imagettes["path"]==folder_choosen) ]
    #Attention ça ne fonction qu'avec le PI_0 pour les autres dossiers il faut rajouter d'autres classes 
    imagettes_PI_0=imagettes_PI_0[   (imagettes_PI_0["classe"]!="ground") & (imagettes_PI_0["classe"]!="incertain")     ]    
    imagettes1=imagettes_PI_0[imagettes_PI_0["filename"]==name_test]
    if len(imagettes1["classe"].unique())>1:
        print("attention il y a des oiseaux de différents type ici le code n'est pas adapté à cette situation" )
    nom_classe=imagettes1["classe"].iloc[0]
    #nom_classe=list(imagettes1["classe"].unique())
    
    #On va comparer les images de ici et probablement test unitaires
    batchImages = []
    liste_table = []
    imageSize= 28
    cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    #On récupère les coordonnées des pixels différent par différence
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        #name = (os.path.split(name_test)[-1]).split(".")[0]
        #name = name + "_" + str(ic) + ".JPG"
        #name = nom_classe + "_" + str(ic) + ".JPG"
        name = "TOTO" + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   

        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif

        subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))
     
    table_full = pd.DataFrame(np.vstack(liste_table))
    
    
    #Ce serait bien de rajouter rename ici !!! Si ça n'entraine pas de bug
    table_full = table_full.rename(columns={0: 'imagettename', 1: 'xmin', 2: 'xmax', 3: 'ymin', 4: 'ymax'})
    table_full.iloc[:,1:]=table_full.iloc[:,1:].astype(int)


    batchImages_stack = np.vstack(batchImages)
    batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
    
    #Il faudra penser à soulever une expetion si rien trouver à l'isuu du filtre en particulier pour la ligne
    #fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
    #image_test image_2019-06-14_15-46-54.jpg image_ref image_2019-06-14_15-46-38.jpg
    """
    table_quantile,index_possible_birds=fn.filtre_quantile(table_full,coef_filtre,height=2448,width=3264)
    table_filtre_RL=table_quantile.copy()
    table_filtre_RL["possible_bird"]=fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
    table_filtre_RL=(table_filtre_RL[table_filtre_RL["possible_bird"]=="O"])
    p_bird=table_filtre_RL.index
    table_filtre_RL.drop("possible_bird",axis=1,inplace=True)     
    index_possible_birds=list(set(index_possible_birds).intersection(p_bird)) 
    #batchImages_filtre = [batchImages_stack_reshape[i] for i in (index_possible_birds)]
    """

    
    to_drop=['path', 'filename', 'width', 'height', 'index']

    im=imagettes1.drop(to_drop,axis=1)
    col=list(im.columns)
    col = col[-1:] + col[:-1]


    annontation_reduit=im[col]
    print("test de débugage")
    print("annontation_reduit",annontation_reduit)
    
    
    #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
    #generate_square=table.iloc[:,1:5]
    if filtre_choice=="No_filtre":
        generate_square=table_full.iloc[:,1:5]
    elif filtre_choice=="quantile_filtre":
        generate_square=table_quantile.iloc[:,1:5]
    elif filtre_choice=="RL_filtre":
        generate_square=table_filtre_RL.iloc[:,1:5]
    
    #liste_carre_diff=generate_square[['xmin', 'ymin', 'xmax', 'ymax']].apply(to_polygon,axis=1)
    

    
    
    xmin_gen=generate_square["xmin"]
    xmax_gen=generate_square["xmax"]
    ymin_gen=generate_square["ymin"]
    ymax_gen=generate_square["ymax"]
    

    
   
    
    liste_DIFF=[]
    
    
    for i in range(len(annontation_reduit)):
        x_min_anote=annontation_reduit["xmin"].iloc[i]
        x_max_anote=annontation_reduit["xmax"].iloc[i]
        y_min_anote=annontation_reduit["ymin"].iloc[i]
        y_max_anote=annontation_reduit["ymax"].iloc[i]
 
        ln_square_gen=len(xmin_gen)
    
        zip_xmin_anote=[x_min_anote]*ln_square_gen
        zip_xmax_anote=[x_max_anote]*ln_square_gen
        zip_ymin_anote=[y_min_anote]*ln_square_gen
        zip_ymax_anote=[y_max_anote]*ln_square_gen
    
        
        
        liste_intersection=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
        max_intersection=max(liste_intersection)

        #Il manque le if
        proportion_maximum=max_intersection/area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
        if proportion_maximum>coverage_threshold:
            #Il faudrait rajouter une boucle while tan que max_intersection>limit_area l'enlever de la liste
        #liste_DIFF2.append(max_intersection.index(proportion_maximum))
            liste_DIFF.append(liste_intersection.index(max_intersection))
        



    nb_birds_match=len(liste_DIFF)



    nb_birds_to_find=len(annontation_reduit)
    print( "nombre d'oiseau repérés", nb_birds_match)
    
    
    fin = time.time()
    print("cela dure", fin-debut)

    

    #En cas de zéro il faut soulever une erreur !!!
    liste_batch_images=range(len(batchImages_stack_reshape))
    liste_not_matche=set(liste_batch_images)-set(liste_DIFF)
    
    batchImages_match_birds = [batchImages_stack_reshape[i] for i in (liste_DIFF)]
    batchImages_match_not_birds = [batchImages_stack_reshape[i] for i in (liste_not_matche)]
    #

    try:
        

        batchImages_match_birds=np.vstack(batchImages_match_birds)
        batchImages_match_not_birds=np.vstack(batchImages_match_not_birds)
        #Prediction modèle

        
                

        estimates_match_not_birds = CNNmodel.predict(batchImages_match_not_birds.reshape(-1,28,28,3))
  
        
        
        estimates_match_birds = CNNmodel.predict(batchImages_match_birds.reshape(-1,28,28,3))
       
        

    
        liste_prediction=list(estimates_match_birds.argmax(axis=1))
        TP=(len(liste_prediction)-liste_prediction.count(0))
        pourcentage_TP=TP/len(liste_prediction)
        
        
        
        liste_prediction_not_match=list(estimates_match_not_birds.argmax(axis=1))
        FP=(len(liste_prediction_not_match)-liste_prediction_not_match.count(0))
        pourcentage_FP=FP/len(liste_prediction_not_match)


        #On va tester une troisième liste, les very True Positif
        #Il faut être le bon type d'oiseaux
        #On retourn +1 si l'oiseau est bien classé, à la fin on regardera le chiffre finale on divisera eventuellement par le nombre d'oiseau prédits
        VTP=0
        debut = time.time()
        #Ici on pourrait probablement gagner du temps en n'utilisant pas le dic à chaque étape mais seulement au début
        #et en comptant le nombre de valeurs qui correspond au bon i
        for i in range(len(liste_prediction)):
            if dictionnaire_conversion[liste_prediction[i]]==nom_classe:
                VTP+=1
        
        fin = time.time()
        print("cela dure", fin-debut)
        #D'abord on va transformer estimates avec préféré le max
        #Dans le tableau il faudra convertir la classe string en classe nombre
        #Puis ensuite on se demande si les deux sont égaux
        
        #Il faut s'assurer que estimates est seulement pour les imagettes tout d'abord (puis ensuite on verra)
      
        if nb_birds_match>0:
            are_there_birds=True
        else:
            are_there_birds=False
        #On va essayer maintenant de faire des prédictions  on l'intègrera ensuite dans les carrés
    except ValueError:
        print("il n'a pas d'imagettes trouvées !")
        pourcentage_TP=0
        TP=0
        VTP=0
        pourcentage_FP=0
        FP=0
        
         
    return nb_birds_to_find,nb_birds_match,pourcentage_TP,pourcentage_FP,TP,FP,VTP






























def birds_precis(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=19,blurFact=15,filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
    
    
    dictionnaire_conversion={}
    dictionnaire_conversion[0]="autre"
    dictionnaire_conversion[1]="chevreuil"
    dictionnaire_conversion[2]="corneille"
    dictionnaire_conversion[3]="faisan"
    dictionnaire_conversion[4]="lapin"
    dictionnaire_conversion[5]="pigeon"
    
    
    
    
    
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    
    
    
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    imagettes=to_reference_labels (imagettes,"classe")
    
    imagettes=imagettes[ (imagettes["classe"]!="oiseau") & (imagettes["classe"]!="autre") & (imagettes["classe"]!="pie") 
                        & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") & (imagettes["classe"]!="sanglier") 
                        & (imagettes["classe"]!="cheval") ]
    folder_choosen="."+ folder
    imagettes_PI_0=imagettes[(imagettes["path"]==folder_choosen) ]
    #Attention ça ne fonction qu'avec le PI_0 pour les autres dossiers il faut rajouter d'autres classes 
    imagettes_PI_0=imagettes_PI_0[   (imagettes_PI_0["classe"]!="ground") & (imagettes_PI_0["classe"]!="incertain")     ]    
    imagettes1=imagettes_PI_0[imagettes_PI_0["filename"]==name_test]
    if len(imagettes1["classe"].unique())>1:
        print("attention le code le prend en charge" )
    #nom_classe=imagettes1["classe"].iloc[0]
    nom_classe=list(imagettes1["classe"].unique())
    
    #On va comparer les images de ici et probablement test unitaires
    batchImages = []
    liste_table = []
    imageSize= 28
    cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    #On récupère les coordonnées des pixels différent par différence
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        #name = (os.path.split(name_test)[-1]).split(".")[0]
        #name = name + "_" + str(ic) + ".JPG"
        #name = nom_classe + "_" + str(ic) + ".JPG"
        name = "TOTO" + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   

        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif

        subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))
     
    table_full = pd.DataFrame(np.vstack(liste_table))
    
    
    #Ce serait bien de rajouter rename ici !!! Si ça n'entraine pas de bug
    table_full = table_full.rename(columns={0: 'imagettename', 1: 'xmin', 2: 'xmax', 3: 'ymin', 4: 'ymax'})
    table_full.iloc[:,1:]=table_full.iloc[:,1:].astype(int)


    batchImages_stack = np.vstack(batchImages)
    batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
    
    #Il faudra penser à soulever une expetion si rien trouver à l'isuu du filtre en particulier pour la ligne
    #fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
    #image_test image_2019-06-14_15-46-54.jpg image_ref image_2019-06-14_15-46-38.jpg
    """
    table_quantile,index_possible_birds=fn.filtre_quantile(table_full,coef_filtre,height=2448,width=3264)
    table_filtre_RL=table_quantile.copy()
    table_filtre_RL["possible_bird"]=fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
    table_filtre_RL=(table_filtre_RL[table_filtre_RL["possible_bird"]=="O"])
    p_bird=table_filtre_RL.index
    table_filtre_RL.drop("possible_bird",axis=1,inplace=True)     
    index_possible_birds=list(set(index_possible_birds).intersection(p_bird)) 
    #batchImages_filtre = [batchImages_stack_reshape[i] for i in (index_possible_birds)]
    """

    
    
    
    #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
    #generate_square=table.iloc[:,1:5]
    if filtre_choice=="No_filtre":
        generate_square=table_full.iloc[:,1:5]
    elif filtre_choice=="quantile_filtre":
        generate_square=table_quantile.iloc[:,1:5]
    elif filtre_choice=="RL_filtre":
        generate_square=table_filtre_RL.iloc[:,1:5]
    
    #liste_carre_diff=generate_square[['xmin', 'ymin', 'xmax', 'ymax']].apply(to_polygon,axis=1)
    

    
    
    xmin_gen=generate_square["xmin"]
    xmax_gen=generate_square["xmax"]
    ymin_gen=generate_square["ymin"]
    ymax_gen=generate_square["ymax"]
    

    
    
    to_drop=['path', 'filename', 'width', 'height', 'index']

    im=imagettes1.drop(to_drop,axis=1)
    col=list(im.columns)
    col = col[-1:] + col[:-1]


    annontation_reduit=im[col]
    
    TP=0
    VTP=0
    nb_birds_match=0
    for classe in nom_classe:
        table_type_birds=annontation_reduit[annontation_reduit["classe"]==classe]

        
    
        
        liste_DIFF=[]
        
        
        for i in range(len(table_type_birds)):
            x_min_anote=table_type_birds["xmin"].iloc[i]
            x_max_anote=table_type_birds["xmax"].iloc[i]
            y_min_anote=table_type_birds["ymin"].iloc[i]
            y_max_anote=table_type_birds["ymax"].iloc[i]
     
            ln_square_gen=len(xmin_gen)
        
            zip_xmin_anote=[x_min_anote]*ln_square_gen
            zip_xmax_anote=[x_max_anote]*ln_square_gen
            zip_ymin_anote=[y_min_anote]*ln_square_gen
            zip_ymax_anote=[y_max_anote]*ln_square_gen
        
            
            
            liste_intersection=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
            max_intersection=max(liste_intersection)
    
            #Il manque le if
            proportion_maximum=max_intersection/area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
            if proportion_maximum>coverage_threshold:
                #Il faudrait rajouter une boucle while tan que max_intersection>limit_area l'enlever de la liste
            #liste_DIFF2.append(max_intersection.index(proportion_maximum))
                liste_DIFF.append(liste_intersection.index(max_intersection))
            
    
    
    
        nb_birds_match_this_bird_type=len(liste_DIFF)
    
    
    
        nb_birds_to_find=len(annontation_reduit)
        print( "nombre d'oiseau repérés", nb_birds_match_this_bird_type)
        nb_birds_match+=nb_birds_match_this_bird_type
        
        
    
        #

        #En cas de zéro il faut soulever une erreur !!!
        liste_batch_images=range(len(batchImages_stack_reshape))
        liste_not_matche=set(liste_batch_images)-set(liste_DIFF)
        
        batchImages_match_birds = [batchImages_stack_reshape[i] for i in (liste_DIFF)]
        batchImages_match_not_birds = [batchImages_stack_reshape[i] for i in (liste_not_matche)]
        #
 
        try:
            
    
            batchImages_match_birds=np.vstack(batchImages_match_birds)
            batchImages_match_not_birds=np.vstack(batchImages_match_not_birds)
            #Prediction modèle
    
            
                    
            estimates_match_not_birds = CNNmodel.predict(batchImages_match_not_birds.reshape(-1,28,28,3))
            estimates_match_birds = CNNmodel.predict(batchImages_match_birds.reshape(-1,28,28,3))

    
        
            liste_prediction=list(estimates_match_birds.argmax(axis=1))
            TP_this_type=(len(liste_prediction)-liste_prediction.count(0))
            #pourcentage_TP=TP/len(liste_prediction)
            TP+=TP_this_type
            
            
            liste_prediction_not_match=list(estimates_match_not_birds.argmax(axis=1))
            FP=(len(liste_prediction_not_match)-liste_prediction_not_match.count(0))
            pourcentage_FP=FP/len(liste_prediction_not_match)
    
    
            #On va tester une troisième liste, les very True Positif
            #Il faut être le bon type d'oiseaux
            #On retourn +1 si l'oiseau est bien classé, à la fin on regardera le chiffre finale on divisera eventuellement par le nombre d'oiseau prédits


            #Ici on pourrait probablement gagner du temps en n'utilisant pas le dic à chaque étape mais seulement au début
            #et en comptant le nombre de valeurs qui correspond au bon i
           
            for i in range(len(liste_prediction)):
                if dictionnaire_conversion[liste_prediction[i]]==classe:
                    VTP+=1

            #D'abord on va transformer estimates avec préféré le max
            #Dans le tableau il faudra convertir la classe string en classe nombre
            #Puis ensuite on se demande si les deux sont égaux
            
            #Il faut s'assurer que estimates est seulement pour les imagettes tout d'abord (puis ensuite on verra)
 
            #On va essayer maintenant de faire des prédictions  on l'intègrera ensuite dans les carrés
        except ValueError:
            print("il n'a pas d'imagettes trouvées !")
            #pourcentage_TP=0
            TP=0
            VTP=0
            pourcentage_FP=0
            FP=0
        
         
    return nb_birds_to_find,nb_birds_match,pourcentage_FP,TP,FP,VTP











#





#



def bien_classe(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=19,blurFact=15,filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
    
    
    dictionnaire_conversion={}
    dictionnaire_conversion[0]="autre"
    dictionnaire_conversion[1]="chevreuil"
    dictionnaire_conversion[2]="corneille"
    dictionnaire_conversion[3]="faisan"
    dictionnaire_conversion[4]="lapin"
    dictionnaire_conversion[5]="pigeon"
    
    
    
    
    
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    
    
    
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    imagettes=to_reference_labels (imagettes,"classe")
    
    imagettes_defined_classes=imagettes[ (imagettes["classe"]!="oiseau") & (imagettes["classe"]!="autre") & (imagettes["classe"]!="pie") 
                        & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") & (imagettes["classe"]!="sanglier") 
                        & (imagettes["classe"]!="cheval") ]
    
    
    
    
    
    
    
    
    
    
    
    
    
    folder_choosen="."+ folder
    imagettes_PI_0=imagettes_defined_classes[(imagettes_defined_classes["path"]==folder_choosen) ]
    #Attention ça ne fonction qu'avec le PI_0 pour les autres dossiers il faut rajouter d'autres classes 
    imagettes_PI_0=imagettes_PI_0[   (imagettes_PI_0["classe"]!="ground") & (imagettes_PI_0["classe"]!="incertain")     ]    
    imagettes1=imagettes_PI_0[imagettes_PI_0["filename"]==name_test]
    
    
    
    if len(imagettes1["classe"].unique())>1:
        print("attention le code le prend en charge" )
        
        
    #nom_classe=imagettes1["classe"].iloc[0]
    nom_classe=list(imagettes1["classe"].unique())
    
    #On va comparer les images de ici et probablement test unitaires
    batchImages = []
    liste_table = []
    imageSize= 28
    cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    #On récupère les coordonnées des pixels différent par différence
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        #name = (os.path.split(name_test)[-1]).split(".")[0]
        #name = name + "_" + str(ic) + ".JPG"
        #name = nom_classe + "_" + str(ic) + ".JPG"
        name = "TOTO" + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   

        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif

        subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))
     
    table_full = pd.DataFrame(np.vstack(liste_table))
    
    
    #Ce serait bien de rajouter rename ici !!! Si ça n'entraine pas de bug
    table_full = table_full.rename(columns={0: 'imagettename', 1: 'xmin', 2: 'xmax', 3: 'ymin', 4: 'ymax'})
    table_full.iloc[:,1:]=table_full.iloc[:,1:].astype(int)


    batchImages_stack = np.vstack(batchImages)
    batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
    
    #Il faudra penser à soulever une expetion si rien trouver à l'isuu du filtre en particulier pour la ligne
    #fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
    #image_test image_2019-06-14_15-46-54.jpg image_ref image_2019-06-14_15-46-38.jpg
    """
    table_quantile,index_possible_animals=fn.filtre_quantile(table_full,coef_filtre,height=2448,width=3264)
    table_filtre_RL=table_quantile.copy()
    table_filtre_RL["possible_bird"]=fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
    table_filtre_RL=(table_filtre_RL[table_filtre_RL["possible_bird"]=="O"])
    p_bird=table_filtre_RL.index
    table_filtre_RL.drop("possible_bird",axis=1,inplace=True)     
    index_possible_animals=list(set(index_possible_animals).intersection(p_bird)) 
    #batchImages_filtre = [batchImages_stack_reshape[i] for i in (index_possible_animals)]
    """

    
    
    
    #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
    #generate_square=table.iloc[:,1:5]
    if filtre_choice=="No_filtre":
        generate_square=table_full.iloc[:,1:5]
    elif filtre_choice=="quantile_filtre":
        generate_square=table_quantile.iloc[:,1:5]
    elif filtre_choice=="RL_filtre":
        generate_square=table_filtre_RL.iloc[:,1:5]
    
    #liste_carre_diff=generate_square[['xmin', 'ymin', 'xmax', 'ymax']].apply(to_polygon,axis=1)
    

    
    
    xmin_gen=generate_square["xmin"]
    xmax_gen=generate_square["xmax"]
    ymin_gen=generate_square["ymin"]
    ymax_gen=generate_square["ymax"]
    

    
    
    to_drop=['path', 'filename', 'width', 'height', 'index']

    im=imagettes1.drop(to_drop,axis=1)
    col=list(im.columns)
    col = col[-1:] + col[:-1]


    annontation_reduit=im[col]
    
    animals_predict=0
    birds_predict=0
    VTP=0
    nb_animals_match=0
    
    #On classe les difference repéré pour les animaux classés
    for classe in nom_classe:
        table_type_animals_all=annontation_reduit[annontation_reduit["classe"]==classe]

        
    
        
        liste_DIFF_class_defined=[]
        
        
        for i in range(len(table_type_animals_all)):
            x_min_anote=table_type_animals_all["xmin"].iloc[i]
            x_max_anote=table_type_animals_all["xmax"].iloc[i]
            y_min_anote=table_type_animals_all["ymin"].iloc[i]
            y_max_anote=table_type_animals_all["ymax"].iloc[i]
     
            ln_square_gen=len(xmin_gen)
        
            zip_xmin_anote=[x_min_anote]*ln_square_gen
            zip_xmax_anote=[x_max_anote]*ln_square_gen
            zip_ymin_anote=[y_min_anote]*ln_square_gen
            zip_ymax_anote=[y_max_anote]*ln_square_gen
        
            
            
            liste_intersection=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
            max_intersection=max(liste_intersection)
    
            #Il manque le if
            proportion_maximum=max_intersection/area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
            if proportion_maximum>coverage_threshold:
                #Il faudrait rajouter une boucle while tan que max_intersection>limit_area l'enlever de la liste
            #liste_DIFF2.append(max_intersection.index(proportion_maximum))
                liste_DIFF_class_defined.append(liste_intersection.index(max_intersection))
            
    
    
    
    
    
    
        #nb_animals_match_this_bird_type=len(liste_DIFF)
    
        nb_animals_match=len(liste_DIFF_class_defined)

        nb_animals_to_find=len(annontation_reduit)
        print( "nombre d'animaux repérés", nb_animals_match)
        birds_table=imagettes[ (imagettes["classe"]=="corneille"  ) | (imagettes["classe"]=="pigeon"  ) |(imagettes["classe"]=="faisan"  ) 
        | (imagettes["classe"]=="oiseaux"  )| (imagettes["classe"]=="pie"  )]
        print("nombre d'oiseaux dans l'image",len(birds_table))
        #print( "nombre d'oiseau repérés", nb_animals_match_this_bird_type)
        #nb_animals_match+=nb_animals_match_this_bird_type
        
        
    
        #

        #En cas de zéro il faut soulever une erreur !!!
        liste_batch_images=range(len(batchImages_stack_reshape))
        liste_not_matche=set(liste_batch_images)-set(liste_DIFF_class_defined)
        
        batchImages_match_animals = [batchImages_stack_reshape[i] for i in (liste_DIFF_class_defined)]
        batchImages_match_not_animal = [batchImages_stack_reshape[i] for i in (liste_not_matche)]
        
 
        try:
            
    
            batchImages_match_animals=np.vstack(batchImages_match_animals)
            batchImages_match_not_animals=np.vstack(batchImages_match_not_animal)
            #Prediction modèle
    
            
                    
            estimates_match_not_animals = CNNmodel.predict(batchImages_match_not_animals.reshape(-1,28,28,3))
            estimates_match_animals = CNNmodel.predict(batchImages_match_animals.reshape(-1,28,28,3))

    
        
            liste_prediction=list(estimates_match_animals.argmax(axis=1))
            #animal_this_type=(len(liste_prediction)-liste_prediction.count(0))
            #animals_predict+=animal_this_type
            animals_predict=(len(liste_prediction)-liste_prediction.count(0))
            #faisan,corbeau,pigeon à vérifier
            birds_predict=liste_prediction.count(2)+liste_prediction.count(3)+liste_prediction.count(5)
            
            
            
            #pourcentage_TP=TP/len(liste_prediction)
            
            
            
            liste_prediction_not_match=list(estimates_match_not_animals.argmax(axis=1))
            FP=(len(liste_prediction_not_match)-liste_prediction_not_match.count(0))
            pourcentage_FP=FP/len(liste_prediction_not_match)
    
    
            #On va tester une troisième liste, les very True Positif
            #Il faut être le bon type d'oiseaux
            #On retourn +1 si l'oiseau est bien classé, à la fin on regardera le chiffre finale on divisera eventuellement par le nombre d'oiseau prédits


            #Ici on pourrait probablement gagner du temps en n'utilisant pas le dic à chaque étape mais seulement au début
            #et en comptant le nombre de valeurs qui correspond au bon i
           
            for i in range(len(liste_prediction)):
                if dictionnaire_conversion[liste_prediction[i]]==classe:
                    VTP+=1

            #D'abord on va transformer estimates avec préféré le max
            #Dans le tableau il faudra convertir la classe string en classe nombre
            #Puis ensuite on se demande si les deux sont égaux
            
            #Il faut s'assurer que estimates est seulement pour les imagettes tout d'abord (puis ensuite on verra)
 
            #On va essayer maintenant de faire des prédictions  on l'intègrera ensuite dans les carrés
        except ValueError:
            print("il n'a pas d'imagettes trouvées !")
            #pourcentage_TP=0
            animals_predict=0
            VTP=0
            pourcentage_FP=0
            FP=0
            birds_predict=0
         
    return nb_animals_to_find,nb_animals_match,pourcentage_FP,animals_predict,FP,VTP,birds_predict

#

































#retourner predict defined birds undefined birds, birds perfect predicts on N possible
#debuger fonction aussi


def bazard2(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=19,blurFact=15,
            filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
    
    #Definition du dicionnaire de conversion
    dictionnaire_conversion={}
    dictionnaire_conversion[0]="autre"
    dictionnaire_conversion[1]="chevreuil"
    dictionnaire_conversion[2]="corneille"
    dictionnaire_conversion[3]="faisan"
    dictionnaire_conversion[4]="lapin"
    dictionnaire_conversion[5]="pigeon"
    #
    #Initialisation de variables et de liste
    batchImages = []
    liste_table = []
    imageSize= 28
    animals_predict=0
    birds_predict=0
    VTP=0
    nb_animals_match=0

    
    
    
    #Definition des images
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    
    
    #Ouverture des fichiers annotés 
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    imagettes=to_reference_labels (imagettes,"classe")
    
    
    #On ne conserve que les annotations dont le label se retrouve dans le Réseau de neurones
    #Eventuellement réincorpoer les animaux mais à priori pas d'intérêt
    imagettes=imagettes[  (imagettes["classe"]!="autre") & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") 
    & (imagettes["classe"]!="sanglier") & (imagettes["classe"]!="cheval") & (imagettes["classe"]!="ground") & (imagettes["classe"]!="autre") ]
    
    
    #Il faut enlever autre
    
    
    #imagettes_birds_indefined=imagettes[ (imagettes["classe"]=="oiseau") |  (imagettes["classe"]=="pie") ]    
    
    
    #On se place dans le bon dossier pour ces anotations (peut être inutile mais on sait jamais s'il y a des doublons à vérifier doublons pour imagettesname)
    #Il y 1485 imagettes différents sur l'ensemble du dossier contre 1487 pour chaque dossier pris séparement, probablement une faute dans le placement.
    folder_choosen="."+ folder
    imagettes_folder=imagettes[(imagettes["path"]==folder_choosen) ]
    #Attention ça ne fonction qu'avec le PI_0 pour les autres dossiers il faut rajouter d'autres classes 
    
    #imagettes_PI_0=imagettes_PI_0[   (imagettes_PI_0["classe"]!="ground") & (imagettes_PI_0["classe"]!="incertain")     ]    
    
    #On selectionne seulement pour la photo sur laquel on veut repérer les oiseaux ou autres animaux et on réarange les colonnes dans le bon ordre
    imagettes_target=imagettes_folder[imagettes_folder["filename"]==name_test]
    to_drop=['path', 'filename', 'width', 'height', 'index']
    imagettes_target=imagettes_target.drop(to_drop,axis=1)
    col = list(imagettes_target.columns)[-1:] + list(imagettes_target.columns)[:-1]
    imagettes_target=imagettes_target[col]
    
    
    #Peut être que je peux séparer ici imagettes_target en deux entre definied et indefiend
    
    
    #On regarde si il y a des imagettes de type différents pas forcément utiles surtout si on garde les oiseaux undefined
    if len(imagettes_target["classe"].unique())>1:
        print("attention le code le prend en charge" )
    #nom_classe=imagettes1["classe"].iloc[0]
    nom_classe=list(imagettes_target["classe"].unique())
    

    
    
    
    
    #On récupère les coordonnées des pixels différent par différence la colonne name parait vraiment optionnelle peut être la retirée
    #on obtien les batchs et une tables
    #Il y a un warning dans le block mais où
    cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        #name = "TOTO" + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))
    

    
    #try:
    azerty=0
    if azerty==0:
      
        batchImages_stack = np.vstack(batchImages)
        batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
        table_non_filtre = pd.DataFrame(np.vstack(liste_table))
        table_non_filtre = table_non_filtre.rename(columns={ 0: 'xmin', 1: 'xmax', 2: 'ymin', 3: 'ymax'})
        table_non_filtre=table_non_filtre.astype(int)
        
    
    
        
        #Maintenant on va procéder à un certain nombre de filtre sur la table (ici desactivé) pour ne proposer que les batch utiles selon l'index
        #Si on peut passer de la table au batch facilement il y a mieux à faire ... ou au mons à liste_table
        
        #Il faudra penser à soulever une expetion si rien trouver à l'isuu du filtre en particulier pour la ligne
        #fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
        #image_test image_2019-06-14_15-46-54.jpg image_ref image_2019-06-14_15-46-38.jpg
        """
        table_quantile,index_possible_animals=fn.filtre_quantile(table_non_filtre,coef_filtre,height=2448,width=3264)
        table_filtre_RL=table_quantile.copy()
        table_filtre_RL["possible_bird"]=fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
        table_filtre_RL=(table_filtre_RL[table_filtre_RL["possible_bird"]=="O"])
        p_bird=table_filtre_RL.index
        table_filtre_RL.drop("possible_bird",axis=1,inplace=True)     
        index_possible_animals=list(set(index_possible_animals).intersection(p_bird)) 
        #batchImages_filtre = [batchImages_stack_reshape[i] for i in (index_possible_animals)]
        """
    
    
        #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
        #generate_square=table.iloc[:,1:5]
        if filtre_choice=="No_filtre":
            generate_square=table_non_filtre
        elif filtre_choice=="quantile_filtre":
            generate_square=table_quantile
        elif filtre_choice=="RL_filtre":
            generate_square=table_filtre_RL
        

        xmin_gen=generate_square["xmin"]
        xmax_gen=generate_square["xmax"]
        ymin_gen=generate_square["ymin"]
        ymax_gen=generate_square["ymax"]
        ln_square_gen=len(generate_square)
    
       
        
        
        
        
        #C'est bien suspect d'avoir cette liste à l'interieur de la boucle
        liste_DIFF_birds_defined=[]
        liste_DIFF_birds_undefined=[]
        liste_DIFF_other_animals=[]
        #On classe les difference repéré pour les animaux classés
        for classe in nom_classe:
            
            #Ou sinon on peut exclure ici les oiseaux non defs... .
            imagettes_annote_1_classe=imagettes_target[imagettes_target["classe"]==classe]
            nb_imagettes_1_classe=len(imagettes_annote_1_classe)
            
            #get max intersection with square generate for each sqaure annotate
            for i in range(nb_imagettes_1_classe):
                x_min_anote=imagettes_annote_1_classe["xmin"].iloc[i]
                x_max_anote=imagettes_annote_1_classe["xmax"].iloc[i]
                y_min_anote=imagettes_annote_1_classe["ymin"].iloc[i]
                y_max_anote=imagettes_annote_1_classe["ymax"].iloc[i]
         
    
                #Replicated the coordinates of annotations the number time of the len  to be able to apply area_intersection function
                zip_xmin_anote=[x_min_anote]*ln_square_gen
                zip_xmax_anote=[x_max_anote]*ln_square_gen
                zip_ymin_anote=[y_min_anote]*ln_square_gen
                zip_ymax_anote=[y_max_anote]*ln_square_gen
            
                
                #apply function area_intersection to calculate the area intersected between all generate square for this annote_squared 
                #and get the square with the max intersectio if there is enough area in commun
                
                #Je vais encore séparer les listes ici pour proposer oiseaux définis et autres animaux... . 
                liste_intersection=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip 
                                    (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
                max_intersection=max(liste_intersection)
                proportion_maximum=max_intersection/area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
                if proportion_maximum>coverage_threshold and (  (classe=="faisan")   or (classe=="corneille") or (classe=="pigeon")):
                    #Il faudrait rajouter une boucle while tan que max_intersection>limit_area l'enlever de la liste
                #liste_DIFF2.append(max_intersection.index(proportion_maximum))
                    liste_DIFF_birds_defined.append(liste_intersection.index(max_intersection))
                if proportion_maximum>coverage_threshold and (  (classe=="oiseau")   or (classe=="pie") or (classe=="incertain") ):
                    liste_DIFF_birds_undefined.append(liste_intersection.index(max_intersection))            
                if proportion_maximum>coverage_threshold and (  (classe=="lapin")   or (classe=="chevreuil") ):
                    liste_DIFF_other_animals.append(liste_intersection.index(max_intersection))           
        
    
            #Capture du nombre d'imagettes à caputerer et captutées
        
        birds_table=imagettes_target[ (imagettes_target["classe"]=="corneille"  ) | (imagettes_target["classe"]=="pigeon"  ) |(imagettes_target["classe"]=="faisan"  ) 
        | (imagettes_target["classe"]=="oiseau"  )| (imagettes_target["classe"]=="pie"  ) |      (imagettes_target["classe"]=="incertain"  )      ]
        nb_animals_match=len(liste_DIFF_birds_defined)+len(liste_DIFF_birds_undefined)+len(liste_DIFF_other_animals)
        nb_animals_to_find=len(imagettes_target)
        birds_to_find=len(birds_table)
        birds_defined_match=len(liste_DIFF_birds_defined)
        birds_match=birds_defined_match+len(liste_DIFF_birds_undefined)
        print( "nombre d'animaux à reperer", nb_animals_to_find)
        print( "nombre d'animaux repérés", nb_animals_match)
        print("nombre d'oiseaux dans l'image",birds_to_find)
        print("nombre d'oiseaux totaux repérés dans l'image",birds_match)
        print("nombre d'oiseaux repérés parmi les oiseaux labélisés",birds_defined_match )

  
        try:
                


            #Predicts birds 
            batchImages_match_birds = [batchImages_stack_reshape[i] for i in (liste_DIFF_birds_defined+liste_DIFF_birds_undefined)]
            batchImages_match_birds=np.vstack(batchImages_match_birds)
            estimates_match_brids = CNNmodel.predict(batchImages_match_birds.reshape(-1,28,28,3))
            liste_prediction_birds=list(estimates_match_brids.argmax(axis=1))
            #animals_predict=(len(liste_prediction)-liste_prediction.count(0))
            birds_predict=liste_prediction_birds.count(2)+liste_prediction_birds.count(3)+liste_prediction_birds.count(5)   

        except ValueError:
                print("il n'a pas d'imagettes d'oiseaux trouvées !")
                #pourcentage_TP=0
                birds_match,birds_predict,birds_defined_match,VTP=(0,0,0,0)
                liste_prediction_birds=[]
                birds_predict=0
        try:

            #Predicts  animals
            #probablement moyen de pas dédoubler le calcul
            #Il faut rajouter les oiseaux ...
            batchImages_match_other_animals = [batchImages_stack_reshape[i] for i in (liste_DIFF_other_animals)]
            batchImages_match_other_animals=np.vstack(batchImages_match_other_animals)
            estimates_match_other_animals = CNNmodel.predict(batchImages_match_other_animals.reshape(-1,28,28,3))
            liste_prediction_other_animals=list(estimates_match_other_animals.argmax(axis=1))
            animals_predict=(len(liste_prediction_other_animals)-liste_prediction_birds.count(0)) + (len(liste_prediction_birds)-liste_prediction_birds.count(0))
            #birds_predict=liste_prediction.count(2)+liste_prediction.count(3)+liste_prediction.count(5)     

        except ValueError:
                print("il n'a pas d'imagettes de lapins ou de chevreuils !")
                animals_predict=birds_predict
                #pourcentage_TP=0

            
            
        try:
            #Ici on pourrait probablement gagner du temps en n'utilisant pas le dic à chaque étape mais seulement au début
            #et en comptant le nombre de valeurs qui correspond au bon i
            #Determiner les vrai positifs à partir des images qui sont bien labélisées seulement
            batch_labels_defined = [batchImages_stack_reshape[i] for i in liste_DIFF_birds_defined]
            batch_labels_defined=np.vstack(batch_labels_defined)
            estimates_labels = CNNmodel.predict(batch_labels_defined.reshape(-1,28,28,3))
            liste_labels_prediction=list(estimates_labels.argmax(axis=1))            
            for i in range(len(liste_labels_prediction)):
                if dictionnaire_conversion[liste_prediction_birds[i]]==classe:
                    VTP+=1
    
        except ValueError:
                print("il n'a pas d'oiseaux identifiés")
                animals_predict=birds_predict
                #pourcentage_TP=0
    
    
        #On peut certainement encore baisser le nombre vu qu'on prend que les oiseaux
        #Predict False Positif
        liste_batch_images=range(len(batchImages_stack_reshape))
        liste_not_matche=set(liste_batch_images)-set(liste_DIFF_birds_defined)-set(liste_DIFF_birds_undefined)-set(liste_DIFF_other_animals)
        batchImages_match_not_animal = [batchImages_stack_reshape[i] for i in liste_not_matche]
        batchImages_match_not_animal=np.vstack(batchImages_match_not_animal)                 
        estimates_match_not_animals = CNNmodel.predict(batchImages_match_not_animal.reshape(-1,28,28,3))
        liste_prediction_not_match=list(estimates_match_not_animals.argmax(axis=1))
        FP=(len(liste_prediction_not_match)-liste_prediction_not_match.count(0))
        nombre_imagettes=len(liste_prediction_not_match)
        pourcentage_FP=FP/nombre_imagettes
                
    

    

    #except ValueError:
        #print("il n'a aucune différence entre les images comparées !")
        #pourcentage_TP=0
        #nb_animals_match,birds_match,birds_predict,birds_defined_match,VTP=(0,0,0,0,0)
        
    return nb_animals_to_find,nb_animals_match,birds_to_find,birds_match,birds_predict,birds_defined_match,VTP,nombre_imagettes,FP



#On pourrait aller chercher le nombre d'oiseau definied à trouver ... . 










def find_square_reverse (imageB,intervalle,zoom,
                    name2,cnts,
                    maxAnalDL, # can be set to -1 to analyse everything
                    CNNmodel,labels,filtre_RL,
                    x_pix_max,y_pix_max,x_pix_min,y_pix_min
                    ,coef_filtre,height=2448,width=3264,table_add=pd.read_csv("testingInputs/oiseau_lab_Alex.csv")) : 
    ## Permet d'enregistrer et de tracer les contours repérés plus haut
    ## les images sont stockées dans le fichier "XXX.JPG_subImages"
#    cv2.imwrite(subImagesDir2+"diffsTrRaw"+timeStamp2+".JPG",thresh)

    
    # open the output CSV file for writing
    annontation_reduit=(table_add.iloc[:,6:12]).drop("index",axis=1)
    
    batchImages = []
    liste_table = []
    imageSize= 28
#    np.empty((1,5), dtype = "int")  
    
    imageRectangles = imageB.copy()
    for ic in range(0,len(cnts)):
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    
        #
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        name = (os.path.split(name2)[-1]).split(".")[0]
        name = name + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")

   
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   

        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif
        if( (f.xmax-f.xmin)<x_pix_max and (f.ymax-f.ymin)<y_pix_max # birds should less than 500 pixels wide and 350 high
           and (f.xmax-f.xmin)>x_pix_min and (f.ymax-f.ymin)>y_pix_min): # according to distribution in annotations
            subI, o, d, imageRectanglesB = GetSquareSubset(imageB,f,verbose=False)
            subI = RecenterImage(subI,o)
            subI = cv2.resize(subI,(imageSize,imageSize))
            subI = np.expand_dims(subI, axis=0)
            # subI = preprocess_input(subI)
            batchImages.append(subI)
            
            liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))
            
            #table.append(np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5)))
#            table1 = np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5))
    
            #   cv2.rectangle(imageRectangles, (o.xmin,o.ymin), (o.xmax,o.ymax), (255, 0, 0), 2)     
        #écriture des images
#        cv2.imwrite("images_carre/"+h.filename[:-4]+".JPG",img1)
#        
    for i in range(len(annontation_reduit)):
        f.xmin, f.xmax, f.ymin, f.ymax = annontation_reduit["xmin"].iloc[i], annontation_reduit["xmax"].iloc[i], annontation_reduit["ymin"].iloc[i], annontation_reduit["ymax"].iloc[i]
        subI, o, d, imageRectanglesB = GetSquareSubset(imageB,f,verbose=False)
        batchImages.append(subI)
        liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))
        
    table = pd.DataFrame(np.vstack(liste_table))
    table,index_possible_birds=filtre_quantile(table,coef_filtre,height=2448,width=3264)
    #table=table.rename(columns={0: "imagetteName", 1: "xmin", 2 : "xmax", 3 : "ymin" , 4 : "ymax" })
    
    #Affecter le max et le min  renommer les variables
    liste_xmax=[]
    liste_xmin=[]
    liste_ymax=[]
    liste_ymin=[]
    for i in range(len(table)):
        XMAX=table["xmax"].iloc[i]
        XMIN=table["xmin"].iloc[i]
        YMAX=table["ymax"].iloc[i]
        YMIN=table["ymin"].iloc[i] 
        
        #MAX=table["xmax"].iloc[i]=agrandissement(XMIN,XMAX,zoom)
        largeur_min,largeur_max=agrandissement(XMIN,XMAX,zoom)
        largeur_min=int(round(largeur_min))
        largeur_max=int(round(largeur_max))
        liste_xmin.append(largeur_min)
        liste_xmax.append(largeur_max)
        
        profondeur_min,profondeur_max=agrandissement(YMIN,YMAX,zoom)
        profondeur_min=int(round(profondeur_min))
        profondeur_max=int(round(profondeur_max))
        liste_ymin.append(profondeur_min)
        liste_ymax.append(profondeur_max)
        
        
    #On ajoute les éléments de la dataframe provenant des anotations de Alex
    liste_xmax=liste_xmax
    liste_xmin=liste_xmin
    liste_ymax=liste_ymax
    liste_ymin=liste_ymin
    
 
    
    table["xmax"]=liste_xmax
    table["xmin"]=liste_xmin
    table["ymax"]=liste_ymax
    table["ymin"]=liste_ymin
    
    #table=pd.concat([table,annontation_reduit])
    
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    #on enleve le filtre qui ne fonctionne pas
 
    # todo : add max_probas to out table in addition to results from DL
    #        and keep all "imagettes" in table
    
    #table = table.iloc[list(iOrderProba)[:min(len(cnts),maxAnalDL)],:]
    #batchImages = [batchImages[i] for i in list(iOrderProba)[:min(len(cnts),maxAnalDL)]]
    
    batchImages_filtre = [batchImages[i] for i in (index_possible_birds+list(range(26)))]
    
    batchImages_filtre = np.vstack(batchImages_filtre)
    #print("features extraction")
    
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
    features=preprocess_input(np.array(batchImages_filtre))
    features=features.reshape((features.shape[0], 28,28,3))
    estimates = CNNmodel.predict(features)
    for i in labels:
        table[i]=0
        
        
        
    
    colonne=0    
    for categorie in labels:
        for l in range(len(table)):
            table[categorie].iloc[l]=estimates[l,colonne]
        colonne+=1
        
    for i in range(len(table)):   
        xmin=int(table.iloc[i,1])
        xmax=int(table.iloc[i,2])
        ymin=int(table.iloc[i,3])
        ymax=int(table.iloc[i,4])
        #cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2) 
        max_col=np.amax(table.iloc[i,5:],axis=0)
        #if table.iloc[i,5]>=max_col:
        if (i>intervalle[0]) and (i<intervalle[1]):
            cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)
            if table.iloc[i,10]==max_col:
                cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2) 
            elif table.iloc[i,5]==max_col:
                cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 255, 255), 2) 
            
            else:
                cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 
        
    return table,imageRectangles,batchImages_filtre,batchImages;







#This script is to find the false positive rectangle or to other category
    

def fp_square(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=19,blurFact=15,
            filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
    
    #Definition du dicionnaire de conversion
    dictionnaire_conversion={}
    dictionnaire_conversion[0]="autre"
    dictionnaire_conversion[1]="chevreuil"
    dictionnaire_conversion[2]="corneille"
    dictionnaire_conversion[3]="faisan"
    dictionnaire_conversion[4]="lapin"
    dictionnaire_conversion[5]="pigeon"
    #
    #Initialisation de variables et de liste
    batchImages = []
    liste_table = []
    imageSize= 28
    animals_predict=0
    birds_predict=0
    VTP=0
    nb_animals_match=0

    
    
    
    #Definition des images
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    
    
    #Ouverture des fichiers annotés 
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    imagettes=to_reference_labels (imagettes,"classe")
    
    
    #On ne conserve que les annotations dont le label se retrouve dans le Réseau de neurones
    #Eventuellement réincorpoer les animaux mais à priori pas d'intérêt
    imagettes=imagettes[  (imagettes["classe"]!="autre") & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") 
    & (imagettes["classe"]!="sanglier") & (imagettes["classe"]!="cheval") & (imagettes["classe"]!="ground") & (imagettes["classe"]!="autre") ]
    
    
    #Il faut enlever autre
    
    
    #imagettes_birds_indefined=imagettes[ (imagettes["classe"]=="oiseau") |  (imagettes["classe"]=="pie") ]    
    
    
    #On se place dans le bon dossier pour ces anotations (peut être inutile mais on sait jamais s'il y a des doublons à vérifier doublons pour imagettesname)
    #Il y 1485 imagettes différents sur l'ensemble du dossier contre 1487 pour chaque dossier pris séparement, probablement une faute dans le placement.
    folder_choosen="."+ folder
    imagettes_folder=imagettes[(imagettes["path"]==folder_choosen) ]
    #Attention ça ne fonction qu'avec le PI_0 pour les autres dossiers il faut rajouter d'autres classes 
    
    #imagettes_PI_0=imagettes_PI_0[   (imagettes_PI_0["classe"]!="ground") & (imagettes_PI_0["classe"]!="incertain")     ]    
    
    #On selectionne seulement pour la photo sur laquel on veut repérer les oiseaux ou autres animaux et on réarange les colonnes dans le bon ordre
    imagettes_target=imagettes_folder[imagettes_folder["filename"]==name_test]
    to_drop=['path', 'filename', 'width', 'height', 'index']
    imagettes_target=imagettes_target.drop(to_drop,axis=1)
    col = list(imagettes_target.columns)[-1:] + list(imagettes_target.columns)[:-1]
    imagettes_target=imagettes_target[col]
    
    
    #Peut être que je peux séparer ici imagettes_target en deux entre definied et indefiend
    
    
    #On regarde si il y a des imagettes de type différents pas forcément utiles surtout si on garde les oiseaux undefined
    if len(imagettes_target["classe"].unique())>1:
        print("attention le code le prend en charge" )
    #nom_classe=imagettes1["classe"].iloc[0]
    nom_classe=list(imagettes_target["classe"].unique())
    

    
    
    
    
    #On récupère les coordonnées des pixels différent par différence la colonne name parait vraiment optionnelle peut être la retirée
    #on obtien les batchs et une tables
    #Il y a un warning dans le block mais où
    cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        #name = "TOTO" + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))
    

    
    #try:
    azerty=0
    if azerty==0:
      
        batchImages_stack = np.vstack(batchImages)
        batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
        table_non_filtre = pd.DataFrame(np.vstack(liste_table))
        table_non_filtre = table_non_filtre.rename(columns={ 0: 'xmin', 1: 'xmax', 2: 'ymin', 3: 'ymax'})
        table_non_filtre=table_non_filtre.astype(int)
        
    
    
        
        #Maintenant on va procéder à un certain nombre de filtre sur la table (ici desactivé) pour ne proposer que les batch utiles selon l'index
        #Si on peut passer de la table au batch facilement il y a mieux à faire ... ou au mons à liste_table
        
        #Il faudra penser à soulever une expetion si rien trouver à l'isuu du filtre en particulier pour la ligne
        #fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
        #image_test image_2019-06-14_15-46-54.jpg image_ref image_2019-06-14_15-46-38.jpg

        #table_quantile,index_possible_animals=fn.filtre_quantile(table_non_filtre,coef_filtre,height=2448,width=3264)
        """
        table_filtre_RL=table_quantile.copy()
        table_filtre_RL["possible_bird"]=fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
        table_filtre_RL=(table_filtre_RL[table_filtre_RL["possible_bird"]=="O"])
        p_bird=table_filtre_RL.index
        table_filtre_RL.drop("possible_bird",axis=1,inplace=True)     
        index_possible_animals=list(set(index_possible_animals).intersection(p_bird)) 
        #batchImages_filtre = [batchImages_stack_reshape[i] for i in (index_possible_animals)]
        """
    
    
        #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
        #generate_square=table.iloc[:,1:5]
        if filtre_choice=="No_filtre":
            generate_square=table_non_filtre
            #generate_square=table_quantile
        elif filtre_choice=="quantile_filtre":
            generate_square=table_quantile
        elif filtre_choice=="RL_filtre":
            generate_square=table_filtre_RL
        


        #Ici on propose le tableau coord imagettes et classe predict
        





        xmin_gen=generate_square["xmin"]
        xmax_gen=generate_square["xmax"]
        ymin_gen=generate_square["ymin"]
        ymax_gen=generate_square["ymax"]
        ln_square_gen=len(generate_square)
    
       
        
        
        
        
        #C'est bien suspect d'avoir cette liste à l'interieur de la boucle
        liste_DIFF_birds_defined=[]
        liste_DIFF_birds_undefined=[]
        liste_DIFF_other_animals=[]
        #On classe les difference repéré pour les animaux classés
        for classe in nom_classe:
            
            #Ou sinon on peut exclure ici les oiseaux non defs... .
            imagettes_annote_1_classe=imagettes_target[imagettes_target["classe"]==classe]
            nb_imagettes_1_classe=len(imagettes_annote_1_classe)
            
            #get max intersection with square generate for each sqaure annotate
            for i in range(nb_imagettes_1_classe):
                x_min_anote=imagettes_annote_1_classe["xmin"].iloc[i]
                x_max_anote=imagettes_annote_1_classe["xmax"].iloc[i]
                y_min_anote=imagettes_annote_1_classe["ymin"].iloc[i]
                y_max_anote=imagettes_annote_1_classe["ymax"].iloc[i]
         
    
                #Replicated the coordinates of annotations the number time of the len  to be able to apply area_intersection function
                zip_xmin_anote=[x_min_anote]*ln_square_gen
                zip_xmax_anote=[x_max_anote]*ln_square_gen
                zip_ymin_anote=[y_min_anote]*ln_square_gen
                zip_ymax_anote=[y_max_anote]*ln_square_gen
            
                
                #apply function area_intersection to calculate the area intersected between all generate square for this annote_squared 
                #and get the square with the max intersectio if there is enough area in commun
                
                #Je vais encore séparer les listes ici pour proposer oiseaux définis et autres animaux... . 
                liste_intersection=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip 
                                    (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
                max_intersection=max(liste_intersection)
                proportion_maximum=max_intersection/area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
                if proportion_maximum>coverage_threshold and (  (classe=="faisan")   or (classe=="corneille") or (classe=="pigeon")):
                    #Il faudrait rajouter une boucle while tan que max_intersection>limit_area l'enlever de la liste
                #liste_DIFF2.append(max_intersection.index(proportion_maximum))
                    liste_DIFF_birds_defined.append(liste_intersection.index(max_intersection))
                if proportion_maximum>coverage_threshold and (  (classe=="oiseau")   or (classe=="pie") or (classe=="incertain") ):
                    liste_DIFF_birds_undefined.append(liste_intersection.index(max_intersection))            
                if proportion_maximum>coverage_threshold and (  (classe=="lapin")   or (classe=="chevreuil") ):
                    liste_DIFF_other_animals.append(liste_intersection.index(max_intersection))           
        
    
            #Capture du nombre d'imagettes à caputerer et captutées
        
        birds_table=imagettes_target[ (imagettes_target["classe"]=="corneille"  ) | (imagettes_target["classe"]=="pigeon"  ) |(imagettes_target["classe"]=="faisan"  ) 
        | (imagettes_target["classe"]=="oiseau"  )| (imagettes_target["classe"]=="pie"  ) |      (imagettes_target["classe"]=="incertain"  )      ]
        nb_animals_match=len(liste_DIFF_birds_defined)+len(liste_DIFF_birds_undefined)+len(liste_DIFF_other_animals)
        nb_animals_to_find=len(imagettes_target)
        birds_to_find=len(birds_table)
        birds_defined_match=len(liste_DIFF_birds_defined)
        birds_match=birds_defined_match+len(liste_DIFF_birds_undefined)
        print( "nombre d'animaux à reperer", nb_animals_to_find)
        print( "nombre d'animaux repérés", nb_animals_match)
        print("nombre d'oiseaux dans l'image",birds_to_find)
        print("nombre d'oiseaux totaux repérés dans l'image",birds_match)
        print("nombre d'oiseaux repérés parmi les oiseaux labélisés",birds_defined_match )

  
        try:
                


            #Predicts birds 
            batchImages_match_birds = [batchImages_stack_reshape[i] for i in (liste_DIFF_birds_defined+liste_DIFF_birds_undefined)]
            batchImages_match_birds=np.vstack(batchImages_match_birds)
            estimates_match_brids = CNNmodel.predict(batchImages_match_birds.reshape(-1,28,28,3))
            liste_prediction_birds=list(estimates_match_brids.argmax(axis=1))
            #animals_predict=(len(liste_prediction)-liste_prediction.count(0))
            birds_predict=liste_prediction_birds.count(2)+liste_prediction_birds.count(3)+liste_prediction_birds.count(5)   

        except ValueError:
                print("il n'a pas d'imagettes d'oiseaux trouvées !")
                #pourcentage_TP=0
                birds_match,birds_predict,birds_defined_match,VTP=(0,0,0,0)
                liste_prediction_birds=[]
                birds_predict=0
        try:

            #Predicts  animals
            #probablement moyen de pas dédoubler le calcul
            #Il faut rajouter les oiseaux ...
            batchImages_match_other_animals = [batchImages_stack_reshape[i] for i in (liste_DIFF_other_animals)]
            batchImages_match_other_animals=np.vstack(batchImages_match_other_animals)
            estimates_match_other_animals = CNNmodel.predict(batchImages_match_other_animals.reshape(-1,28,28,3))
            liste_prediction_other_animals=list(estimates_match_other_animals.argmax(axis=1))
            animals_predict=(len(liste_prediction_other_animals)-liste_prediction_birds.count(0)) + (len(liste_prediction_birds)-liste_prediction_birds.count(0))
            #birds_predict=liste_prediction.count(2)+liste_prediction.count(3)+liste_prediction.count(5)     

        except ValueError:
                print("il n'a pas d'imagettes de lapins ou de chevreuils !")
                animals_predict=birds_predict
                #pourcentage_TP=0

            
            
        try:
            #Ici on pourrait probablement gagner du temps en n'utilisant pas le dic à chaque étape mais seulement au début
            #et en comptant le nombre de valeurs qui correspond au bon i
            #Determiner les vrai positifs à partir des images qui sont bien labélisées seulement
            batch_labels_defined = [batchImages_stack_reshape[i] for i in liste_DIFF_birds_defined]
            batch_labels_defined=np.vstack(batch_labels_defined)
            estimates_labels = CNNmodel.predict(batch_labels_defined.reshape(-1,28,28,3))
            liste_labels_prediction=list(estimates_labels.argmax(axis=1))            
            for i in range(len(liste_labels_prediction)):
                if dictionnaire_conversion[liste_prediction_birds[i]]==classe:
                    VTP+=1
    
        except ValueError:
                print("il n'a pas d'oiseaux identifiés")
                animals_predict=birds_predict
                #pourcentage_TP=0
    
    
        #On peut certainement encore baisser le nombre vu qu'on prend que les oiseaux
        #Predict False Positif
        liste_batch_images=range(len(batchImages_stack_reshape))
        liste_not_matche=set(liste_batch_images)-set(liste_DIFF_birds_defined)-set(liste_DIFF_birds_undefined)-set(liste_DIFF_other_animals)
        batchImages_match_not_animal = [batchImages_stack_reshape[i] for i in liste_not_matche]
        batchImages_match_not_animal=np.vstack(batchImages_match_not_animal)                 
        estimates_match_not_animals = CNNmodel.predict(batchImages_match_not_animal.reshape(-1,28,28,3))
        liste_prediction_not_match=list(estimates_match_not_animals.argmax(axis=1))
        FP=(len(liste_prediction_not_match)-liste_prediction_not_match.count(0))
        FP_birds=(liste_prediction_not_match.count(2)+ liste_prediction_not_match.count(3)+liste_prediction_not_match.count(5)  )
        nombre_imagettes=len(liste_prediction_not_match)
        pourcentage_FP=FP/nombre_imagettes
                
        #On va maintenant faire un tableau avec les ccordonées des carrés et leur prédictions
        table_coordinate_predict=generate_square.loc[liste_not_matche]
        table_coordinate_predict["class_predict"]=liste_prediction_not_match
        
        image_fp=imageB.copy()
        
        table=table_coordinate_predict
        for i in range(len(table)):   
            xmin=int(table.iloc[i,0])
            xmax=int(table.iloc[i,1])
            ymin=int(table.iloc[i,2])
            ymax=int(table.iloc[i,3])
            #cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2) 
            #if table.iloc[i,5]>=max_col:
            if table["class_predict"].iloc[i]==0:
                cv2.rectangle(image_fp, (xmin,ymin), (xmax,ymax), (255, 255, 255), 2) 
            else:
                cv2.rectangle(image_fp, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 
            
        cv2.imwrite("/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/test_filtre_quantile.jpg",image_fp)
        
        
        #retricir au non match en un tableau de n lignes à partir liste_not_match
        #après tout simplement dire que classe predict[i]= liste_prediction_not_match
        
        #Sinon on peut essayer de faire pour tous les predctions et ensuite filtre selon les différents liste other_animals_birds ... . 
    

    #except ValueError:
        #print("il n'a aucune différence entre les images comparées !")
        #pourcentage_TP=0
        #nb_animals_match,birds_match,birds_predict,birds_defined_match,VTP=(0,0,0,0,0)
        
    return generate_square,liste_not_matche,liste_prediction_not_match 









#falspe positive regression quantile to display the square of false positif after regression quantile
def fsq(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=19,blurFact=15,
            filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
    
    #Definition du dicionnaire de conversion
    dictionnaire_conversion={}
    dictionnaire_conversion[0]="autre"
    dictionnaire_conversion[1]="chevreuil"
    dictionnaire_conversion[2]="corneille"
    dictionnaire_conversion[3]="faisan"
    dictionnaire_conversion[4]="lapin"
    dictionnaire_conversion[5]="pigeon"
    #
    #Initialisation de variables et de liste
    batchImages = []
    liste_table = []
    imageSize= 28
    animals_predict=0
    birds_predict=0
    VTP=0
    nb_animals_match=0

    
    
    
    #Definition des images
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    
    
    #Ouverture des fichiers annotés 
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    imagettes=to_reference_labels (imagettes,"classe")
    
    
    #On ne conserve que les annotations dont le label se retrouve dans le Réseau de neurones
    #Eventuellement réincorpoer les animaux mais à priori pas d'intérêt
    imagettes=imagettes[  (imagettes["classe"]!="autre") & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") 
    & (imagettes["classe"]!="sanglier") & (imagettes["classe"]!="cheval") & (imagettes["classe"]!="ground") & (imagettes["classe"]!="autre") ]
    
    
    #Il faut enlever autre
    
    
    #imagettes_birds_indefined=imagettes[ (imagettes["classe"]=="oiseau") |  (imagettes["classe"]=="pie") ]    
    
    
    #On se place dans le bon dossier pour ces anotations (peut être inutile mais on sait jamais s'il y a des doublons à vérifier doublons pour imagettesname)
    #Il y 1485 imagettes différents sur l'ensemble du dossier contre 1487 pour chaque dossier pris séparement, probablement une faute dans le placement.
    folder_choosen="."+ folder
    imagettes_folder=imagettes[(imagettes["path"]==folder_choosen) ]
    #Attention ça ne fonction qu'avec le PI_0 pour les autres dossiers il faut rajouter d'autres classes 
    
    #imagettes_PI_0=imagettes_PI_0[   (imagettes_PI_0["classe"]!="ground") & (imagettes_PI_0["classe"]!="incertain")     ]    
    
    #On selectionne seulement pour la photo sur laquel on veut repérer les oiseaux ou autres animaux et on réarange les colonnes dans le bon ordre
    imagettes_target=imagettes_folder[imagettes_folder["filename"]==name_test]
    to_drop=['path', 'filename', 'width', 'height', 'index']
    imagettes_target=imagettes_target.drop(to_drop,axis=1)
    col = list(imagettes_target.columns)[-1:] + list(imagettes_target.columns)[:-1]
    imagettes_target=imagettes_target[col]
    
    
    #Peut être que je peux séparer ici imagettes_target en deux entre definied et indefiend
    
    
    #On regarde si il y a des imagettes de type différents pas forcément utiles surtout si on garde les oiseaux undefined
    if len(imagettes_target["classe"].unique())>1:
        print("attention le code le prend en charge" )
    #nom_classe=imagettes1["classe"].iloc[0]
    nom_classe=list(imagettes_target["classe"].unique())
    

    
    
    
    
    #On récupère les coordonnées des pixels différent par différence la colonne name parait vraiment optionnelle peut être la retirée
    #on obtien les batchs et une tables
    #Il y a un warning dans le block mais où
    cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        #name = "TOTO" + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))
    

    
    #try:
    azerty=0
    if azerty==0:
      
        batchImages_stack = np.vstack(batchImages)
        batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
        table_non_filtre = pd.DataFrame(np.vstack(liste_table))
        table_non_filtre = table_non_filtre.rename(columns={ 0: 'xmin', 1: 'xmax', 2: 'ymin', 3: 'ymax'})
        table_non_filtre=table_non_filtre.astype(int)
        
    
    
        
        #Maintenant on va procéder à un certain nombre de filtre sur la table (ici desactivé) pour ne proposer que les batch utiles selon l'index
        #Si on peut passer de la table au batch facilement il y a mieux à faire ... ou au mons à liste_table
        
        #Il faudra penser à soulever une expetion si rien trouver à l'isuu du filtre en particulier pour la ligne
        #fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
        #image_test image_2019-06-14_15-46-54.jpg image_ref image_2019-06-14_15-46-38.jpg

        table_quantile,index_possible_animals=filtre_quantile(table_non_filtre,coef_filtre,height=2448,width=3264)
        """
        table_filtre_RL=table_quantile.copy()
        table_filtre_RL["possible_bird"]=fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
        table_filtre_RL=(table_filtre_RL[table_filtre_RL["possible_bird"]=="O"])
        p_bird=table_filtre_RL.index
        table_filtre_RL.drop("possible_bird",axis=1,inplace=True)     
        index_possible_animals=list(set(index_possible_animals).intersection(p_bird)) 
        #batchImages_filtre = [batchImages_stack_reshape[i] for i in (index_possible_animals)]
        """
    
    
        #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
        #generate_square=table.iloc[:,1:5]
        if filtre_choice=="No_filtre":
            generate_square=table_non_filtre
            #generate_square=table_quantile
        elif filtre_choice=="quantile_filtre":
            generate_square=table_quantile
        elif filtre_choice=="RL_filtre":
            generate_square=table_filtre_RL
        


        #Ici on propose le tableau coord imagettes et classe predict
        





        xmin_gen=generate_square["xmin"]
        xmax_gen=generate_square["xmax"]
        ymin_gen=generate_square["ymin"]
        ymax_gen=generate_square["ymax"]
        ln_square_gen=len(generate_square)
    
       
        
        
        
        
        #C'est bien suspect d'avoir cette liste à l'interieur de la boucle
        liste_DIFF_birds_defined=[]
        liste_DIFF_birds_undefined=[]
        liste_DIFF_other_animals=[]
        #On classe les difference repéré pour les animaux classés
        for classe in nom_classe:
            
            #Ou sinon on peut exclure ici les oiseaux non defs... .
            imagettes_annote_1_classe=imagettes_target[imagettes_target["classe"]==classe]
            nb_imagettes_1_classe=len(imagettes_annote_1_classe)
            
            #get max intersection with square generate for each sqaure annotate
            for i in range(nb_imagettes_1_classe):
                x_min_anote=imagettes_annote_1_classe["xmin"].iloc[i]
                x_max_anote=imagettes_annote_1_classe["xmax"].iloc[i]
                y_min_anote=imagettes_annote_1_classe["ymin"].iloc[i]
                y_max_anote=imagettes_annote_1_classe["ymax"].iloc[i]
         
    
                #Replicated the coordinates of annotations the number time of the len  to be able to apply area_intersection function
                zip_xmin_anote=[x_min_anote]*ln_square_gen
                zip_xmax_anote=[x_max_anote]*ln_square_gen
                zip_ymin_anote=[y_min_anote]*ln_square_gen
                zip_ymax_anote=[y_max_anote]*ln_square_gen
            
                
                #apply function area_intersection to calculate the area intersected between all generate square for this annote_squared 
                #and get the square with the max intersectio if there is enough area in commun
                
                #Je vais encore séparer les listes ici pour proposer oiseaux définis et autres animaux... . 
                liste_intersection=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip 
                                    (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
                max_intersection=max(liste_intersection)
                proportion_maximum=max_intersection/area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
                if proportion_maximum>coverage_threshold and (  (classe=="faisan")   or (classe=="corneille") or (classe=="pigeon")):
                    #Il faudrait rajouter une boucle while tan que max_intersection>limit_area l'enlever de la liste
                #liste_DIFF2.append(max_intersection.index(proportion_maximum))
                    liste_DIFF_birds_defined.append(liste_intersection.index(max_intersection))
                if proportion_maximum>coverage_threshold and (  (classe=="oiseau")   or (classe=="pie") or (classe=="incertain") ):
                    liste_DIFF_birds_undefined.append(liste_intersection.index(max_intersection))            
                if proportion_maximum>coverage_threshold and (  (classe=="lapin")   or (classe=="chevreuil") ):
                    liste_DIFF_other_animals.append(liste_intersection.index(max_intersection))           
        
    
            #Capture du nombre d'imagettes à caputerer et captutées
        
        birds_table=imagettes_target[ (imagettes_target["classe"]=="corneille"  ) | (imagettes_target["classe"]=="pigeon"  ) |(imagettes_target["classe"]=="faisan"  ) 
        | (imagettes_target["classe"]=="oiseau"  )| (imagettes_target["classe"]=="pie"  ) |      (imagettes_target["classe"]=="incertain"  )      ]
        nb_animals_match=len(liste_DIFF_birds_defined)+len(liste_DIFF_birds_undefined)+len(liste_DIFF_other_animals)
        nb_animals_to_find=len(imagettes_target)
        birds_to_find=len(birds_table)
        birds_defined_match=len(liste_DIFF_birds_defined)
        birds_match=birds_defined_match+len(liste_DIFF_birds_undefined)
        print( "nombre d'animaux à reperer", nb_animals_to_find)
        print( "nombre d'animaux repérés", nb_animals_match)
        print("nombre d'oiseaux dans l'image",birds_to_find)
        print("nombre d'oiseaux totaux repérés dans l'image",birds_match)
        print("nombre d'oiseaux repérés parmi les oiseaux labélisés",birds_defined_match )

  
        try:
                
            

            #Predicts birds 
            batchImages_match_birds = [batchImages_stack_reshape[i] for i in (liste_DIFF_birds_defined+liste_DIFF_birds_undefined)]
            batchImages_match_birds=np.vstack(batchImages_match_birds)
            estimates_match_brids = CNNmodel.predict(batchImages_match_birds.reshape(-1,28,28,3))
            liste_prediction_birds=list(estimates_match_brids.argmax(axis=1))
            #animals_predict=(len(liste_prediction)-liste_prediction.count(0))
            birds_predict=liste_prediction_birds.count(2)+liste_prediction_birds.count(3)+liste_prediction_birds.count(5)   

        except ValueError:
                print("il n'a pas d'imagettes d'oiseaux trouvées !")
                #pourcentage_TP=0
                birds_match,birds_predict,birds_defined_match,VTP=(0,0,0,0)
                liste_prediction_birds=[]
                birds_predict=0
        try:

            #Predicts  animals
            #probablement moyen de pas dédoubler le calcul
            #Il faut rajouter les oiseaux ...
            batchImages_match_other_animals = [batchImages_stack_reshape[i] for i in (liste_DIFF_other_animals)]
            batchImages_match_other_animals=np.vstack(batchImages_match_other_animals)
            estimates_match_other_animals = CNNmodel.predict(batchImages_match_other_animals.reshape(-1,28,28,3))
            liste_prediction_other_animals=list(estimates_match_other_animals.argmax(axis=1))
            animals_predict=(len(liste_prediction_other_animals)-liste_prediction_birds.count(0)) + (len(liste_prediction_birds)-liste_prediction_birds.count(0))
            #birds_predict=liste_prediction.count(2)+liste_prediction.count(3)+liste_prediction.count(5)     

        except ValueError:
                print("il n'a pas d'imagettes de lapins ou de chevreuils !")
                animals_predict=birds_predict
                #pourcentage_TP=0

            
            
        try:
            #Ici on pourrait probablement gagner du temps en n'utilisant pas le dic à chaque étape mais seulement au début
            #et en comptant le nombre de valeurs qui correspond au bon i
            #Determiner les vrai positifs à partir des images qui sont bien labélisées seulement
            batch_labels_defined = [batchImages_stack_reshape[i] for i in liste_DIFF_birds_defined]
            batch_labels_defined=np.vstack(batch_labels_defined)
            estimates_labels = CNNmodel.predict(batch_labels_defined.reshape(-1,28,28,3))
            liste_labels_prediction=list(estimates_labels.argmax(axis=1))            
            for i in range(len(liste_labels_prediction)):
                if dictionnaire_conversion[liste_prediction_birds[i]]==classe:
                    VTP+=1
    
        except ValueError:
                print("il n'a pas d'oiseaux identifiés")
                animals_predict=birds_predict
                #pourcentage_TP=0
    
        """
        #On peut certainement encore baisser le nombre vu qu'on prend que les oiseaux
        #Predict False Positif
        liste_batch_images=range(len(batchImages_stack_reshape))
        liste_not_matche=set(liste_batch_images)-set(liste_DIFF_birds_defined)-set(liste_DIFF_birds_undefined)-set(liste_DIFF_other_animals)
        batchImages_match_not_animal = [batchImages_stack_reshape[i] for i in liste_not_matche]
        batchImages_match_not_animal=np.vstack(batchImages_match_not_animal)                 
        estimates_match_not_animals = CNNmodel.predict(batchImages_match_not_animal.reshape(-1,28,28,3))
        liste_prediction_not_match=list(estimates_match_not_animals.argmax(axis=1))
        FP=(len(liste_prediction_not_match)-liste_prediction_not_match.count(0))
        FP_birds=(liste_prediction_not_match.count(2)+ liste_prediction_not_match.count(3)+liste_prediction_not_match.count(5)  )
        nombre_imagettes=len(liste_prediction_not_match)
        pourcentage_FP=FP/nombre_imagettes """
                
        #On va maintenant faire un tableau avec les ccordonées des carrés et leur prédictions
        
        
                #Predict False Positif
        liste_batch_images=range(len(batchImages_stack_reshape))
        liste_not_matche=set(liste_batch_images)-set(liste_DIFF_birds_defined)-set(liste_DIFF_birds_undefined)-set(liste_DIFF_other_animals)
        #On vient d'insérer cette ligne
        possible_fp=list(set(index_possible_animals).intersection(liste_not_matche))
        batchImages_match_not_animal = [batchImages_stack_reshape[i] for i in possible_fp]
        batchImages_match_not_animal=np.vstack(batchImages_match_not_animal)                 
        estimates_match_not_animals = CNNmodel.predict(batchImages_match_not_animal.reshape(-1,28,28,3))
        liste_prediction_not_match=list(estimates_match_not_animals.argmax(axis=1))
        
        FP=(len(liste_prediction_not_match)-liste_prediction_not_match.count(0))
        FP_birds=(liste_prediction_not_match.count(2)+ liste_prediction_not_match.count(3)+liste_prediction_not_match.count(5)  )
        nombre_imagettes=len(liste_prediction_not_match)
        pourcentage_FP=FP/nombre_imagettes
        
        
        
        #table_coordinate_predict=generate_square.loc[liste_not_matche]
        
      
        table_coordinate_predict=generate_square.loc[possible_fp]
        table_coordinate_predict["class_predict"]=liste_prediction_not_match
        
        image_fp=imageB.copy()
        
        table=table_coordinate_predict
        for i in range(len(table)):   
            xmin=int(table.iloc[i,0])
            xmax=int(table.iloc[i,1])
            ymin=int(table.iloc[i,2])
            ymax=int(table.iloc[i,3])
            #cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2) 
            #if table.iloc[i,5]>=max_col:
            if table["class_predict"].iloc[i]==0:
                cv2.rectangle(image_fp, (xmin,ymin), (xmax,ymax), (255, 255, 255), 2) 
            else:
                cv2.rectangle(image_fp, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 
            
        cv2.imwrite("/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/test_filtre_quantile.jpg",image_fp)
        
        
        #retricir au non match en un tableau de n lignes à partir liste_not_match
        #après tout simplement dire que classe predict[i]= liste_prediction_not_match
        
        #Sinon on peut essayer de faire pour tous les predctions et ensuite filtre selon les différents liste other_animals_birds ... . 
    

    #except ValueError:
        #print("il n'a aucune différence entre les images comparées !")
        #pourcentage_TP=0
        #nb_animals_match,birds_match,birds_predict,birds_defined_match,VTP=(0,0,0,0,0)
        
    return generate_square,liste_not_matche,liste_prediction_not_match 



#find what is the mean and distribution probability of fp 
def threshold_fp(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=19,blurFact=15,
            filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
    
    #Definition du dicionnaire de conversion
    dictionnaire_conversion={}
    dictionnaire_conversion[0]="autre"
    dictionnaire_conversion[1]="chevreuil"
    dictionnaire_conversion[2]="corneille"
    dictionnaire_conversion[3]="faisan"
    dictionnaire_conversion[4]="lapin"
    dictionnaire_conversion[5]="pigeon"
    #
    #Initialisation de variables et de liste
    batchImages = []
    liste_table = []
    imageSize= 28
    animals_predict=0
    birds_predict=0
    VTP=0
    nb_animals_match=0

    
    
    
    #Definition des images
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    
    
    #Ouverture des fichiers annotés 
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    imagettes=to_reference_labels (imagettes,"classe")
    
    
    #On ne conserve que les annotations dont le label se retrouve dans le Réseau de neurones
    #Eventuellement réincorpoer les animaux mais à priori pas d'intérêt
    imagettes=imagettes[  (imagettes["classe"]!="autre") & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") 
    & (imagettes["classe"]!="sanglier") & (imagettes["classe"]!="cheval") & (imagettes["classe"]!="ground") & (imagettes["classe"]!="autre") ]
    
    
    #Il faut enlever autre
    
    
    #imagettes_birds_indefined=imagettes[ (imagettes["classe"]=="oiseau") |  (imagettes["classe"]=="pie") ]    
    
    
    #On se place dans le bon dossier pour ces anotations (peut être inutile mais on sait jamais s'il y a des doublons à vérifier doublons pour imagettesname)
    #Il y 1485 imagettes différents sur l'ensemble du dossier contre 1487 pour chaque dossier pris séparement, probablement une faute dans le placement.
    folder_choosen="."+ folder
    imagettes_folder=imagettes[(imagettes["path"]==folder_choosen) ]
    #Attention ça ne fonction qu'avec le PI_0 pour les autres dossiers il faut rajouter d'autres classes 
    
    #imagettes_PI_0=imagettes_PI_0[   (imagettes_PI_0["classe"]!="ground") & (imagettes_PI_0["classe"]!="incertain")     ]    
    
    #On selectionne seulement pour la photo sur laquel on veut repérer les oiseaux ou autres animaux et on réarange les colonnes dans le bon ordre
    imagettes_target=imagettes_folder[imagettes_folder["filename"]==name_test]
    to_drop=['path', 'filename', 'width', 'height', 'index']
    imagettes_target=imagettes_target.drop(to_drop,axis=1)
    col = list(imagettes_target.columns)[-1:] + list(imagettes_target.columns)[:-1]
    imagettes_target=imagettes_target[col]
    
    
    #Peut être que je peux séparer ici imagettes_target en deux entre definied et indefiend
    
    
    #On regarde si il y a des imagettes de type différents pas forcément utiles surtout si on garde les oiseaux undefined
    if len(imagettes_target["classe"].unique())>1:
        print("attention le code le prend en charge" )
    #nom_classe=imagettes1["classe"].iloc[0]
    nom_classe=list(imagettes_target["classe"].unique())
    

    
    
    
    
    #On récupère les coordonnées des pixels différent par différence la colonne name parait vraiment optionnelle peut être la retirée
    #on obtien les batchs et une tables
    #Il y a un warning dans le block mais où
    cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        #name = "TOTO" + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))
    

    
    #try:
    azerty=0
    if azerty==0:
      
        batchImages_stack = np.vstack(batchImages)
        batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
        table_non_filtre = pd.DataFrame(np.vstack(liste_table))
        table_non_filtre = table_non_filtre.rename(columns={ 0: 'xmin', 1: 'xmax', 2: 'ymin', 3: 'ymax'})
        table_non_filtre=table_non_filtre.astype(int)
        
    
    
        
        #Maintenant on va procéder à un certain nombre de filtre sur la table (ici desactivé) pour ne proposer que les batch utiles selon l'index
        #Si on peut passer de la table au batch facilement il y a mieux à faire ... ou au mons à liste_table
        
        #Il faudra penser à soulever une expetion si rien trouver à l'isuu du filtre en particulier pour la ligne
        #fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
        #image_test image_2019-06-14_15-46-54.jpg image_ref image_2019-06-14_15-46-38.jpg

        table_quantile,index_possible_animals=filtre_quantile(table_non_filtre,coef_filtre,height=2448,width=3264)

    
    
        #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
        #generate_square=table.iloc[:,1:5]
        if filtre_choice=="No_filtre":
            generate_square=table_non_filtre
            #generate_square=table_quantile
        elif filtre_choice=="quantile_filtre":
            generate_square=table_quantile
        elif filtre_choice=="RL_filtre":
            generate_square=table_filtre_RL
        


        #Ici on propose le tableau coord imagettes et classe predict
        





        xmin_gen=generate_square["xmin"]
        xmax_gen=generate_square["xmax"]
        ymin_gen=generate_square["ymin"]
        ymax_gen=generate_square["ymax"]
        ln_square_gen=len(generate_square)
    
       
        
        
        
        
        #C'est bien suspect d'avoir cette liste à l'interieur de la boucle
        liste_DIFF_birds_defined=[]
        liste_DIFF_birds_undefined=[]
        liste_DIFF_other_animals=[]
        #On classe les difference repéré pour les animaux classés
        for classe in nom_classe:
            
            #Ou sinon on peut exclure ici les oiseaux non defs... .
            imagettes_annote_1_classe=imagettes_target[imagettes_target["classe"]==classe]
            nb_imagettes_1_classe=len(imagettes_annote_1_classe)
            
            #get max intersection with square generate for each sqaure annotate
            for i in range(nb_imagettes_1_classe):
                x_min_anote=imagettes_annote_1_classe["xmin"].iloc[i]
                x_max_anote=imagettes_annote_1_classe["xmax"].iloc[i]
                y_min_anote=imagettes_annote_1_classe["ymin"].iloc[i]
                y_max_anote=imagettes_annote_1_classe["ymax"].iloc[i]
         
    
                #Replicated the coordinates of annotations the number time of the len  to be able to apply area_intersection function
                zip_xmin_anote=[x_min_anote]*ln_square_gen
                zip_xmax_anote=[x_max_anote]*ln_square_gen
                zip_ymin_anote=[y_min_anote]*ln_square_gen
                zip_ymax_anote=[y_max_anote]*ln_square_gen
            
                
                #apply function area_intersection to calculate the area intersected between all generate square for this annote_squared 
                #and get the square with the max intersectio if there is enough area in commun
                
                #Je vais encore séparer les listes ici pour proposer oiseaux définis et autres animaux... . 
                liste_intersection=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip 
                                    (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
                max_intersection=max(liste_intersection)
                proportion_maximum=max_intersection/area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
                if proportion_maximum>coverage_threshold and (  (classe=="faisan")   or (classe=="corneille") or (classe=="pigeon")):
                    #Il faudrait rajouter une boucle while tan que max_intersection>limit_area l'enlever de la liste
                #liste_DIFF2.append(max_intersection.index(proportion_maximum))
                    liste_DIFF_birds_defined.append(liste_intersection.index(max_intersection))
                if proportion_maximum>coverage_threshold and (  (classe=="oiseau")   or (classe=="pie") or (classe=="incertain") ):
                    liste_DIFF_birds_undefined.append(liste_intersection.index(max_intersection))            
                if proportion_maximum>coverage_threshold and (  (classe=="lapin")   or (classe=="chevreuil") ):
                    liste_DIFF_other_animals.append(liste_intersection.index(max_intersection))           
        
    
            #Capture du nombre d'imagettes à caputerer et captutées
        
        birds_table=imagettes_target[ (imagettes_target["classe"]=="corneille"  ) | (imagettes_target["classe"]=="pigeon"  ) |(imagettes_target["classe"]=="faisan"  ) 
        | (imagettes_target["classe"]=="oiseau"  )| (imagettes_target["classe"]=="pie"  ) |      (imagettes_target["classe"]=="incertain"  )      ]
        nb_animals_match=len(liste_DIFF_birds_defined)+len(liste_DIFF_birds_undefined)+len(liste_DIFF_other_animals)
        nb_animals_to_find=len(imagettes_target)
        birds_to_find=len(birds_table)
        birds_defined_match=len(liste_DIFF_birds_defined)
        birds_match=birds_defined_match+len(liste_DIFF_birds_undefined)
        print( "nombre d'animaux à reperer", nb_animals_to_find)
        print( "nombre d'animaux repérés", nb_animals_match)
        print("nombre d'oiseaux dans l'image",birds_to_find)
        print("nombre d'oiseaux totaux repérés dans l'image",birds_match)
        print("nombre d'oiseaux repérés parmi les oiseaux labélisés",birds_defined_match )

  
        try:
                


            #Predicts birds 
            if filtre_choice=="quantile_filtre":
                liste_DIFF_birds_defined=list(set(liste_DIFF_birds_defined).intersection(index_possible_animals))
                liste_DIFF_birds_undefined=list(set(liste_DIFF_birds_undefined).intersection(index_possible_animals))
                
            batchImages_match_birds = [batchImages_stack_reshape[i] for i in (liste_DIFF_birds_defined+liste_DIFF_birds_undefined)]
            batchImages_match_birds=np.vstack(batchImages_match_birds)
            estimates_match_brids = CNNmodel.predict(batchImages_match_birds.reshape(-1,28,28,3))
            liste_prediction_birds=list(estimates_match_brids.argmax(axis=1))
            estimates_match_brids =list(map(max, estimates_match_brids ))
            #animals_predict=(len(liste_prediction)-liste_prediction.count(0))
            birds_predict=liste_prediction_birds.count(2)+liste_prediction_birds.count(3)+liste_prediction_birds.count(5)   

        except ValueError:
                print("il n'a pas d'imagettes d'oiseaux trouvées !")
                #pourcentage_TP=0
                birds_match,birds_predict,birds_defined_match,VTP=(0,0,0,0)
                liste_prediction_birds=[]
                birds_predict=0
                estimates_match_brids =[]
        try:

            #Predicts  animals
            #probablement moyen de pas dédoubler le calcul
            #Il faut rajouter les oiseaux ...
            batchImages_match_other_animals = [batchImages_stack_reshape[i] for i in (liste_DIFF_other_animals)]
            batchImages_match_other_animals=np.vstack(batchImages_match_other_animals)
            estimates_match_other_animals = CNNmodel.predict(batchImages_match_other_animals.reshape(-1,28,28,3))
            liste_prediction_other_animals=list(estimates_match_other_animals.argmax(axis=1))
            animals_predict=(len(liste_prediction_other_animals)-liste_prediction_birds.count(0)) + (len(liste_prediction_birds)-liste_prediction_birds.count(0))
            #birds_predict=liste_prediction.count(2)+liste_prediction.count(3)+liste_prediction.count(5)     

        except ValueError:
                print("il n'a pas d'imagettes de lapins ou de chevreuils !")
                animals_predict=birds_predict
                #pourcentage_TP=0

            
            
        try:
            #Ici on pourrait probablement gagner du temps en n'utilisant pas le dic à chaque étape mais seulement au début
            #et en comptant le nombre de valeurs qui correspond au bon i
            #Determiner les vrai positifs à partir des images qui sont bien labélisées seulement
            batch_labels_defined = [batchImages_stack_reshape[i] for i in liste_DIFF_birds_defined]
            batch_labels_defined=np.vstack(batch_labels_defined)
            estimates_labels = CNNmodel.predict(batch_labels_defined.reshape(-1,28,28,3))
            liste_labels_prediction=list(estimates_labels.argmax(axis=1))            
            for i in range(len(liste_labels_prediction)):
                if dictionnaire_conversion[liste_prediction_birds[i]]==classe:
                    VTP+=1
    
        except ValueError:
                print("il n'a pas d'oiseaux identifiés")
                animals_predict=birds_predict
                #pourcentage_TP=0
    

                
        #On va maintenant faire un tableau avec les ccordonées des carrés et leur prédictions
        
        
        
        
        
        
        
        
        
                #Predict False Positif
        liste_batch_images=range(len(batchImages_stack_reshape))
        liste_not_matche=set(liste_batch_images)-set(liste_DIFF_birds_defined)-set(liste_DIFF_birds_undefined)-set(liste_DIFF_other_animals)
        
        
        
        #Select only birds prediction match and not match
        
        estimates_positve = CNNmodel.predict(batchImages_stack_reshape.reshape(-1,28,28,3))
        liste_labels_prediction=list(estimates_positve.argmax(axis=1))            
        liste_index_positif=[liste_labels_prediction[i] for i in liste_labels_prediction]
        
        
        #On vient d'insérer cette ligne
        possible_fp=list(set(index_possible_animals).intersection(liste_not_matche))
        batchImages_match_not_animal = [batchImages_stack_reshape[i] for i in possible_fp]
        batchImages_match_not_animal=np.vstack(batchImages_match_not_animal)                 
        estimates_match_not_animals = CNNmodel.predict(batchImages_match_not_animal.reshape(-1,28,28,3))
        liste_prediction_not_match=list(estimates_match_not_animals.argmax(axis=1))
        
        #On va maintenant sélectionner les indexs et éléments faux négatifs
        liste_VN=[]
        for i, j in enumerate(liste_prediction_not_match):
            if j == 0:
                liste_VN.append(i)
        liste_FP=set(range(len(liste_prediction_not_match)))-  set(liste_VN)
        
        if filtre_choice=="quantile_filtre":
            liste_FP=list(set(liste_FP).intersection(index_possible_animals))
        estimates_FP = [max(estimates_match_not_animals[i]) for i in (liste_FP)]
        
        
        #Prochaine étape simplement trouver la probab max de chaque fp.
        
        
        
        
        FP=(len(liste_prediction_not_match)-liste_prediction_not_match.count(0))
        FP_birds=(liste_prediction_not_match.count(2)+ liste_prediction_not_match.count(3)+liste_prediction_not_match.count(5)  )
        nombre_imagettes=len(liste_prediction_not_match)
        pourcentage_FP=FP/nombre_imagettes
        
        
        
        #table_coordinate_predict=generate_square.loc[liste_not_matche]
        
      
        table_coordinate_predict=generate_square.loc[possible_fp]
        table_coordinate_predict["class_predict"]=liste_prediction_not_match
        

        table=table_coordinate_predict



        
    return estimates_FP,estimates_match_brids









def extract_best_fp(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=19,blurFact=15,
            filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
    
    #Definition du dicionnaire de conversion
    dictionnaire_conversion={}
    dictionnaire_conversion[0]="autre"
    dictionnaire_conversion[1]="chevreuil"
    dictionnaire_conversion[2]="corneille"
    dictionnaire_conversion[3]="faisan"
    dictionnaire_conversion[4]="lapin"
    dictionnaire_conversion[5]="pigeon"
    #
    #Initialisation de variables et de liste
    batchImages = []
    liste_table = []
    imageSize= 28
    animals_predict=0
    birds_predict=0
    VTP=0
    nb_animals_match=0

    
    
    
    
    #Definition des images
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    imageRectangle=imageB.copy()
    
    #Ouverture des fichiers annotés 
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    imagettes=to_reference_labels (imagettes,"classe")
    
    
    #On ne conserve que les annotations dont le label se retrouve dans le Réseau de neurones
    #Eventuellement réincorpoer les animaux mais à priori pas d'intérêt
    imagettes=imagettes[  (imagettes["classe"]!="autre") & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") 
    & (imagettes["classe"]!="sanglier") & (imagettes["classe"]!="cheval") & (imagettes["classe"]!="ground") & (imagettes["classe"]!="autre") ]
    
    
    #Il faut enlever autre
    
    
    #imagettes_birds_indefined=imagettes[ (imagettes["classe"]=="oiseau") |  (imagettes["classe"]=="pie") ]    
    
    
    #On se place dans le bon dossier pour ces anotations (peut être inutile mais on sait jamais s'il y a des doublons à vérifier doublons pour imagettesname)
    #Il y 1485 imagettes différents sur l'ensemble du dossier contre 1487 pour chaque dossier pris séparement, probablement une faute dans le placement.
    folder_choosen="."+ folder
    imagettes_folder=imagettes[(imagettes["path"]==folder_choosen) ]
    #Attention ça ne fonction qu'avec le PI_0 pour les autres dossiers il faut rajouter d'autres classes 
    
    #imagettes_PI_0=imagettes_PI_0[   (imagettes_PI_0["classe"]!="ground") & (imagettes_PI_0["classe"]!="incertain")     ]    
    
    #On selectionne seulement pour la photo sur laquel on veut repérer les oiseaux ou autres animaux et on réarange les colonnes dans le bon ordre
    imagettes_target=imagettes_folder[imagettes_folder["filename"]==name_test]
    to_drop=['path', 'filename', 'width', 'height', 'index']
    imagettes_target=imagettes_target.drop(to_drop,axis=1)
    col = list(imagettes_target.columns)[-1:] + list(imagettes_target.columns)[:-1]
    imagettes_target=imagettes_target[col]
    
    
    #Peut être que je peux séparer ici imagettes_target en deux entre definied et indefiend
    
    
    #On regarde si il y a des imagettes de type différents pas forcément utiles surtout si on garde les oiseaux undefined
    if len(imagettes_target["classe"].unique())>1:
        print("attention le code le prend en charge" )
    #nom_classe=imagettes1["classe"].iloc[0]
    nom_classe=list(imagettes_target["classe"].unique())
    

    
    
    
    
    #On récupère les coordonnées des pixels différent par différence la colonne name parait vraiment optionnelle peut être la retirée
    #on obtien les batchs et une tables
    #Il y a un warning dans le block mais où
    cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        #name = "TOTO" + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))
    

    
    #try:
    azerty=0
    if azerty==0:
      
        batchImages_stack = np.vstack(batchImages)
        batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
        table_non_filtre = pd.DataFrame(np.vstack(liste_table))
        table_non_filtre = table_non_filtre.rename(columns={ 0: 'xmin', 1: 'xmax', 2: 'ymin', 3: 'ymax'})
        table_non_filtre=table_non_filtre.astype(int)
        
    
    
        
        #Maintenant on va procéder à un certain nombre de filtre sur la table (ici desactivé) pour ne proposer que les batch utiles selon l'index
        #Si on peut passer de la table au batch facilement il y a mieux à faire ... ou au mons à liste_table
        
        #Il faudra penser à soulever une expetion si rien trouver à l'isuu du filtre en particulier pour la ligne
        #fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
        #image_test image_2019-06-14_15-46-54.jpg image_ref image_2019-06-14_15-46-38.jpg

        table_quantile,index_possible_animals=filtre_quantile(table_non_filtre,coef_filtre,height=2448,width=3264)

    
    
        #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
        #generate_square=table.iloc[:,1:5]
        if filtre_choice=="No_filtre":
            generate_square=table_non_filtre
            #generate_square=table_quantile
        elif filtre_choice=="quantile_filtre":
            generate_square=table_quantile
        elif filtre_choice=="RL_filtre":
            generate_square=table_filtre_RL
        


        #Ici on propose le tableau coord imagettes et classe predict
        





        xmin_gen=generate_square["xmin"]
        xmax_gen=generate_square["xmax"]
        ymin_gen=generate_square["ymin"]
        ymax_gen=generate_square["ymax"]
        ln_square_gen=len(generate_square)
    
       
        
        #imageRectangle
        
        #C'est bien suspect d'avoir cette liste à l'interieur de la boucle
        liste_DIFF_birds_defined=[]
        liste_DIFF_birds_undefined=[]
        liste_DIFF_other_animals=[]
        #On classe les difference repéré pour les animaux classés
        for classe in nom_classe:
            
            #Ou sinon on peut exclure ici les oiseaux non defs... .
            imagettes_annote_1_classe=imagettes_target[imagettes_target["classe"]==classe]
            nb_imagettes_1_classe=len(imagettes_annote_1_classe)
            
            #get max intersection with square generate for each sqaure annotate
            for i in range(nb_imagettes_1_classe):
                x_min_anote=imagettes_annote_1_classe["xmin"].iloc[i]
                x_max_anote=imagettes_annote_1_classe["xmax"].iloc[i]
                y_min_anote=imagettes_annote_1_classe["ymin"].iloc[i]
                y_max_anote=imagettes_annote_1_classe["ymax"].iloc[i]
         
            
                
                #cv2.rectangle(imageRectangle, (x_min_anote,y_min_anote), (x_max_anote,y_max_anote), (255, 0, 0), 2) 
    
                #Replicated the coordinates of annotations the number time of the len  to be able to apply area_intersection function
                zip_xmin_anote=[x_min_anote]*ln_square_gen
                zip_xmax_anote=[x_max_anote]*ln_square_gen
                zip_ymin_anote=[y_min_anote]*ln_square_gen
                zip_ymax_anote=[y_max_anote]*ln_square_gen
            
                
                #apply function area_intersection to calculate the area intersected between all generate square for this annote_squared 
                #and get the square with the max intersectio if there is enough area in commun
                
                #Je vais encore séparer les listes ici pour proposer oiseaux définis et autres animaux... . 
                liste_intersection=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip 
                                    (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
                max_intersection=max(liste_intersection)
                proportion_maximum=max_intersection/area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
                if proportion_maximum>coverage_threshold and (  (classe=="faisan")   or (classe=="corneille") or (classe=="pigeon")):
                    #Il faudrait rajouter une boucle while tan que max_intersection>limit_area l'enlever de la liste
                #liste_DIFF2.append(max_intersection.index(proportion_maximum))
                    liste_DIFF_birds_defined.append(liste_intersection.index(max_intersection))
                if proportion_maximum>coverage_threshold and (  (classe=="oiseau")   or (classe=="pie") or (classe=="incertain") ):
                    liste_DIFF_birds_undefined.append(liste_intersection.index(max_intersection))            
                if proportion_maximum>coverage_threshold and (  (classe=="lapin")   or (classe=="chevreuil") ):
                    liste_DIFF_other_animals.append(liste_intersection.index(max_intersection))           
        
    
            #Capture du nombre d'imagettes à caputerer et captutées
        
        birds_table=imagettes_target[ (imagettes_target["classe"]=="corneille"  ) | (imagettes_target["classe"]=="pigeon"  ) |(imagettes_target["classe"]=="faisan"  ) 
        | (imagettes_target["classe"]=="oiseau"  )| (imagettes_target["classe"]=="pie"  ) |      (imagettes_target["classe"]=="incertain"  )      ]
        nb_animals_match=len(liste_DIFF_birds_defined)+len(liste_DIFF_birds_undefined)+len(liste_DIFF_other_animals)
        nb_animals_to_find=len(imagettes_target)
        birds_to_find=len(birds_table)
        birds_defined_match=len(liste_DIFF_birds_defined)
        birds_match=birds_defined_match+len(liste_DIFF_birds_undefined)
        print( "nombre d'animaux à reperer", nb_animals_to_find)
        print( "nombre d'animaux repérés", nb_animals_match)
        print("nombre d'oiseaux dans l'image",birds_to_find)
        print("nombre d'oiseaux totaux repérés dans l'image",birds_match)
        print("nombre d'oiseaux repérés parmi les oiseaux labélisés",birds_defined_match )

  
        try:
                


            #Predicts birds 
            batchImages_match_birds = [batchImages_stack_reshape[i] for i in (liste_DIFF_birds_defined+liste_DIFF_birds_undefined)]
            batchImages_match_birds=np.vstack(batchImages_match_birds)
            estimates_match_brids = CNNmodel.predict(batchImages_match_birds.reshape(-1,28,28,3))
            liste_prediction_birds=list(estimates_match_brids.argmax(axis=1))
            estimates_match_brids =list(map(max, estimates_match_brids ))
            #animals_predict=(len(liste_prediction)-liste_prediction.count(0))
            birds_predict=liste_prediction_birds.count(2)+liste_prediction_birds.count(3)+liste_prediction_birds.count(5)   

        except ValueError:
                print("il n'a pas d'imagettes d'oiseaux trouvées !")
                #pourcentage_TP=0
                birds_match,birds_predict,birds_defined_match,VTP=(0,0,0,0)
                liste_prediction_birds=[]
                birds_predict=0
                estimates_match_brids =[]
        try:

            #Predicts  animals
            #probablement moyen de pas dédoubler le calcul
            #Il faut rajouter les oiseaux ...
            batchImages_match_other_animals = [batchImages_stack_reshape[i] for i in (liste_DIFF_other_animals)]
            batchImages_match_other_animals=np.vstack(batchImages_match_other_animals)
            estimates_match_other_animals = CNNmodel.predict(batchImages_match_other_animals.reshape(-1,28,28,3))
            liste_prediction_other_animals=list(estimates_match_other_animals.argmax(axis=1))
            animals_predict=(len(liste_prediction_other_animals)-liste_prediction_birds.count(0)) + (len(liste_prediction_birds)-liste_prediction_birds.count(0))
            #birds_predict=liste_prediction.count(2)+liste_prediction.count(3)+liste_prediction.count(5)     

        except ValueError:
                print("il n'a pas d'imagettes de lapins ou de chevreuils !")
                animals_predict=birds_predict
                #pourcentage_TP=0

            
            
        try:
            #Ici on pourrait probablement gagner du temps en n'utilisant pas le dic à chaque étape mais seulement au début
            #et en comptant le nombre de valeurs qui correspond au bon i
            #Determiner les vrai positifs à partir des images qui sont bien labélisées seulement
            batch_labels_defined = [batchImages_stack_reshape[i] for i in liste_DIFF_birds_defined]
            batch_labels_defined=np.vstack(batch_labels_defined)
            estimates_labels = CNNmodel.predict(batch_labels_defined.reshape(-1,28,28,3))
            liste_labels_prediction=list(estimates_labels.argmax(axis=1))            
            for i in range(len(liste_labels_prediction)):
                if dictionnaire_conversion[liste_prediction_birds[i]]==classe:
                    VTP+=1
    
        except ValueError:
                print("il n'a pas d'oiseaux identifiés")
                animals_predict=birds_predict
                #pourcentage_TP=0
    

                
        #On va maintenant faire un tableau avec les ccordonées des carrés et leur prédictions
        
        
        
        
        
        
        
        
        
                #Predict False Positif
        liste_batch_images=range(len(batchImages_stack_reshape))
        liste_not_matche=set(liste_batch_images)-set(liste_DIFF_birds_defined)-set(liste_DIFF_birds_undefined)-set(liste_DIFF_other_animals)
        #On vient d'insérer cette ligne
        possible_fp=list(set(index_possible_animals).intersection(liste_not_matche))
        batchImages_match_not_animal = [batchImages_stack_reshape[i] for i in possible_fp]
        batchImages_match_not_animal=np.vstack(batchImages_match_not_animal)                 
        estimates_match_not_animals = CNNmodel.predict(batchImages_match_not_animal.reshape(-1,28,28,3))
        liste_prediction_not_match=list(estimates_match_not_animals.argmax(axis=1))
        
        #On va maintenant sélectionner les indexs et éléments faux négatifs
        liste_VN=[]
        for i, j in enumerate(liste_prediction_not_match):
            if j == 0:
                liste_VN.append(i)
        liste_FP=set(range(len(liste_prediction_not_match)))-  set(liste_VN)
        
        estimates_FP = [max(estimates_match_not_animals[i]) for i in (liste_FP)]
        
        
        #Prochaine étape simplement trouver la probab max de chaque fp.
        
        
        
        
        FP=(len(liste_prediction_not_match)-liste_prediction_not_match.count(0))
        FP_birds=(liste_prediction_not_match.count(2)+ liste_prediction_not_match.count(3)+liste_prediction_not_match.count(5)  )
        nombre_imagettes=len(liste_prediction_not_match)
        pourcentage_FP=FP/nombre_imagettes
        
        
        
        #table_coordinate_predict=generate_square.loc[liste_not_matche]
        
      
        table_coordinate_predict=generate_square.loc[possible_fp]
        table_coordinate_predict["class_predict"]=liste_prediction_not_match
        

        table=table_coordinate_predict



        
    return estimates_FP,estimates_match_brids








def reorganize_predict(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=19,blurFact=15,
            filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
    

    #
    #Initialisation de variables et de liste
    batchImages = []
    liste_table = []
    imageSize= 28
    animals_predict=0
    birds_predict=0
    VTP=0
    nb_animals_match=0

    
    
    
    #Definition des images
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    
    
    #Ouverture des fichiers annotés 
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    imagettes=to_reference_labels (imagettes,"classe")
    
    
    #On ne conserve que les annotations dont le label se retrouve dans le Réseau de neurones
    #Eventuellement réincorpoer les animaux mais à priori pas d'intérêt
    imagettes=imagettes[  (imagettes["classe"]!="autre") & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") 
    & (imagettes["classe"]!="sanglier") & (imagettes["classe"]!="cheval") & (imagettes["classe"]!="ground") & (imagettes["classe"]!="autre") ]
    
    
    #Il faut enlever autre
    
    
    #imagettes_birds_indefined=imagettes[ (imagettes["classe"]=="oiseau") |  (imagettes["classe"]=="pie") ]    
    
    
    #On se place dans le bon dossier pour ces anotations (peut être inutile mais on sait jamais s'il y a des doublons à vérifier doublons pour imagettesname)
    #Il y 1485 imagettes différents sur l'ensemble du dossier contre 1487 pour chaque dossier pris séparement, probablement une faute dans le placement.
    folder_choosen="."+ folder
    imagettes_folder=imagettes[(imagettes["path"]==folder_choosen) ]
    #Attention ça ne fonction qu'avec le PI_0 pour les autres dossiers il faut rajouter d'autres classes 
    
    #imagettes_PI_0=imagettes_PI_0[   (imagettes_PI_0["classe"]!="ground") & (imagettes_PI_0["classe"]!="incertain")     ]    
    
    #On selectionne seulement pour la photo sur laquel on veut repérer les oiseaux ou autres animaux et on réarange les colonnes dans le bon ordre
    imagettes_target=imagettes_folder[imagettes_folder["filename"]==name_test]
    to_drop=['path', 'filename', 'width', 'height', 'index']
    imagettes_target=imagettes_target.drop(to_drop,axis=1)
    col = list(imagettes_target.columns)[-1:] + list(imagettes_target.columns)[:-1]
    imagettes_target=imagettes_target[col]
    
    
    #Peut être que je peux séparer ici imagettes_target en deux entre definied et indefiend
    
    
    #On regarde si il y a des imagettes de type différents pas forcément utiles surtout si on garde les oiseaux undefined
    if len(imagettes_target["classe"].unique())>1:
        print("attention le code le prend en charge" )
    #nom_classe=imagettes1["classe"].iloc[0]
    nom_classe=list(imagettes_target["classe"].unique())
    

    
    
    
    
    #On récupère les coordonnées des pixels différent par différence la colonne name parait vraiment optionnelle peut être la retirée
    #on obtien les batchs et une tables
    #Il y a un warning dans le block mais où
    cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        #name = "TOTO" + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))
    

    
    #try:
    azerty=0
    if azerty==0:
      
        batchImages_stack = np.vstack(batchImages)
        batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
        table_non_filtre = pd.DataFrame(np.vstack(liste_table))
        table_non_filtre = table_non_filtre.rename(columns={ 0: 'xmin', 1: 'xmax', 2: 'ymin', 3: 'ymax'})
        table_non_filtre=table_non_filtre.astype(int)
        
    
    
        
        #Maintenant on va procéder à un certain nombre de filtre sur la table (ici desactivé) pour ne proposer que les batch utiles selon l'index
        #Si on peut passer de la table au batch facilement il y a mieux à faire ... ou au mons à liste_table
        
        #Il faudra penser à soulever une expetion si rien trouver à l'isuu du filtre en particulier pour la ligne
        #fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
        #image_test image_2019-06-14_15-46-54.jpg image_ref image_2019-06-14_15-46-38.jpg

        #table_quantile,index_possible_animals=fn.filtre_quantile(table_non_filtre,coef_filtre,height=2448,width=3264)
        """
        table_filtre_RL=table_quantile.copy()
        table_filtre_RL["possible_bird"]=fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
        table_filtre_RL=(table_filtre_RL[table_filtre_RL["possible_bird"]=="O"])
        p_bird=table_filtre_RL.index
        table_filtre_RL.drop("possible_bird",axis=1,inplace=True)     
        index_possible_animals=list(set(index_possible_animals).intersection(p_bird)) 
        #batchImages_filtre = [batchImages_stack_reshape[i] for i in (index_possible_animals)]
        """
    
    
        #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
        #generate_square=table.iloc[:,1:5]
        if filtre_choice=="No_filtre":
            generate_square=table_non_filtre
            #generate_square=table_quantile
        elif filtre_choice=="quantile_filtre":
            generate_square=table_quantile
        elif filtre_choice=="RL_filtre":
            generate_square=table_filtre_RL
        


        #Ici on propose le tableau coord imagettes et classe predict
        





        xmin_gen=generate_square["xmin"]
        xmax_gen=generate_square["xmax"]
        ymin_gen=generate_square["ymin"]
        ymax_gen=generate_square["ymax"]
        ln_square_gen=len(generate_square)
    
       
        
        
        
        
        #Les diffs correspondent bien aux coordonnées de l'oiseaux.
        liste_DIFF_birds_defined=[]
        liste_DIFF_birds_undefined=[]
        liste_DIFF_other_animals=[]
        liste_DIFF_faisan=[]
        liste_DIFF_corbeau=[]
        liste_DIFF_pigeon=[]
        #On classe les difference repéré pour les animaux classés
        for classe in nom_classe:
            
            #Ou sinon on peut exclure ici les oiseaux non defs... .
            imagettes_annote_1_classe=imagettes_target[imagettes_target["classe"]==classe]
            nb_imagettes_1_classe=len(imagettes_annote_1_classe)
            
            #get max intersection with square generate for each sqaure annotate
            for i in range(nb_imagettes_1_classe):
                x_min_anote=imagettes_annote_1_classe["xmin"].iloc[i]
                x_max_anote=imagettes_annote_1_classe["xmax"].iloc[i]
                y_min_anote=imagettes_annote_1_classe["ymin"].iloc[i]
                y_max_anote=imagettes_annote_1_classe["ymax"].iloc[i]
         
    
                #Replicated the coordinates of annotations the number time of the len  to be able to apply area_intersection function
                zip_xmin_anote=[x_min_anote]*ln_square_gen
                zip_xmax_anote=[x_max_anote]*ln_square_gen
                zip_ymin_anote=[y_min_anote]*ln_square_gen
                zip_ymax_anote=[y_max_anote]*ln_square_gen
            
                
                #apply function area_intersection to calculate the area intersected between all generate square for this annote_squared 
                #and get the square with the max intersectio if there is enough area in commun
                
                #Je vais encore séparer les listes ici pour proposer oiseaux définis et autres animaux... . 
                liste_intersection=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip 
                                    (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
                max_intersection=max(liste_intersection)
                proportion_maximum=max_intersection/area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
                if proportion_maximum>coverage_threshold and (  (classe=="faisan")   or (classe=="corneille") or (classe=="pigeon")):
                    #Il faudrait rajouter une boucle while tan que max_intersection>limit_area l'enlever de la liste
                #liste_DIFF2.append(max_intersection.index(proportion_maximum))
                    liste_DIFF_birds_defined.append(liste_intersection.index(max_intersection))
                if proportion_maximum>coverage_threshold and (  (classe=="oiseau")   or (classe=="pie") or (classe=="incertain") ):
                    liste_DIFF_birds_undefined.append(liste_intersection.index(max_intersection))            
                if proportion_maximum>coverage_threshold and (  (classe=="lapin")   or (classe=="chevreuil") ):
                    liste_DIFF_other_animals.append(liste_intersection.index(max_intersection))     
                    
                if proportion_maximum>coverage_threshold and (  (classe=="faisan")   ):
                    liste_DIFF_faisan.append(liste_intersection.index(max_intersection))        
                if proportion_maximum>coverage_threshold and (  (classe=="corneille")    ):
                    liste_DIFF_corbeau.append(liste_intersection.index(max_intersection))                            
                if proportion_maximum>coverage_threshold and (  (classe=="pigeon")   ):
                    liste_DIFF_pigeon.append(liste_intersection.index(max_intersection))        
       
    
            #Capture du nombre d'imagettes à caputerer et captutées
        
        birds_table=imagettes_target[ (imagettes_target["classe"]=="corneille"  ) | (imagettes_target["classe"]=="pigeon"  ) |(imagettes_target["classe"]=="faisan"  ) 
        | (imagettes_target["classe"]=="oiseau"  )| (imagettes_target["classe"]=="pie"  ) |      (imagettes_target["classe"]=="incertain"  )      ]
        nb_animals_match=len(liste_DIFF_birds_defined)+len(liste_DIFF_birds_undefined)+len(liste_DIFF_other_animals)
        nb_animals_to_find=len(imagettes_target)
        birds_to_find=len(birds_table)
        birds_defined_match=len(liste_DIFF_birds_defined)
        birds_match=birds_defined_match+len(liste_DIFF_birds_undefined)
        print( "nombre d'animaux à reperer", nb_animals_to_find)
        print( "nombre d'animaux repérés", nb_animals_match)
        print("nombre d'oiseaux dans l'image",birds_to_find)
        print("nombre d'oiseaux totaux repérés dans l'image",birds_match)
        print("nombre d'oiseaux repérés parmi les oiseaux labélisés",birds_defined_match )



    #batchImages_stack = np.vstack(batchImages)
    #batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
    
    estimates = CNNmodel.predict(batchImages_stack_reshape)
    liste_prediction=list(estimates.argmax(axis=1))

    index_others=[]
    index_birds=[]
    index_other_animals=[]
    
    
    index_chevreuil=[]
    index_corbeau=[]
    index_lapin=[]
    index_faisan=[]
    index_pigeon=[]
    
    for i, j in enumerate(liste_prediction):
        if j == 0:
            index_others.append(i)
        if (j == 2) or (j == 3) or(j == 5) :
            index_birds.append(i)            
        if (j == 1) or (j == 4):
            index_other_animals.append(i) 
        if j==1:
            index_chevreuil.append(i)
        if j==2:
            index_corbeau.append(i)            
        if j==3:
            index_lapin.append(i)
        if j==4:
            index_faisan.append(i)
        if j==5:
            index_pigeon.append(i)

            
    #liste_FP=set(range(len(liste_prediction_not_match)))-  set(liste_VN)
        






            
            #On change de méthode
            
    index_animals=index_birds+index_other_animals
    liste_batch_images=range(len(batchImages_stack_reshape))
        
    liste_Diff_birds=liste_DIFF_birds_defined+liste_DIFF_birds_undefined
    liste_Diff_not_birds=set(liste_batch_images)-set(liste_Diff_birds)
    liste_Diff_animals=liste_Diff_birds+liste_DIFF_other_animals
    liste_DIFF_not_matche=set(liste_batch_images)-set(liste_Diff_birds)-set(liste_DIFF_other_animals)
    
            
    #D'une part on propose l'index de la liste qui renvoie la catégorie souhaité, d'autre part diff_X coorespond au numéro de la cat souhaité
    Birds_predicts=set(liste_Diff_birds).intersection(index_birds)
    FP_birds=set(liste_DIFF_not_matche).intersection(index_birds)
    FP=set(liste_DIFF_not_matche).intersection(index_animals)
    TP_birds=list(set(liste_DIFF_corbeau).intersection(index_corbeau))+list(set(liste_DIFF_faisan).intersection(index_faisan))+list(set(liste_DIFF_pigeon).intersection(index_pigeon))
    predicts_animals=set(liste_Diff_animals).intersection(index_animals)
  
            
        
        
    estimates_others = [max(estimates[i]) for i in (index_others)]
    estimates_birds = [max(estimates[i]) for i in (index_birds)]
    estimates_other_animals = [max(estimates[i]) for i in (index_other_animals)]
    
    estimates_FP_birds=[estimates[i] for i in (FP_birds)]
    estimates_TP_birds=[estimates[i] for i in (TP_birds)]
            

                
    #On va maintenant faire un tableau avec les ccordonées des carrés et leur prédictions
        
    #table_coordinate_predict["class_predict"]=liste_prediction_not_match
        
    imageRectangle=imageB.copy()
    
    table=generate_square.loc[FP_birds]
    #table=table_coordinate_predict
    for i in range(len(table)):   
        xmin=int(table["xmin"].iloc[i])
        xmax=int(table["xmax"].iloc[i])
        ymin=int(table["ymin"].iloc[i])
        ymax=int(table["ymax"].iloc[i])
        cv2.rectangle(imageRectangle, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2) 
    
        
        #table_coordinate_predict["class_predict"]=liste_prediction_not_match
        
    print("TP")
        
    #table=table_coordinate_predict
    table=generate_square.loc[TP_birds]
    for i in range(len(table)):   
        xmin=int(table["xmin"].iloc[i])
        xmax=int(table["xmax"].iloc[i])
        ymin=int(table["ymin"].iloc[i])
        ymax=int(table["ymax"].iloc[i])
        cv2.rectangle(imageRectangle, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2) 

            
    print("anotation")
        
        
    table=imagettes_target
    for i in range(len(table)):   
        xmin=int(table["xmin"].iloc[i])
        xmax=int(table["xmax"].iloc[i])
        ymin=int(table["ymin"].iloc[i])
        ymax=int(table["ymax"].iloc[i])
        cv2.rectangle(imageRectangle, (xmin,ymin), (xmax,ymax), (0, 255, 0), 2) 
        
        
    table=generate_square
    for i in range(len(table)):   
        xmin=int(table["xmin"].iloc[i])
        xmax=int(table["xmax"].iloc[i])
        ymin=int(table["ymin"].iloc[i])
        ymax=int(table["ymax"].iloc[i])
        cv2.rectangle(imageRectangle, (xmin,ymin), (xmax,ymax), (255, 255, 255), 2) 
        
    cv2.imwrite("/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/test_filtre_quantile.jpg",imageRectangle)        
 
    return estimates_FP_birds,estimates_TP_birds


#

        









#This function extract fp and animals detect and retrunds their imagettes and a table with their lables


        
        
  


def dictionnaire_conversion():
    dic_labels_to_num={}
    dic_num_to_labels={}

    dic_labels_to_num["autre"]=0
    dic_labels_to_num["chevreuil"]=1
    dic_labels_to_num["corneille"]=2
    dic_labels_to_num["faisan"]=3
    dic_labels_to_num["lapin"]=4
    dic_labels_to_num["pigeon"]=5
    dic_labels_to_num["oiseau"]=6
    
    dic_num_to_labels[0]="autre"
    dic_num_to_labels[1]="chevreuil"
    dic_num_to_labels[2]="corneille"
    dic_num_to_labels[3]="faisan"
    dic_num_to_labels[4]="lapin"
    dic_num_to_labels[5]="pigeon"
    dic_num_to_labels[6]="oiseau"
    
    return dic_labels_to_num,dic_num_to_labels


def extract_precise_fp(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=53,blurFact=15,
            filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
    #Initialisation de variables et de liste
    batchImages = []
    liste_table = []
    imageSize= 28
    animals_predict=0
    birds_predict=0
    VTP=0
    nb_animals_match=0

    
    #Dictionnary to convert string labels to num labels
    dic_labels_to_num,dic_num_to_labels=dictionnaire_conversion()
    
    #
    dict_anotation_index_to_classe={}
    
    #Definition des images
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    
    
    #Ouverture des fichiers annotés 
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    imagettes=to_reference_labels (imagettes,"classe")
    imagettes=imagettes[  (imagettes["classe"]!="autre") & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") 
    & (imagettes["classe"]!="sanglier") & (imagettes["classe"]!="cheval") & (imagettes["classe"]!="ground") & (imagettes["classe"]!="autre") ]
    
    
    folder_choosen="."+ folder
    imagettes_folder=imagettes[(imagettes["path"]==folder_choosen) ]


    #On selectionne seulement pour la photo sur laquel on veut repérer les oiseaux ou autres animaux et on réarange les colonnes dans le bon ordre
    imagettes_target=imagettes_folder[imagettes_folder["filename"]==name_test]
    to_drop=['path', 'filename', 'width', 'height', 'index']
    imagettes_target=imagettes_target.drop(to_drop,axis=1)
    col = list(imagettes_target.columns)[-1:] + list(imagettes_target.columns)[:-1]
    imagettes_target=imagettes_target[col]
    
    
    
    #On regarde si il y a des imagettes de type différents pas forcément utiles surtout si on garde les oiseaux undefined
    if len(imagettes_target["classe"].unique())>1:
        print("attention le code le prend en charge" )
    #nom_classe=imagettes1["classe"].iloc[0]
    nom_classe=list(imagettes_target["classe"].unique())
    

    
    
    
    
    #On récupère les coordonnées des pixels différent par différence 
    cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))

    #try:

      
    batchImages_stack = np.vstack(batchImages)
    batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
    table_non_filtre = pd.DataFrame(np.vstack(liste_table))
    table_non_filtre = table_non_filtre.rename(columns={ 0: 'xmin', 1: 'xmax', 2: 'ymin', 3: 'ymax'})
    table_non_filtre=table_non_filtre.astype(int)
        
 
        



    generate_square=table_non_filtre


    #Les diffs correspondent bien aux coordonnées de l'oiseaux.
    liste_DIFF_birds_defined=[]
    liste_DIFF_birds_undefined=[]
    liste_DIFF_other_animals=[]
    liste_DIFF_faisan=[]
    liste_DIFF_corbeau=[]
    liste_DIFF_pigeon=[]


    xmin_gen=generate_square["xmin"]
    xmax_gen=generate_square["xmax"]
    ymin_gen=generate_square["ymin"]
    ymax_gen=generate_square["ymax"]
    ln_square_gen=len(generate_square)
    


    #On classe les difference repéré pour les animaux classés
    for classe in nom_classe:
            
        
        liste_Diff_birds_this_class=[]
        liste_DIFF_birds_defined_this_class=[]
        liste_DIFF_birds_undefined_this_class=[]
        liste_DIFF_other_animals_this_class=[]
        

    
        #Initialisation du dic à utiliser plus tard pour récupérer pour chaque index la classe
        
        
        imagettes_annote_1_classe=imagettes_target[imagettes_target["classe"]==classe]
        nb_imagettes_1_classe=len(imagettes_annote_1_classe)
        

            
        #get max intersection with square generate for each sqaure annotate
        for i in range(nb_imagettes_1_classe):
            x_min_anote=imagettes_annote_1_classe["xmin"].iloc[i]
            x_max_anote=imagettes_annote_1_classe["xmax"].iloc[i]
            y_min_anote=imagettes_annote_1_classe["ymin"].iloc[i]
            y_max_anote=imagettes_annote_1_classe["ymax"].iloc[i]
         
    
            #Replicated the coordinates of annotations the number time of the len  to be able to apply area_intersection function
            zip_xmin_anote=[x_min_anote]*ln_square_gen
            zip_xmax_anote=[x_max_anote]*ln_square_gen
            zip_ymin_anote=[y_min_anote]*ln_square_gen
            zip_ymax_anote=[y_max_anote]*ln_square_gen
            
                
            #apply function area_intersection to calculate the area intersected between all generate square for this annote_squared 
            #and get the square with the max intersectio if there is enough area in commun
                
            #Je vais encore séparer les listes ici pour proposer oiseaux définis et autres animaux... . 
            liste_intersection=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip 
                                (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
            max_intersection=max(liste_intersection)
            proportion_maximum=max_intersection/area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
            if proportion_maximum>coverage_threshold and (  (classe=="faisan")   or (classe=="corneille") or (classe=="pigeon")):
                #Il faudrait rajouter une boucle while tan que max_intersection>limit_area l'enlever de la liste
                #liste_DIFF2.append(max_intersection.index(proportion_maximum))
                liste_DIFF_birds_defined.append(liste_intersection.index(max_intersection))
                liste_DIFF_birds_defined_this_class.append(liste_intersection.index(max_intersection))
            if proportion_maximum>coverage_threshold and (  (classe=="oiseau")   or (classe=="pie") or (classe=="incertain") ):
                liste_DIFF_birds_undefined.append(liste_intersection.index(max_intersection))     
                liste_DIFF_birds_undefined_this_class.append(liste_intersection.index(max_intersection))     
            if proportion_maximum>coverage_threshold and (  (classe=="lapin")   or (classe=="chevreuil") ):
                liste_DIFF_other_animals.append(liste_intersection.index(max_intersection))     
                liste_DIFF_other_animals_this_class.append(liste_intersection.index(max_intersection))    
                
            if proportion_maximum>coverage_threshold and (  (classe=="faisan")   ):
                liste_DIFF_faisan.append(liste_intersection.index(max_intersection))        
            if proportion_maximum>coverage_threshold and (  (classe=="corneille")    ):
                liste_DIFF_corbeau.append(liste_intersection.index(max_intersection))                            
            if proportion_maximum>coverage_threshold and (  (classe=="pigeon")   ):
                liste_DIFF_pigeon.append(liste_intersection.index(max_intersection))        
       
        
            liste_Diff_birds=liste_DIFF_birds_defined+liste_DIFF_birds_undefined
            liste_Diff_animals=liste_Diff_birds+liste_DIFF_other_animals
    
            #Need that to fill the columns "reel cat"
            liste_Diff_birds_this_class=liste_DIFF_birds_defined_this_class+liste_DIFF_birds_undefined_this_class
            liste_Diff_animals_this_class=liste_Diff_birds_this_class+liste_DIFF_other_animals_this_class
            
            index_anote=list(liste_Diff_animals_this_class)
            for i in index_anote:
                dict_anotation_index_to_classe[str(i)]=dic_labels_to_num[classe]
    
        #Capture du nombre d'imagettes à caputerer et captutées   
    birds_table=imagettes_target[ (imagettes_target["classe"]=="corneille"  ) | (imagettes_target["classe"]=="pigeon"  ) |(imagettes_target["classe"]=="faisan"  ) 
    | (imagettes_target["classe"]=="oiseau"  )| (imagettes_target["classe"]=="pie"  ) |      (imagettes_target["classe"]=="incertain"  )      ]
    nb_animals_match=len(liste_DIFF_birds_defined)+len(liste_DIFF_birds_undefined)+len(liste_DIFF_other_animals)
    nb_animals_to_find=len(imagettes_target)
    birds_to_find=len(birds_table)
    birds_defined_match=len(liste_DIFF_birds_defined)
    birds_match=birds_defined_match+len(liste_DIFF_birds_undefined)
    print( "nombre d'animaux à reperer", nb_animals_to_find)
    print( "nombre d'animaux repérés", nb_animals_match)
    print("nombre d'oiseaux dans l'image",birds_to_find)
    print("nombre d'oiseaux totaux repérés dans l'image",birds_match)
    print("nombre d'oiseaux repérés parmi les oiseaux labélisés",birds_defined_match )


    estimates = CNNmodel.predict(batchImages_stack_reshape)
    liste_prediction=list(estimates.argmax(axis=1))

    index_others=[]
    index_birds=[]
    index_other_animals=[]
    
    
    index_chevreuil=[]
    index_corbeau=[]
    index_lapin=[]
    index_faisan=[]
    index_pigeon=[]
    
    for i, j in enumerate(liste_prediction):
        if j == 0:
            index_others.append(i)
        if (j == 2) or (j == 3) or(j == 5) :
            index_birds.append(i)            
        if (j == 1) or (j == 4):
            index_other_animals.append(i) 
        if j==1:
            index_chevreuil.append(i)
        if j==2:
            index_corbeau.append(i)            
        if j==3:
            index_lapin.append(i)
        if j==4:
            index_faisan.append(i)
        if j==5:
            index_pigeon.append(i)


            
    index_animals=index_birds+index_other_animals
    liste_batch_images=range(len(batchImages_stack_reshape))
        
    liste_Diff_birds=liste_DIFF_birds_defined+liste_DIFF_birds_undefined
    liste_Diff_not_birds=set(liste_batch_images)-set(liste_Diff_birds)
    liste_Diff_animals=liste_Diff_birds+liste_DIFF_other_animals
    liste_DIFF_not_matche=set(liste_batch_images)-set(liste_Diff_birds)-set(liste_DIFF_other_animals)
    
            
    #D'une part on propose l'index de la liste qui renvoie la catégorie souhaité, d'autre part diff_X coorespond au numéro de la cat souhaité
    Birds_predicts=set(liste_Diff_birds).intersection(index_birds)
    FP_birds=set(liste_DIFF_not_matche).intersection(index_birds)
    FP=set(liste_DIFF_not_matche).intersection(index_animals)
    TP_birds=list(set(liste_DIFF_corbeau).intersection(index_corbeau))+list(set(liste_DIFF_faisan).intersection(index_faisan))+list(set(liste_DIFF_pigeon).intersection(index_pigeon))
    predicts_animals=set(liste_Diff_animals).intersection(index_animals)
  
            
        
        
    estimates_others = [max(estimates[i]) for i in (index_others)]
    estimates_birds = [max(estimates[i]) for i in (index_birds)]
    estimates_other_animals = [max(estimates[i]) for i in (index_other_animals)]
    estimates_FP_birds=[max(estimates[i]) for i in (FP_birds)]
    estimates_TP_birds=[estimates[i] for i in (TP_birds)]
    estimates_FP=[max(estimates[i]) for i in (FP)]
    
    liste_prediction_FP=[liste_prediction[i] for i in FP]
    liste_prediction_animals_match=[liste_prediction[i] for i in liste_Diff_animals]
    estimates_animals_match=[max(estimates[i]) for i in liste_Diff_animals]
    #On va maintenant faire un tableau avec les ccordonées des carrés et leur prédictions
        
    #Regarde le cas où il y a du FP dans diff_animals
    
    #Va falloir créer liste_pred_animals(peut être simple index et estimation)

    table=generate_square.loc[list(FP)+liste_Diff_animals]
    table["predict_cat"]=liste_prediction_FP+liste_prediction_animals_match
    table["max_cat"]=estimates_FP+estimates_animals_match
    table["filename"]=folder+"/"+name_test
    table["former_index"]=list(table.index)
    
    
    
    
    
    
    #table_total=table
    liste_imagette_names=[]

    for i in range(len(table)):
        name=(table["filename"].iloc[i].split("/")[-1]).split(".")[0]+"_"+str(table.index[i])+"_"+str(round(table["max_cat"].iloc[i],2))[2:]+"_"+str(table["predict_cat"].iloc[i])+".jpg"
        liste_imagette_names.append(name)
    
    table["imagetteName"]=liste_imagette_names





    path_liste=[]

    for i in range(len(table)):
        string=table["filename"].iloc[i]
        path="."+"/".join(string.split("/")[:3])
        path_liste.append(path)
    
    table["path"]= path_liste



    filename_liste=[]
    for i in range(len(table)):
        string=table["filename"].iloc[i]
        filename=string.split("/")[-1]
        filename_liste.append(filename)

    table["filename"]=filename_liste


    col=['path','filename', 'imagetteName','max_cat','predict_cat','xmin', 'xmax', 'ymin', 'ymax',   
         'former_index']

    #col_to_drop= list(  set(list(table_entiere.columns)) -set(col)   )

    #A partir ligne par ligne
    table=table[col]
    table["IsFP"]=0
    for i in FP:
        table["IsFP"].loc[i]=1
        
        
    table["reel_classe"]=0
    for i in liste_Diff_animals:
        table["reel_classe"].loc[i]=dict_anotation_index_to_classe[str(i)]
    #Ici on devrait rajouter la colonne pour différencier les FP et les matchs animals
    
    #On va créer une fonction pour extaire les imagettes avec itération selon cnts pour fp et avec un i 
    batchImages_FP=[batchImages[i] for i in FP ]     
    batchImages_match_animals=[batchImages[i] for i in liste_Diff_animals ]  
    batch_c=batchImages_FP+batchImages_match_animals
    #batchImages_FP=[batchImages[i] for i in FP +liste_Diff_animals ]   
    #chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images")
    
    chdir("/mnt/VegaSlowDataDisk/c3po/Annotation_automatique/tf_200ep/")
    
    
    for i in range( len(table) ):
        name=(table["filename"].iloc[i].split("/")[-1]).split(".")[0]+"_"+str(table.index[i])+"_"+str(round(table["max_cat"].iloc[i],2))[2:]+"_"+str(table["predict_cat"].iloc[i])+".jpg"
        image=batch_c[i]

        cv2.imwrite(name,image)
        
    #chdir("/mnt/VegaSlowDataDisk/c3po/Annotation_automatique/")
    
    #table.to_csv("imagettes_fp_animals_birds",index=False)
      
   
    return table,subI,liste_Diff_animals






def open_imagettes_file(imagettes,folder,name_test):
    
    
    
    imagettes=to_reference_labels (imagettes,"classe")
    imagettes=imagettes[  (imagettes["classe"]!="autre") & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") 
    & (imagettes["classe"]!="sanglier") & (imagettes["classe"]!="cheval") & (imagettes["classe"]!="ground") & (imagettes["classe"]!="autre") ]
    
    
    folder_choosen="."+ folder
    imagettes_folder=imagettes[(imagettes["path"]==folder_choosen) ]


    #On selectionne seulement pour la photo sur laquel on veut repérer les oiseaux ou autres animaux et on réarange les colonnes dans le bon ordre
    imagettes_target=imagettes_folder[imagettes_folder["filename"]==name_test]
    to_drop=['path', 'filename', 'width', 'height', 'index']
    imagettes_target=imagettes_target.drop(to_drop,axis=1)
    col = list(imagettes_target.columns)[-1:] + list(imagettes_target.columns)[:-1]
    imagettes_target=imagettes_target[col]
    
    
    
    #On regarde si il y a des imagettes de type différents pas forcément utiles surtout si on garde les oiseaux undefined
    if len(imagettes_target["classe"].unique())>1:
        print("attention le code le prend en charge" )
    #nom_classe=imagettes1["classe"].iloc[0]
    nom_classe=list(imagettes_target["classe"].unique())
    
    return nom_classe,imagettes_target









def im_diff_to_org_list(generate_square,coverage_threshold,
                        nom_classe,imagettes_target,dic_labels_to_num):

    
    
    dict_anotation_index_to_classe={}
    liste_DIFF_birds_defined=[]
    liste_DIFF_birds_undefined=[]
    liste_DIFF_other_animals=[]
    liste_DIFF_faisan=[]
    liste_DIFF_corbeau=[]
    liste_DIFF_pigeon=[]


    xmin_gen=generate_square["xmin"]
    xmax_gen=generate_square["xmax"]
    ymin_gen=generate_square["ymin"]
    ymax_gen=generate_square["ymax"]
    ln_square_gen=len(generate_square)
    
    


    #On classe les difference repéré pour les animaux classés
    for classe in nom_classe:
            
        
        liste_Diff_birds_this_class=[]
        liste_DIFF_birds_defined_this_class=[]
        liste_DIFF_birds_undefined_this_class=[]
        liste_DIFF_other_animals_this_class=[]
        

    
        #Initialisation du dic à utiliser plus tard pour récupérer pour chaque index la classe
        
        
        imagettes_annote_1_classe=imagettes_target[imagettes_target["classe"]==classe]
        nb_imagettes_1_classe=len(imagettes_annote_1_classe)
        

            
        #get max intersection with square generate for each sqaure annotate
        for i in range(nb_imagettes_1_classe):
            x_min_anote=imagettes_annote_1_classe["xmin"].iloc[i]
            x_max_anote=imagettes_annote_1_classe["xmax"].iloc[i]
            y_min_anote=imagettes_annote_1_classe["ymin"].iloc[i]
            y_max_anote=imagettes_annote_1_classe["ymax"].iloc[i]
         
    
            #Replicated the coordinates of annotations the number time of the len  to be able to apply area_intersection function
            zip_xmin_anote=[x_min_anote]*ln_square_gen
            zip_xmax_anote=[x_max_anote]*ln_square_gen
            zip_ymin_anote=[y_min_anote]*ln_square_gen
            zip_ymax_anote=[y_max_anote]*ln_square_gen
            
                
            #apply function area_intersection to calculate the area intersected between all generate square for this annote_squared 
            #and get the square with the max intersectio if there is enough area in commun
                
            #Je vais encore séparer les listes ici pour proposer oiseaux définis et autres animaux... . 
            liste_intersection=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip 
                                (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
            max_intersection=max(liste_intersection)
            proportion_maximum=max_intersection/area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
            if proportion_maximum>coverage_threshold and (  (classe=="faisan")   or (classe=="corneille") or (classe=="pigeon")):
                #Il faudrait rajouter une boucle while tan que max_intersection>limit_area l'enlever de la liste
                #liste_DIFF2.append(max_intersection.index(proportion_maximum))
                liste_DIFF_birds_defined.append(liste_intersection.index(max_intersection))
                liste_DIFF_birds_defined_this_class.append(liste_intersection.index(max_intersection))
            if proportion_maximum>coverage_threshold and (  (classe=="oiseau")   or (classe=="pie") or (classe=="incertain") ):
                liste_DIFF_birds_undefined.append(liste_intersection.index(max_intersection))     
                liste_DIFF_birds_undefined_this_class.append(liste_intersection.index(max_intersection))     
            if proportion_maximum>coverage_threshold and (  (classe=="lapin")   or (classe=="chevreuil") ):
                liste_DIFF_other_animals.append(liste_intersection.index(max_intersection))     
                liste_DIFF_other_animals_this_class.append(liste_intersection.index(max_intersection))    
                
            if proportion_maximum>coverage_threshold and (  (classe=="faisan")   ):
                liste_DIFF_faisan.append(liste_intersection.index(max_intersection))        
            if proportion_maximum>coverage_threshold and (  (classe=="corneille")    ):
                liste_DIFF_corbeau.append(liste_intersection.index(max_intersection))                            
            if proportion_maximum>coverage_threshold and (  (classe=="pigeon")   ):
                liste_DIFF_pigeon.append(liste_intersection.index(max_intersection))        
       
        
            liste_Diff_birds=liste_DIFF_birds_defined+liste_DIFF_birds_undefined
            liste_Diff_animals=liste_Diff_birds+liste_DIFF_other_animals
    
            #Need that to fill the columns "reel cat"
            liste_Diff_birds_this_class=liste_DIFF_birds_defined_this_class+liste_DIFF_birds_undefined_this_class
            liste_Diff_animals_this_class=liste_Diff_birds_this_class+liste_DIFF_other_animals_this_class
            
            index_anote=list(liste_Diff_animals_this_class)
            for i in index_anote:
                dict_anotation_index_to_classe[str(i)]=dic_labels_to_num[classe]
                
                
                
                
                
    
        #Capture du nombre d'imagettes à caputerer et captutées   
    birds_table=imagettes_target[ (imagettes_target["classe"]=="corneille"  ) | (imagettes_target["classe"]=="pigeon"  ) |(imagettes_target["classe"]=="faisan"  ) 
    | (imagettes_target["classe"]=="oiseau"  )| (imagettes_target["classe"]=="pie"  ) |      (imagettes_target["classe"]=="incertain"  )      ]
    nb_animals_match=len(liste_DIFF_birds_defined)+len(liste_DIFF_birds_undefined)+len(liste_DIFF_other_animals)
    nb_animals_to_find=len(imagettes_target)
    birds_to_find=len(birds_table)
    birds_defined_match=len(liste_DIFF_birds_defined)
    
    #Je prends pas
    #birds_match=birds_defined_match+len(liste_DIFF_birds_undefined)
    
    
    
    
    print( "nombre d'animaux à reperer", nb_animals_to_find)
    print( "nombre d'animaux repérés", nb_animals_match)
    print("nombre d'oiseaux dans l'image",birds_to_find)
    #print("nombre d'oiseaux totaux repérés dans l'image",birds_match)
    print("nombre d'oiseaux repérés parmi les oiseaux labélisés",birds_defined_match )
    
    return (liste_Diff_animals,dict_anotation_index_to_classe,liste_DIFF_birds_defined,liste_DIFF_birds_undefined,birds_defined_match,
liste_Diff_birds_this_class,liste_DIFF_birds_defined_this_class,liste_DIFF_birds_undefined_this_class,liste_DIFF_other_animals_this_class,
liste_DIFF_corbeau,liste_DIFF_faisan,liste_DIFF_pigeon,liste_DIFF_other_animals)
    



def class_predictions(liste_prediction):
    
    
    index_others=[]
    index_birds=[]
    index_other_animals=[]
    
    
    index_chevreuil=[]
    index_corbeau=[]
    index_lapin=[]
    index_faisan=[]
    index_pigeon=[]
    
    for i, j in enumerate(liste_prediction):
        if j == 0:
            index_others.append(i)
        if (j == 2) or (j == 3) or(j == 5) :
            index_birds.append(i)            
        if (j == 1) or (j == 4):
            index_other_animals.append(i) 
        if j==1:
            index_chevreuil.append(i)
        if j==2:
            index_corbeau.append(i)            
        if j==3:
            index_lapin.append(i)
        if j==4:
            index_faisan.append(i)
        if j==5:
            index_pigeon.append(i)

    return index_others,index_birds,index_other_animals,index_chevreuil,index_corbeau,index_lapin,index_faisan,index_pigeon









def table_gen_im(table,FP,liste_Diff_animals,dict_anotation_index_to_classe):
    
    liste_imagette_names=[]

    for i in range(len(table)):
        name=(table["filename"].iloc[i].split("/")[-1]).split(".")[0]+"_"+str(table.index[i])+"_"+str(round(table["max_cat"].iloc[i],2))[2:]+"_"+str(table["predict_cat"].iloc[i])+".jpg"
        liste_imagette_names.append(name)
    
    table["imagetteName"]=liste_imagette_names





    path_liste=[]

    for i in range(len(table)):
        string=table["filename"].iloc[i]
        path="."+"/".join(string.split("/")[:3])
        path_liste.append(path)
    
    table["path"]= path_liste



    filename_liste=[]
    for i in range(len(table)):
        string=table["filename"].iloc[i]
        filename=string.split("/")[-1]
        filename_liste.append(filename)

    table["filename"]=filename_liste


    col=['path','filename', 'imagetteName','max_cat','predict_cat','xmin', 'xmax', 'ymin', 'ymax',   
         'former_index']

    #col_to_drop= list(  set(list(table_entiere.columns)) -set(col)   )

    #A partir ligne par ligne
    table=table[col]
    table["IsFP"]=0
    for i in FP:
        table["IsFP"].loc[i]=1
        
        
    table["reel_classe"]=0
    for i in liste_Diff_animals:
        table["reel_classe"].loc[i]=dict_anotation_index_to_classe[str(i)]

    return table


def extract_fp_court(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=53,blurFact=15,
            filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
    #Initialisation de variables et de liste
    batchImages = []
    liste_table = []
    imageSize= 28


    
    #Dictionnary to convert string labels to num labels
    dic_labels_to_num,dic_num_to_labels=dictionnaire_conversion()
    #dict_anotation_index_to_classe={}
    
    #Definition des images
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    

    #Ouverture des fichiers annotés 
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    nom_classe,imagettes_target=open_imagettes_file(imagettes,folder,name_test)
    



    
    
    #On effectue une différentiation des images et on récupères le filtre des images sous forme de batch
    cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))      
    batchImages_stack = np.vstack(batchImages)
    batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
    table_non_filtre = pd.DataFrame(np.vstack(liste_table))
    table_non_filtre = table_non_filtre.rename(columns={ 0: 'xmin', 1: 'xmax', 2: 'ymin', 3: 'ymax'})
    table_non_filtre=table_non_filtre.astype(int)
        
    generate_square=table_non_filtre

    
    
    #On classe les imagettes générées en fonction de l'espèce avec laquelle elle correspond sur l'image
    (liste_Diff_animals,dict_anotation_index_to_classe,liste_DIFF_birds_defined,liste_DIFF_birds_undefined,birds_defined_match,liste_Diff_birds_this_class,
     liste_DIFF_birds_defined_this_class,liste_DIFF_birds_undefined_this_class,liste_DIFF_other_animals_this_class,liste_DIFF_corbeau,liste_DIFF_faisan,
     liste_DIFF_pigeon,liste_DIFF_other_animals)=im_diff_to_org_list(generate_square,coverage_threshold, nom_classe,imagettes_target,dic_labels_to_num)  
    liste_Diff_birds=liste_DIFF_birds_defined+liste_DIFF_birds_undefined
    liste_Diff_animals=liste_Diff_birds+liste_DIFF_other_animals
    birds_match=birds_defined_match+len(liste_DIFF_birds_undefined)
    
    
    #birds_match=birds_defined_match+len(liste_DIFF_birds_undefined)
    #print("nombre d'oiseaux totaux repérés dans l'image",birds_match)
    
    
    estimates = CNNmodel.predict(batchImages_stack_reshape)
    liste_prediction=list(estimates.argmax(axis=1))


    
    
    
    index_others,index_birds,index_other_animals,index_chevreuil,index_corbeau,index_lapin,index_faisan,index_pigeon=class_predictions(liste_prediction)
            
    index_animals=index_birds+index_other_animals
    liste_batch_images=range(len(batchImages_stack_reshape))
        
    liste_Diff_birds=liste_DIFF_birds_defined+liste_DIFF_birds_undefined
    liste_Diff_not_birds=set(liste_batch_images)-set(liste_Diff_birds)
    liste_Diff_animals=liste_Diff_birds+liste_DIFF_other_animals
    liste_DIFF_not_matche=set(liste_batch_images)-set(liste_Diff_birds)-set(liste_DIFF_other_animals)
    
            
    #D'une part on propose l'index de la liste qui renvoie la catégorie souhaité, d'autre part diff_X coorespond au numéro de la cat souhaité
    Birds_predicts=set(liste_Diff_birds).intersection(index_birds)
    FP_birds=set(liste_DIFF_not_matche).intersection(index_birds)
    FP=set(liste_DIFF_not_matche).intersection(index_animals)
    TP_birds=list(set(liste_DIFF_corbeau).intersection(index_corbeau))+list(set(liste_DIFF_faisan).intersection(index_faisan))+list(set(liste_DIFF_pigeon).intersection(index_pigeon))
    predicts_animals=set(liste_Diff_animals).intersection(index_animals)
  
            
        
        
    estimates_others = [max(estimates[i]) for i in (index_others)]
    estimates_birds = [max(estimates[i]) for i in (index_birds)]
    estimates_other_animals = [max(estimates[i]) for i in (index_other_animals)]
    estimates_FP_birds=[max(estimates[i]) for i in (FP_birds)]
    estimates_TP_birds=[estimates[i] for i in (TP_birds)]
    estimates_FP=[max(estimates[i]) for i in (FP)]
    
    liste_prediction_FP=[liste_prediction[i] for i in FP]
    liste_prediction_animals_match=[liste_prediction[i] for i in liste_Diff_animals]
    estimates_animals_match=[max(estimates[i]) for i in liste_Diff_animals]
    #On va maintenant faire un tableau avec les ccordonées des carrés et leur prédictions
        
    #Regarde le cas où il y a du FP dans diff_animals
    
    #Va falloir créer liste_pred_animals(peut être simple index et estimation)

    table=generate_square.loc[list(FP)+liste_Diff_animals]
    table["predict_cat"]=liste_prediction_FP+liste_prediction_animals_match
    table["max_cat"]=estimates_FP+estimates_animals_match
    table["filename"]=folder+"/"+name_test
    table["former_index"]=list(table.index)
    
    
    
    
    
    
    table=table_gen_im(table,FP,liste_Diff_animals,dict_anotation_index_to_classe)
    #Ici on devrait rajouter la colonne pour différencier les FP et les matchs animals
    
    #On va créer une fonction pour extaire les imagettes avec itération selon cnts pour fp et avec un i 
    batchImages_FP=[batchImages[i] for i in FP ]     
    batchImages_match_animals=[batchImages[i] for i in liste_Diff_animals ]  
    batch_to_select=batchImages_FP+batchImages_match_animals
    #batchImages_FP=[batchImages[i] for i in FP +liste_Diff_animals ]   
    #chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images")
    
    """
    chdir("/mnt/VegaSlowDataDisk/c3po/Annotation_automatique/tf_200ep/")
    
    
    for i in range( len(table) ):
        name=(table["filename"].iloc[i].split("/")[-1]).split(".")[0]+"_"+str(table.index[i])+"_"+str(round(table["max_cat"].iloc[i],2))[2:]+"_"+str(table["predict_cat"].iloc[i])+".jpg"
        image=batch_to_select[i]

        cv2.imwrite(name,image)
        
    #chdir("/mnt/VegaSlowDataDisk/c3po/Annotation_automatique/")
    
    #table.to_csv("imagettes_fp_animals_birds",index=False)
    """  
   
    return table,subI,liste_Diff_animals







