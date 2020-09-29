#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import tensorflow
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

#permet d'eliminer les images qui ne sont pas oiseaux
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




    for i in range(len(table)):
        logHIS=table["logHIS"].iloc[i]
        logHIS_high=table["logHIS_pred_high"].iloc[i]
        logHIS_low=table["logHIS_pred_low"].iloc[i]
        #possiblebird
        table["PossibleBird"].iloc[i]=( logHIS< 0.8*logHIS_high) and (logHIS > 1.2*logHIS_low)
    
    
    
    
    
    

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










def filtre_light(imageA,imageB):
    img2 = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    img1 = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
    blurFact = 25
    absDiff2 = cv2.absdiff(img1, img2)
    diff = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
    th2 = cv2.adaptiveThreshold(src=diff,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
                                thresholdType=cv2.THRESH_BINARY,blockSize=221,C=-30) # adaptation de C à histogram de la photo ?

    th2Blur=cv2.GaussianBlur(th2,(blurFact,blurFact),sigmaX=0)
    th2BlurTh = cv2.adaptiveThreshold(src=th2Blur,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
            thresholdType=cv2.THRESH_BINARY,blockSize=121,C=-30) # adaptation de C à histogram de la photo ?
    threshS=th2BlurTh

        # defines corresponding regions of change
    cnts = cv2.findContours(threshS.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    return cnts


















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
            
    
    cv2.imwrite("testingInputs/oiseau_pasOis.jpg", imageRectangles)    
    

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








def birds_is_catched(neurone_features,imageA,imageB,filtre_choice,coef_filtre,path_anotation,name2,height=2448,width=3264):


    #On va comparer les images de ici et probablement test unitaires
    batchImages = []
    liste_table = []
    imageSize= 28
    cnts=filtre_light(imageA,imageB)
    #On récupère les coordonnées des pixels différent par différence
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        name = (os.path.split(name2)[-1]).split(".")[0]
        name = name + "_" + str(ic) + ".JPG"
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
    table_quantile,index_possible_birds=filtre_quantile(table_full,coef_filtre,height=2448,width=3264)
    table_filtre_RL=table_quantile.copy()
    table_filtre_RL["possible_bird"]=filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
    table_filtre_RL=(table_filtre_RL[table_filtre_RL["possible_bird"]=="O"])
    p_bird=table_filtre_RL.index
    table_filtre_RL.drop("possible_bird",axis=1,inplace=True)     
    index_possible_birds=list(set(index_possible_birds).intersection(p_bird)) 
    #batchImages_filtre = [batchImages_stack_reshape[i] for i in (index_possible_birds)]


    #Construisons les carrés pour les coordonnées issus de l'annotation 

    

    table_add=pd.read_csv(path_anotation)
    annontation_reduit=(table_add.iloc[:,6:12]).drop("index",axis=1)
    annontation_reduit=annontation_reduit.iloc[::2]
    
    
    liste_carre_annote=annontation_reduit[['xmin', 'ymin', 'xmax', 'ymax']].apply(to_polygon,axis=1)
    
    
    
    #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
    #generate_square=table.iloc[:,1:5]
    if filtre_choice=="No_filtre":
        generate_square=table_full.iloc[:,1:5]
    elif filtre_choice=="quantile_filtre":
        generate_square=table_quantile.iloc[:,1:5]
    elif filtre_choice=="RL_filtre":
        generate_square=table_filtre_RL.iloc[:,1:5]
    
    liste_carre_diff=generate_square[['xmin', 'ymin', 'xmax', 'ymax']].apply(to_polygon,axis=1)
    
    #Maintenant on va voir si les carrés des diffs ont suffisament de surfaces en commun avec les carrés des annotations

    
    #Initialisation
    liste_ANNOTATION=[]
    liste_DIFF=[]
    nb_carre_diff=0
    nombre_imagette_oiseau=0    
    
    proportion_limit=0.5
    
    for i in liste_carre_diff:
        max_proportion_test=0
        
        #A chaque fois qu'on change de carre_diff on va remettre le compteur à 0 pour les annotations
        nb_carre_annotation=0
        
        for polygon_ref in liste_carre_annote:
            intersection=polygon_ref.intersection(i)
            proportion=intersection.area/polygon_ref.area
            if proportion>proportion_limit:
                max_proportion_test=proportion
                liste_ANNOTATION.append(nb_carre_annotation)
            
            #On passe au carré un mais peut etre que le compteur devrait se faire plus bas. 
            nb_carre_annotation+=1
        if (max_proportion_test>proportion_limit) :  
            liste_DIFF.append(nb_carre_diff)
            nombre_imagette_oiseau+=1
        
        #On passe au carre_diff suivant 
        nb_carre_diff+=1
    
    
    print( "nombre d'oiseau repérés", nombre_imagette_oiseau)
    
    
    
    #On va initialiser xmin,ymin .... de sorte de n'avoir à changer que la valeur de i pour les diffs
        
        
    imageRectangles=imageB.copy()
    #liste_DIFF=[499,610]
    
    for i in liste_DIFF:
        xmin=generate_square["xmin"].iloc[i]
        ymin=generate_square["ymin"].iloc[i]
        xmax=generate_square["xmax"].iloc[i]
        ymax=generate_square["ymax"].iloc[i]
        
        cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 
    
    #On va initialiser xmin,ymin .... de sorte de n'avoir à changer que la valeur de i pour les annontations
    
    #liste_ANNOTATION=[4,8]
    for i in liste_ANNOTATION:
        xmin=annontation_reduit["xmin"].iloc[i]
        ymin=annontation_reduit["ymin"].iloc[i]
        xmax=annontation_reduit["xmax"].iloc[i]
        ymax=annontation_reduit["ymax"].iloc[i]
    
        cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 255,0), 2) 
    #On va essayer d'écrites maintenant les carrés qui sont censés repérés les oiseaux et vérifier qu'ils matchent bien avec des oiseaux ( pas de décalage d'indices par exemple)
    
    #cv2.imwrite("Output_images/annotation_is_match.jpg",imageRectangles)
    
    batchImages_test = [batchImages_stack_reshape[i] for i in (liste_DIFF)]
    batchImages=np.vstack(batchImages_test)
    
        #Prediction modèle
    model = load_model(neurone_features,compile=False)
    CNNmodel = Model(inputs=model.input, outputs=model.layers[-1].output)
    estimates = CNNmodel.predict(batchImages.reshape(-1,28,28,3))
    
    
    if nombre_imagette_oiseau>0:
        are_there_birds=True
    else:
        are_there_birds=False
    #On va essayer maintenant de faire des prédictions  on l'intègrera ensuite dans les carrés
    return are_there_birds