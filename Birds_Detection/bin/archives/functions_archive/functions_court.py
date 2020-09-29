#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:49:00 2020

@author: marcpozzo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:15:20 2020

@author: marcpozzo
"""
#Supprimer fonctions doublons
#C est a dire celles avec les memes sorties
#Et celles notese bis
#On peut faire une fonction pour aller plus vite dans le traçage du rectangle.

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
import itertools
#Pour faire fonctionner le code Khalid
from pandas.core.common import flatten
import joblib
from keras.applications import  imagenet_utils
from keras.applications import VGG16
from os import chdir
from numpy import load


model = VGG16(weights="imagenet", include_top=False)
#c3poFolder="/mnt/VegaSlowDataDisk/c3po_interface/"

c3poFolder="/mnt/VegaSlowDataDisk/c3po_interface_mark/"
Model1 = joblib.load("/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/output/model.cpickle")
filtre_RL = joblib.load("/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/output/RL_annotation_model")
coef_filtre=pd.read_csv("/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/pictures/image_demonstration/testingInputs/coefs_filtre_RQ.csv")








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
def filtre_quantile(table,coef_filtre=coef_filtre,height=2448,width=3264):
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






def filtre_thresh(imageA,imageB,contrast=-5,blockSize=51,blurFact = 25):
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





def filtre_masque(imageA,imageB,contrast=-5,blockSize=51,blurFact = 25):
    
    #Essayer petu être de squeezer le passage au grey ou bien au hsv
    #Essayer de poser un nouveau masque avec une forme plus simple
    path_images=folder+"/"
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    mask_path='/mnt/VegaSlowDataDisk/c3po_interface_mark/find_dominant_colors/how_to/masque/mask_0.npy'
    
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    #img2 = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    #img1 = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
    img1=imageA
    img2=imageB
    blurFact = blurFact
    absDiff2 = cv2.absdiff(img1, img2)
    diff = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
    th2 = cv2.adaptiveThreshold(src=absDiff2,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
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
def RecenterImage_bis(subI,o):
    
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



#On en a vraiment besoin encore
def to_polygon(line):
    xmin=line["xmin"]
    xmax=line["xmax"]
    ymin=line["ymin"]
    ymax=line["ymax"]
    polygon=Polygon( [(xmin,ymin ),(xmin,ymax),(xmax,ymax),(xmax,ymin)])
    return polygon






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



#Open target file to extract only the lines pour l'image étudiées et renvoie la  ou les classes d'animaux 
def open_imagettes_file(imagettes,folder,name_test):
    
    #Select only animals categories
    liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
    imagettes=to_reference_labels (imagettes,"classe")
    imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)]    
    
    
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
        print("attention il y a plusieurs espèces d'animaux" )
    #nom_classe=imagettes1["classe"].iloc[0]
    nom_classe=list(imagettes_target["classe"].unique())
    
    return nom_classe,imagettes_target




#Class imagettes in list according to ther coordonates
def class_imagettes(generate_square,coverage_threshold,
                        nom_classe,imagettes_target,dic_labels_to_num):

    
    
    #Initialize empty list and dictionnary
    liste_DIFF_birds_defined,liste_DIFF_birds_undefined,liste_DIFF_other_animals,liste_DIFF_faisan,liste_DIFF_corbeau,liste_DIFF_pigeon=([] for i in range(6))
    dict_anotation_index_to_classe={}

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
    
    
    
    
    #print( "nombre d'animaux à reperer", nb_animals_to_find)
    #print( "nombre d'animaux repérés", nb_animals_match)
    print("nombre d'oiseaux dans l'image",birds_to_find)
    #print("nombre d'oiseaux totaux repérés dans l'image",birds_match)
    print("nombre d'oiseaux repérés parmi les oiseaux labélisés",birds_defined_match )
    
    return (liste_Diff_animals,dict_anotation_index_to_classe,liste_DIFF_birds_defined,liste_DIFF_birds_undefined,birds_defined_match,
liste_Diff_birds_this_class,liste_DIFF_birds_defined_this_class,liste_DIFF_birds_undefined_this_class,liste_DIFF_other_animals_this_class,
liste_DIFF_corbeau,liste_DIFF_faisan,liste_DIFF_pigeon,liste_DIFF_other_animals)
    




def flat(list_of_list):
    return list(flatten(list_of_list))


def class_imagettes_bis(generate_square,coverage_threshold,
                        nom_classe,imagettes_target,dic_labels_to_num):

    
    
    #Initialize empty list and dictionnary
    liste_DIFF_birds_defined,liste_DIFF_birds_undefined,liste_DIFF_other_animals,liste_DIFF_faisan,liste_DIFF_corbeau,liste_DIFF_pigeon=([] for i in range(6))
    dict_anotation_index_to_classe={}

    birds_liste=["corneille","pigeon","faisan","oiseau","pie","incertain"]
    animals_liste=["lapin","chevreuil"]+birds_liste
    
    xmin_gen=generate_square["xmin"]
    xmax_gen=generate_square["xmax"]
    ymin_gen=generate_square["ymin"]
    ymax_gen=generate_square["ymax"]
    ln_square_gen=len(generate_square)
    
    


    #On classe les difference repéré pour les animaux classés
    for classe in nom_classe:
            
        
        liste_Diff_animals_this_class=[]
        
        #Initialisation du dic à utiliser plus tard pour récupérer pour chaque index la classe
        
        
        imagettes_annote_1_classe=imagettes_target[imagettes_target["classe"]==classe]
        nb_imagettes_1_classe=len(imagettes_annote_1_classe)
        

            
        #get max intersection with square generate for each sqaure annotate
        for i in range(nb_imagettes_1_classe):
            x_min_anote=imagettes_annote_1_classe["xmin"].iloc[i]
            x_max_anote=imagettes_annote_1_classe["xmax"].iloc[i]
            y_min_anote=imagettes_annote_1_classe["ymin"].iloc[i]
            y_max_anote=imagettes_annote_1_classe["ymax"].iloc[i]
            surface=area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
         
    
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
            
            index_intersection_match=[]
            
            for i, j in enumerate(liste_intersection):
                if j/surface >coverage_threshold :
                    index_intersection_match.append(i)
            
            
            
            #max_intersection=max(liste_intersection)
            
            
            
            if  (classe=="faisan"):
                liste_DIFF_faisan.append(index_intersection_match)        
            if  (  (classe=="corneille")    ):
                liste_DIFF_corbeau.append(index_intersection_match)                            
            if (  (classe=="pigeon")   ):
                liste_DIFF_pigeon.append(index_intersection_match)    
            if  (  (classe=="oiseau")   or (classe=="pie") or (classe=="incertain") ):
                liste_DIFF_birds_undefined.append(index_intersection_match)
            if  (  (classe=="lapin")   or (classe=="chevreuil") ):
                liste_DIFF_other_animals.append(index_intersection_match)     
                
                
             
            (liste_DIFF_faisan,liste_DIFF_corbeau,liste_DIFF_pigeon,liste_DIFF_birds_undefined,liste_DIFF_other_animals)=( 
                    flat(liste_DIFF_faisan),flat(liste_DIFF_corbeau),
            (flat(liste_DIFF_pigeon)),flat(liste_DIFF_birds_undefined),flat(liste_DIFF_other_animals)  )
            
            
            liste_DIFF_birds_defined=liste_DIFF_faisan+liste_DIFF_corbeau+liste_DIFF_pigeon
            liste_Diff_birds=liste_DIFF_birds_defined+liste_DIFF_birds_undefined
            liste_Diff_animals=liste_Diff_birds+liste_DIFF_other_animals

            index_anote=index_intersection_match
            for i in index_anote:
                dict_anotation_index_to_classe[str(i)]=dic_labels_to_num[classe]
                
                
                
                
    #Capture du nombre d'imagettes à caputerer et captutées   
    birds_table=imagettes_target.isin(birds_liste)
    nb_animals_match=len(liste_DIFF_birds_defined)+len(liste_DIFF_birds_undefined)+len(liste_DIFF_other_animals)
    nb_animals_to_find=len(imagettes_target)
    birds_to_find=len(birds_table)
    birds_defined_match=len(liste_DIFF_birds_defined)
    print("nombre d'oiseaux dans l'image",birds_to_find)
    print("nombre d'oiseaux repérés parmi les oiseaux labélisés",birds_defined_match )
    
    return (liste_Diff_animals,dict_anotation_index_to_classe,liste_DIFF_birds_defined,liste_DIFF_birds_undefined,birds_defined_match,
liste_DIFF_corbeau,liste_DIFF_faisan,liste_DIFF_pigeon,liste_DIFF_other_animals)









#Class prediction in list
def class_predictions(liste_prediction): 
   
    index_others,index_birds,index_other_animals,index_chevreuil,index_corbeau,index_lapin,index_faisan,index_pigeon = ([] for i in range(8))
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








#reorganize imagettes, coordonates and prediction  in a dataframe 
def table_gen_im(table,FP,liste_Diff_animals,dict_anotation_index_to_classe):
    
    liste_imagette_names=[]
    path_liste=[]
    filename_liste=[]
    col=['path','filename', 'imagetteName','max_cat','predict_cat','xmin', 'xmax', 'ymin', 'ymax',   
         'former_index']
    range_table=range(len(table))
    
    
    for i in  range_table:
        name=(table["filename"].iloc[i].split("/")[-1]).split(".")[0]+"_"+str(table.index[i])+"_"+str(round(table["max_cat"].iloc[i],2))[2:]+"_"+str(table["predict_cat"].iloc[i])+".jpg"
        liste_imagette_names.append(name)
    table["imagetteName"]=liste_imagette_names



    for i in  range_table:
        string=table["filename"].iloc[i]
        path="."+"/".join(string.split("/")[:3])
        path_liste.append(path)
    table["path"]= path_liste



    
    for i in  range_table:
        string=table["filename"].iloc[i]
        filename=string.split("/")[-1]
        filename_liste.append(filename)
    table["filename"]=filename_liste


    #A partir ligne par ligne
    table=table[col]
    table["IsFP"]=0
    for i in FP:
        table["IsFP"].loc[i]=1
        
        
    table["reel_classe"]=0
    for i in liste_Diff_animals:
        table["reel_classe"].loc[i]=dict_anotation_index_to_classe[str(i)]

    return table








#draw colorated rectangles in image 
def draw_rectangle(table,color,image_name):

    for i in range(len(table)):
        xmin=table["xmin"].iloc[i]
        ymin=table["ymin"].iloc[i]
        xmax=table["xmax"].iloc[i]
        ymax=table["ymax"].iloc[i]
        
        if color=="Blue":
            cv2.rectangle(image_name, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 
        if color=="Green":
            cv2.rectangle(image_name, (xmin,ymin), (xmax,ymax), (0, 255,0), 2) 
        if color=="Red":
            cv2.rectangle(image_name, (xmin,ymin), (xmax,ymax), (0, 0,255), 2) 
        if color=="White":
            cv2.rectangle(image_name, (xmin,ymin), (xmax,ymax), (255, 255,255), 2)
        
    return image_name
            
            




    


def batched_cnts(cnts,imageB,imageSize= 28,filtre_color=False):
    
    #Initialisation de variables et de liste
    batchImages = []
    liste_table = []
    
    if filtre_color==False:
        for ic in range(0,len(cnts)):
            (x, y, w, h) = cv2.boundingRect(cnts[ic])
            f = pd.Series(dtype= "float64")
            f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
            subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
            subI = RecenterImage(subI,o)
            subI = cv2.resize(subI,(imageSize,imageSize))
            batchImages.append(subI)
            liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))     
    
    
    if filtre_color==True:
        
        for ic in range(0,len(cnts)):
            (x, y, w, h) = cv2.boundingRect(cnts[ic])
            f = pd.Series(dtype= "float64")
            f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
            subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
            subI = RecenterImage(subI,o)
            subI = cv2.resize(subI,(imageSize,imageSize))
            
            red=np.mean(subI[:,:,0])
            green=np.mean(subI[:,:,1])
            blue=np.mean(subI[:,:,2])
            colors=(red,green,blue)
            
            #or !=
            if np.max(colors)==blue:
                batchImages.append(subI)
                liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))     
    
    
    batchImages_stack = np.vstack(batchImages)
    batchImages_stack_reshape=batchImages_stack.reshape((-1, imageSize,imageSize,3))
    table_non_filtre = pd.DataFrame(np.vstack(liste_table))
    table_non_filtre = table_non_filtre.rename(columns={ 0: 'xmin', 1: 'xmax', 2: 'ymin', 3: 'ymax'})
    table_non_filtre=table_non_filtre.astype(int)
    
    
    return batchImages_stack_reshape,table_non_filtre  




def dictionnaire_conversion_mclasses(numb_classes):
    dic_labels_to_num={}
    dic_num_to_labels={}
    
    liste_8_classes=["arbre","chevreuil","ciel","corneille","faisan","lapin","pigeon","terre","oiseau"]
    liste_6_classes=["autre","chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
    
    if numb_classes==6:
        liste_classes=liste_6_classes
    if numb_classes==8:
        liste_classes=liste_8_classes
    
    for i, j in enumerate(liste_classes):
        dic_labels_to_num[j]=i
        dic_num_to_labels[i]=j

    
    return dic_labels_to_num,dic_num_to_labels






def class_predictions_mclasses(liste_prediction,class_num): 
   index_others,index_birds,index_other_animals,index_chevreuil,index_corbeau,index_lapin,index_faisan,index_pigeon = ([] for i in range(8))
   
   if class_num==8:
        
        for i, j in enumerate(liste_prediction):
            
            if j == 0 or (j == 2) or(j == 7):
                index_others.append(i)
            if j==1:
                index_chevreuil.append(i)
            if j==3:
                index_corbeau.append(i)            
            if j==4:
                index_lapin.append(i)
            if j==5:
                index_faisan.append(i)
            if j==6:
                index_pigeon.append(i)
                
                
   if class_num==6:    
                
        for i, j in enumerate(liste_prediction):
            if j == 0:
                index_others.append(i)
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
           
   index_birds=index_corbeau+index_faisan+index_pigeon
   index_other_animals=index_chevreuil+index_lapin
   return index_others,index_birds,index_other_animals,index_chevreuil,index_corbeau,index_lapin,index_faisan,index_pigeon


    



def mask_function(folder,cnts,imageB,mask):
    
    if folder=='/DonneesPI/timeLapsePhotos_Pi1_4':
        mask_path='/mnt/VegaSlowDataDisk/c3po_interface_mark/find_dominant_colors/how_to/masque/mask_4_precis.npy'
    if folder=='/DonneesPI/timeLapsePhotos_Pi1_3':
        mask_path='/mnt/VegaSlowDataDisk/c3po_interface_mark/find_dominant_colors/how_to/masque/mask_3.npy'
    if folder=='/DonneesPI/timeLapsePhotos_Pi1_2':
        mask_path='/mnt/VegaSlowDataDisk/c3po_interface_mark/find_dominant_colors/how_to/masque/mask_2.npy'
    if folder=='/DonneesPI/timeLapsePhotos_Pi1_1':
        mask_path='/mnt/VegaSlowDataDisk/c3po_interface_mark/find_dominant_colors/how_to/masque/mask_1.npy'  
    if folder=='/DonneesPI/timeLapsePhotos_Pi1_0':
        mask_path='/mnt/VegaSlowDataDisk/c3po_interface_mark/find_dominant_colors/how_to/masque/mask_0.npy'



    
    if mask==True:
        print("masque activée")
        mask_image = load(mask_path)
        ImageMaksed=np.multiply(imageB, mask_image) 
        imageB=ImageMaksed.astype(int)
        batchImages_mask,table_non_filtre=batched_cnts(cnts,imageB)
    #No Mask
    if mask==False:
        batchImages_mask,table_non_filtre=batched_cnts(cnts,imageB)


    return batchImages_mask,table_non_filtre,imageB







#Ensemble des choix filtres
#Il ya un probleme sur le troisième choix
#Mais aussi pour le batchImages_stack_reshape
def filtre_line(table_non_filtre,filtre_choice):

    #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
    if filtre_choice=="No_filtre":
        generate_square=table_non_filtre
    
        
    elif filtre_choice=="quantile_filtre":
        table_quantile,index_possible_animals=filtre_quantile(table_non_filtre)
        generate_square=table_quantile
        batchImages_stack_reshape_quantile=[batchImages_stack_reshape[i] for i in index_possible_animals ]
        batchImages_stack_reshape=np.array(batchImages_stack_reshape_quantile)
        
    elif filtre_choice=="RL_filtre":
        generate_square=table_filtre_RL

    return generate_square





def diff_im_slow(cnts,range_cnts,imageA,imageB):
    Diff_image=[]
    for ic in range_cnts:
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        imagetteA, o, d, imageRectangles = GetSquareSubset(imageA,f,verbose=False)
        imagetteB, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        hsvA=cv2.cvtColor(imagetteA, cv2.COLOR_RGB2HSV)
        flat_hsvA=list(hsvA.reshape(-1))
        hsvB=cv2.cvtColor(imagetteB, cv2.COLOR_RGB2HSV)
        flat_hsvB=list(hsvB.reshape(-1).astype(int))
        Diff_pixel=list(map(distance_hsv,flat_hsvA,flat_hsvB))
        Diff_image.append(Diff_pixel)
    return Diff_image






def diff_im_slow_bis(cnts,range_cnts,imageA,imageB):
    Diff_image=[]
    
    #myfunc_vec = np.vectorize(distance_hsv_bis)
    #myfunc_vec(x,y)
    
    for ic in range_cnts:
        #start = time.time()
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        imagetteA, o, d, imageRectangles = GetSquareSubset(imageA,f,verbose=False)
        imagetteB, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)

        
        flat_hsvA=list(imagetteA.reshape(-1).astype(int))
        flat_hsvB=list(imagetteB.reshape(-1).astype(int))

        

        Diff_pixel=list(map(distance_hsv_bis,flat_hsvA,flat_hsvB))
        Diff_image.append(Diff_pixel)
        #end=time.time()
        #print(end -start)
    return Diff_image




#return the difference pixel by pixel expriming in hsv after and before the image substraction with faster method
def list_hsv(cnts,range_cnts,imageA,imageB):
    Diff_image=[]
    flat_hsvA_liste=[]
    flat_hsvB_liste=[]
    #myfunc_vec = np.vectorize(distance_hsv_bis)
    #myfunc_vec(x,y)
    
    for ic in range_cnts:
        #start = time.time()
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
        imagetteA, o, d, imageRectangles = GetSquareSubset(imageA,f,verbose=False)
        imagetteB, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)

        
        flat_hsvA=list(imagetteA.reshape(-1).astype(int))
        flat_hsvB=list(imagetteB.reshape(-1).astype(int))
        
        flat_hsvA_liste.append(flat_hsvA)
        flat_hsvB_liste.append(flat_hsvB)

    distance_vec = np.vectorize(distance_hsv_bis)
    Diff_image=distance_vec(flat_hsvA_liste,flat_hsvB_liste)
    
        #end=time.time()
        #print(end -start)
    return Diff_image





def rearrange_dif(liste_DIFF_birds_defined,liste_DIFF_birds_undefined,liste_DIFF_other_animals,birds_defined_match,batchImages_stack_reshape):
    
    liste_batch_images=range(len(batchImages_stack_reshape))
    liste_Diff_birds=liste_DIFF_birds_defined+liste_DIFF_birds_undefined
    liste_Diff_animals=liste_Diff_birds+liste_DIFF_other_animals
    birds_match=birds_defined_match+len(liste_DIFF_birds_undefined)
    liste_Diff_not_birds=set(liste_batch_images)-set(liste_Diff_birds)
    liste_DIFF_not_matche=set(liste_batch_images)-set(liste_Diff_birds)-set(liste_DIFF_other_animals)
    print("nombre d'oiseaux capturés", birds_match)

    return liste_Diff_birds,liste_Diff_animals,birds_match,liste_Diff_not_birds,liste_Diff_animals,liste_DIFF_not_matche




#distance in hsv for all imagette of an image without any distinction of type
def extract_distance(path_images,name_test,name_ref,folder,contrast=-5,blockSize=53,blurFact=15):
            
    
    #Initialisation de variables et de liste

    Diff_image=[]

    
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
    range_cnts=range(0,len(cnts))
    
    #Make difference of pixel in hsv
    Diff_image=diff_im_slow(cnts,range_cnts,imageA,imageB)
    

 
    return Diff_image




def distance_hsv_bis(pixelA,pixelB):
    min_dist=min([pixelB-pixelA,pixelB-pixelA-360,pixelB-pixelA+360], key=abs)

    return min_dist




def distance_hsv(imageA,imageB):
    minimum=min(np.abs(imageB-imageA),np.abs(imageB-imageA-360),np.abs(imageB-imageA+360))
    max([3, 7, -10], key=abs)

    if minimum==np.abs(imageB-imageA):
        resultat=(imageB-imageA)
    elif minimum==np.abs(imageB-imageA-360):
        resultat=(imageB-imageA-360)
    elif minimum==np.abs(imageB-imageA+360):
        resultat=(imageB-imageA+360)  
    return resultat

#return the difference pixel by pixel expriming in hsv after and before the image substraction



#Classe prediction and estimation in fonction in True of False negative
def miss_well_class(estimates,liste_Diff_birds,liste_DIFF_not_matche,
                    liste_DIFF_corbeau,liste_DIFF_faisan,liste_DIFF_pigeon,liste_Diff_animals,
                    thresh_active,thresh,numb_classes=6,focus="bird_large"):
    
    
    liste_prediction=list(estimates.argmax(axis=1))
    
    #liste_prediction_animals_match=[liste_prediction[i] for i in liste_Diff_animals]
    
    index_others,index_birds,index_other_animals,index_chevreuil,index_corbeau,index_lapin,index_faisan,index_pigeon=class_predictions_mclasses(liste_prediction,numb_classes)
    index_animals=index_birds+index_other_animals
    
    
    if focus=="bird_large":
        others_pr_birds=set(liste_DIFF_not_matche).intersection(index_birds)
        birds_pr_birds=set(liste_Diff_birds).intersection(index_birds)
        TP=birds_pr_birds
        FP=others_pr_birds
        
        
        FP_birds_estimates=[max(estimates[i]) for i in (others_pr_birds)]
        birds_pr_bird_estimates=[max(estimates[i]) for i in birds_pr_birds]
        TP_estimates=birds_pr_bird_estimates
        FP_estimates=FP_birds_estimates
        
    if focus=="bird_precise":
        TP_birds=list(set(liste_DIFF_corbeau).intersection(index_corbeau))+list(set(liste_DIFF_faisan).intersection(index_faisan))+list(set(liste_DIFF_pigeon).intersection(index_pigeon))
        TP=TP_birds 
        print("attention on a pas encore reglé FP")
        
        TP_birds_estimates=[estimates[i] for i in (TP_birds)]
        TP_estimates=TP_birds_estimates
        FP_estimates=[max(estimates[i]) for i in (FP)]
        
    if focus=="animals":
        FP_animals=set(liste_DIFF_not_matche).intersection(index_animals)
        animals_predict=set(liste_Diff_animals).intersection(index_animals)
        TP=animals_predict
        FP=FP_animals
        
        other_animals_estimates = [max(estimates[i]) for i in (index_other_animals)]
        FP_estimates=other_animals_estimates
        
        animals_match_estimates=[max(estimates[i]) for i in liste_Diff_animals]
        TP_estimates=animals_match_estimates
        
    if thresh_active==True:

        TP_estimates=[i for i in TP_estimates if i > thresh]
        FP_estimates=[i for i in FP_estimates if i > thresh]
        
        
    
    return TP,FP,TP_estimates,FP_estimates
    
    



#Cette fonction devrait servir pour toutes les autres fonctions
def base(name_test,name_ref,folder,CNNmodel
                                 ,numb_classes=6,mask=True,coverage_threshold=0.99,contrast=-5,blockSize=53,blurFact=15,filtre_choice="No_filtre", thresh=0.9, thresh_active=True):
    

    
    #Dictionnary to convert string labels to num labels
    dic_labels_to_num,dic_num_to_labels=dictionnaire_conversion_mclasses(numb_classes)
    #Definition des images
    path_images=folder+"/"
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
    
    

    #On peut rassemble les deux dans la meêm image
    #On pourrait changer filtre_line pour rajouter Filename ou bien imagetteName
    batchImages_stack_reshape,table_non_filtre,imageB=mask_function(folder,cnts,imageB,mask)    
    generate_square=filtre_line(table_non_filtre,filtre_choice)
    
    
    #On classe les imagettes générées en fonction de l'espèce avec laquelle ses coordonnées correspondent sur l'image
    
    (liste_Diff_animals,dict_anotation_index_to_classe,liste_DIFF_birds_defined,liste_DIFF_birds_undefined,birds_defined_match,liste_DIFF_corbeau,
     liste_DIFF_faisan,liste_DIFF_pigeon,liste_DIFF_other_animals)=class_imagettes_bis(generate_square,coverage_threshold, nom_classe,imagettes_target,dic_labels_to_num) 
    
    (liste_Diff_birds,liste_Diff_animals,birds_match,liste_Diff_not_birds,liste_Diff_animals,
    liste_DIFF_not_matche)=rearrange_dif(liste_DIFF_birds_defined,liste_DIFF_birds_undefined,liste_DIFF_other_animals,birds_defined_match,batchImages_stack_reshape)
    
    
    #On effectue les predictions et on les classes les index selon l'espèce
    estimates = CNNmodel.predict(batchImages_stack_reshape)
    TP_birds,FP,TP_estimates,FP_estimates= miss_well_class(estimates,liste_Diff_birds,liste_DIFF_not_matche,
                    liste_DIFF_corbeau,liste_DIFF_faisan,liste_DIFF_pigeon,liste_Diff_animals,
                    thresh_active,thresh,numb_classes=6,focus="bird_large")
    

    return imageA,imageB,cnts,batchImages_stack_reshape,generate_square,
    TP_birds,FP,TP_estimates,FP_estimates





def extract_dist_by_type_shorter(name_test,name_ref,folder,CNNmodel,numb_classes
                                 ,mask=True,coverage_threshold=0.99,contrast=-5,blockSize=53,blurFact=15,filtre_choice="No_filtre", thresh=0.9, thresh_active=True):
    

    
    (imageA,imageB,cnts,batchImages_stack_reshape,generate_square,
    TP,FP,TP_estimates,FP_estimates)= base(name_test,name_ref,folder,CNNmodel)
    

    
    Diff_image_FP=diff_im_slow_bis(cnts,FP,imageA,imageB)
    Diff_image_animals=diff_im_slow_bis(cnts,TP,imageA,imageB)
    Diff_image_total=diff_im_slow_bis(cnts,range(len(cnts)),imageA,imageB)

    return Diff_image_FP,Diff_image_animals,Diff_image_total




#Gen imagettes tp and fp before to make colors test
def colors_imagettes(name_test,name_ref,folder,CNNmodel,numb_classes,to_Select="FP",coverage_threshold=0.99,contrast=-5,blockSize=53,blurFact=15,
            filtre_choice="No_filtre", thresh=0.9, thresh_active=True):
    
    
    (imageA,imageB,cnts,batchImages_stack_reshape,generate_square,
    TP,FP,TP_estimates,FP_estimates)= base(name_test,name_ref,folder,CNNmodel,numb_classes)
    
    
    #Draw Rectangles
    imageRectangle=imageB.copy()
    birds_square=generate_square.loc[TP]
    draw_rectangle(birds_square,"Green",imageRectangle)
    FP_square=generate_square.loc[FP]
    draw_rectangle(FP_square,"Blue",imageRectangle)
    image_path="/mnt/VegaSlowDataDisk/c3po_interface_mark/bin/image_squared/bird.jpg"
    cv2.imwrite(image_path,imageRectangle)
    print("importation réussie")

    

    chdir("/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/pictures/imagettes_colors/")
    #On va créer une fonction pour extaire les imagettes avec itération selon cnts pour fp et avec un i 
    batchImages_FP=[batchImages_stack_reshape[i] for i in FP ]     
    batchImages_match_animals=[batchImages_stack_reshape[i] for i in FP ]  
    #batch_to_select=batchImages_FP+batchImages_match_animals
    
    
    if to_Select=="FP":
        batch_to_select=batchImages_FP
    
    if to_Select=="TP":
        batch_to_select=batchImages_match_animals

    for i in range(len(batch_to_select) ):
        name=str(i)+".jpg"
        image=batch_to_select[i]
        cv2.imwrite(name,image)
        
        
        
        
#return birds match and fp if they are above 0.9 , have a short function
def predict_8classes(name_test,name_ref,folder,CNNmodel,numb_classes,mask=True,coverage_threshold=0.99,contrast=-5,blockSize=53,blurFact=15,
            filtre_choice="No_filtre", thresh=0.9, thresh_active=True):
    

    (imageA,imageB,cnts,batchImages_stack_reshape,generate_square,
    TP,FP,TP_estimates,FP_estimates)= base(name_test,name_ref,folder,CNNmodel,numb_classes)
    ln_FP=len(FP)
    ln_TP=len(TP)

    return ln_FP, ln_TP

        

def draw_fp(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=53,blurFact=15,
            filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
    (imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP,FP,TP_estimates,FP_estimates)= base(name_test,name_ref,folder,CNNmodel)
    
    
    #Draw Rectangles
    imageRectangle=imageB.copy()
    birds_square=generate_square.loc[TP]
    draw_rectangle(birds_square,"Green",imageRectangle)
    FP_square=generate_square.loc[FP]
    draw_rectangle(FP_square,"Blue",imageRectangle)
    image_path="/mnt/VegaSlowDataDisk/c3po_interface/bin/image_squared/birds.jpg"
    cv2.imwrite(image_path,imageRectangle)
    

    return len(TP)






#return birds match and fp if they are above 0.9 , have a short function
def predict_under_thres(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.99,contrast=-5,blockSize=53,blurFact=15,
            filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000, thresh=0.9, thresh_active=True):
    
    (imageA,imageB,cnts,batchImages_stack_reshape,generate_square,
    TP,FP,TP_estimates,FP_estimates)= base(name_test,name_ref,folder,CNNmodel)
    
    (ln_FP,ln_birds)=(len(FP),len(TP))
 
    return ln_FP,ln_birds




#extract fp, match animals or other category with a short script
#Modifié par rapport à l'original
    
#def extract_fp_court(path_images,name_test,name_ref,folder,CNNmodel,coverage_threshold=0.5,contrast=-5,blockSize=53,blurFact=15,
#            filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264,limit_area_square=100000000000):
    
def extract_fp_court(name_test,name_ref,folder,CNNmodel):

    
    (imageA,imageB,cnts,batchImages_stack_reshape,generate_square,
    TP,FP,TP_estimates,FP_estimates)= base(name_test,name_ref,folder,CNNmodel)

    
    #On va créer une fonction pour extaire les imagettes avec itération selon cnts pour fp et avec un i 
    batchImages_FP=[batchImages_stack_reshape[i] for i in FP ]     
    batchImages_match_animals=[batchImages_stack_reshape[i] for i in TP ]  
    batch_to_select=batchImages_FP+batchImages_match_animals
    #batchImages_FP=[batchImages[i] for i in FP +liste_Diff_animals ]   
    #chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images")
    
    
    chdir("/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/pictures/BD_ML/fp_4C/")
    
    table=generate_square
    for i in range( len(table) ):
        name=(table["filename"].iloc[i].split("/")[-1]).split(".")[0]+"_"+str(table.index[i])+"_"+str(round(table["max_cat"].iloc[i],2))[2:]+"_"+str(table["predict_cat"].iloc[i])+".jpg"
        image=batch_to_select[i]
        cv2.imwrite(name,image)
        
   
   
    return table,TP










#Cette fonction devrait servir de base si on utilise 4 canaux au lieu de 3
def base_4C(name_test,name_ref,folder,CNNmodel,
                                 diff_mod="HSV",numb_classes=6,mask=True,coverage_threshold=0.99,contrast=-5,blockSize=53,blurFact=15,filtre_choice="No_filtre", thresh=0.9, thresh_active=True):
    

    imageSize=28
    #Dictionnary to convert string labels to num labels
    dic_labels_to_num,dic_num_to_labels=dictionnaire_conversion_mclasses(numb_classes)
    #Definition des images
    path_images=folder+"/"
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    

    

    
    
    
    #Ouverture des fichiers annotés  si on voulait gagner du temps on pourrait le sortir de la fonction
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    nom_classe,imagettes_target=open_imagettes_file(imagettes,folder,name_test)
    #On effectue une différentiation des images et on récupères le filtre des images sous forme de batch
    cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    


    #Deuxième temps ouvrir un masque ex:
    #'/mnt/VegaSlowDataDisk/c3po_interface_mark/find_dominant_colors/how_to/masque/mask_4_precis.npy'
    batchImages_stack_reshape,table_non_filtre,imageB=mask_function(folder,cnts,imageB,mask=False)    
    
    Diff=diff_filtre(imageA,imageB,method=diff_mod)


    subI_diff_liste=[]
    for i in range(len(table_non_filtre)):
        
        x_min, x_max, y_min, y_max=get_table_coord(table_non_filtre.iloc[i])
        
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x_min, x_max, y_min, y_max
        
        subI_diff, o, d, imageRectangles = GetSquareSubset(Diff,f,verbose=False)
        subi_expand=np.expand_dims(subI_diff,axis=2)
        subI_recenter = RecenterImage(subi_expand,o)
        subI_resize = cv2.resize(subI_recenter,(imageSize,imageSize))
        subI_resize=np.expand_dims(subI_resize,axis=2)
        subI_diff_liste.append(subI_resize)
        
    #On en fait quoi alors
    list_4C=list(map(add_chanel,batchImages_stack_reshape,subI_diff_liste))
    batchImages_stack_reshape=np.array(list_4C)
    
    
    #On peut aussi mettre ici batchImages4C apply func chanel
    generate_square=filtre_line(table_non_filtre,filtre_choice)
    
    
    #On classe les imagettes générées en fonction de l'espèce avec laquelle ses coordonnées correspondent sur l'image
    
    (liste_Diff_animals,dict_anotation_index_to_classe,liste_DIFF_birds_defined,liste_DIFF_birds_undefined,birds_defined_match,liste_DIFF_corbeau,
     liste_DIFF_faisan,liste_DIFF_pigeon,liste_DIFF_other_animals)=class_imagettes_bis(generate_square,coverage_threshold, nom_classe,imagettes_target,dic_labels_to_num) 
    
    (liste_Diff_birds,liste_Diff_animals,birds_match,liste_Diff_not_birds,liste_Diff_animals,
    liste_DIFF_not_matche)=rearrange_dif(liste_DIFF_birds_defined,liste_DIFF_birds_undefined,liste_DIFF_other_animals,birds_defined_match,batchImages_stack_reshape)
    
    
    #On effectue les predictions et on les classes les index selon l'espèce
    #
    estimates = CNNmodel.predict(batchImages_stack_reshape)
    
    #estimates = model_import.predict(batchImages_stack_reshape)
    TP_birds,FP,TP_estimates,FP_estimates= miss_well_class(estimates,liste_Diff_birds,liste_DIFF_not_matche,
                    liste_DIFF_corbeau,liste_DIFF_faisan,liste_DIFF_pigeon,liste_Diff_animals,
                    thresh_active,thresh,numb_classes=6,focus="bird_large")
    

    return imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates




### Partie test



def filtre_light_4C(imageA,imageB,contrast=-5,blockSize=51,blurFact = 25):

    diff_Gray=diff_filtre(imageA,imageB)
    
    blurFact = blurFact
    th2 = cv2.adaptiveThreshold(src=diff_Gray,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
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

def diff_filtre(imageA,imageB,method="HSV"):
    if method=="HSV":
        img2 = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
        img1 = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
        absDiff2 = cv2.absdiff(img1, img2)
        diff_Gray = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
        
    elif method=="BGR":
        absDiff2 = cv2.absdiff(imageA, imageB)
        diff_Gray = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
        
    else:
        print("erreur de saisie de la méthode")
        
    return diff_Gray
    
        
    
#On va faire une fonction avec table_non_filtre en prenant les coords de xmin...
#Mais ça va faire une étape en plus, l'idéal ce serait de changer direct la fonction mask_function
#C'est ce que je vais faire dans un second temps avec un if 4C et 3C par defaut
    

def add_chanel(img,array_to_add):
    b_channel, g_channel, r_channel = cv2.split(img)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, array_to_add))
    
    return img_BGRA






def batched_cnts_4C(diff,cnts,imageB,imageSize= 28,filtre_color=False):
    
    #Initialisation de variables et de liste
    batchImages = []
    liste_table = []
    
    #Change that
    diff=HSV_Diff
    
    if filtre_color==False:
        for ic in range(0,16):
            (x, y, w, h) = cv2.boundingRect(cnts[ic])
            f = pd.Series(dtype= "float64")
            f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
            #Il faut dédoubler ça pour le 3eme canal... . 
            subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
            subI = RecenterImage(subI,o)
            subI_resize = cv2.resize(subI,(imageSize,imageSize))
            
            #Donc chaque subI est déjà en 28 et f_xmin... aussi, si c'est le cas ça va être facile
            #Si x_min...,ymax déjà en 28,28 on peut sortir map add_chanel de la fonction
            batchImages.append(subI)
            liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))     
    
    
    if filtre_color==True:
        
        for ic in range(0,len(cnts)):
            (x, y, w, h) = cv2.boundingRect(cnts[ic])
            f = pd.Series(dtype= "float64")
            f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
            subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
            subI = RecenterImage(subI,o)
            subI = cv2.resize(subI,(imageSize,imageSize))
            
            red=np.mean(subI[:,:,0])
            green=np.mean(subI[:,:,1])
            blue=np.mean(subI[:,:,2])
            colors=(red,green,blue)
            
            #or !=
            if np.max(colors)==blue:
                batchImages.append(subI)
                liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))     
    
    
    #create with coordonates of imagettes
    table_non_filtre = pd.DataFrame(np.vstack(liste_table))
    table_non_filtre = table_non_filtre.rename(columns={ 0: 'xmin', 1: 'xmax', 2: 'ymin', 3: 'ymax'})
    table_non_filtre=table_non_filtre.astype(int)
    
    
    #Ici il faut surveiller
    #Dabord regarder s'il n'y a pas un resize qui se balade
    batchImages_stack_reshape_trans=list(map(add_chanel,batchImages_stack_reshape,HSV_Diff))
    
    batchImages_stack = np.vstack(batchImages)
    batchImages_stack_reshape=batchImages_stack.reshape((-1, imageSize,imageSize,3))
    

    
    
    return batchImages_stack_reshape,table_non_filtre  
    
#extract x_mi,x_max,y_min,y_max
def get_table_coord(table_line):
    x_min=table_line["xmin"]
    y_min=table_line["ymin"]
    x_max=table_line["xmax"]
    y_max=table_line["ymax"]
    
    return x_min,x_max,y_min,y_max


#Poue avoir un resultat en 3ème dim ou en 1 dim
def RecenterImage(subI,o):
    
    h,l,r=subI.shape #on sait que r=3 (3 channels)
    
    # add to the dimension the dimensions of the cuts (due to image borders)
    h = h + o.ymincut + o.ymaxcut
    l = l + o.xmincut + o.xmaxcut

    t= np.full((h, l, r), fill_value=int(round(subI.mean())),dtype=np.uint8) # black image the size of the final thing

    t[o.ymincut:(h-o.ymaxcut),o.xmincut:(l-o.xmaxcut)] = subI
    
    return t

