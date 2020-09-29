#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os # to manipulate files

## flag file, to tell UI that we started (Stop is what should be displayed on the button)
flagFile = "signalFiles/flagBoucle.txt"
sF = open(flagFile,'w')
sF.write("Stop")
sF.close()

## signal file, needs to be at the very beginning to avoid error in stopping
signalFile = "signalFiles/signalfile.txt"
sF = open(signalFile,'w')
sF.write("global stop\nstop=False")
sF.close()

## import the necessary packages
from skimage.measure import compare_ssim
#import argparse
import imutils


import cv2
import sys,getopt
import time
import datetime
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
#import statistics as st
#from scipy.stats.mstats import mquantiles
#from numpy import loadtxt
#from scipy.optimize import leastsq
#import pylab
#from matplotlib import pyplot as plt
from sklearn.externals import joblib 
from sklearn.preprocessing import LabelEncoder
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import pickle
import random
from keras.applications import VGG16
import pandas as pd
from IPython import embed # for debug

# check if user just tried the button but quickly repented

def ShouldItStop():
    exec(open(signalFile).read())
    if(stop):
        print("Boucle.py received stop signal from signal file. Stopping...")
    
        sF = open(flagFile,'w')
        sF.write("Start")
        sF.close()
        
        print("Boucle.py stopped.")
        sys.exit()

ShouldItStop()
    
## tableau final 
column_name = ['name', 'xmin', 'xmax', 'ymin', 'ymax', 'max_probas']

## Choose the method and the model
blurFact=25
use_diff = "ssim" # "None", "ssim","diff","HSVabsdiff","absdiff"
    # None: just acquisition of the images
    # Note: this is the default, options superseed this 

maxAnalDL = 12 # maximum number of "imagette" to analyze
#Filtre = "NoFiltre" # "RandomForesttable = pd.Series([name, x,x+w, y, y+h])","NoFiltre","zones"
waitTime = 15 # in seconds, 15 still good for testing

## Durée de fonctionnement 
# heure de début de prises de vue 
timeOnHours = 4      #datetime.time(5,0,0)   attention timeOn doit être plus petit que timeOff
timeOnMinutes = 15      #datetime.time(5,0,0)   attention timeOn doit être plus petit que timeOff
# heure de fin de prises de vue
timeOffHours = 21    # datetime.ctime(22)
timeOffMinutes = 15


if(timeOnHours > timeOffHours):
    start = datetime.datetime.today()
    start = start.replace(hour = timeOnHours, minute = timeOnMinutes, second = 0)
    fin = datetime.datetime.today()
    fin = fin.replace(day = fin.day+1, hour = timeOffHours, minute = timeOffMinutes, second = 0)

else:
    start = datetime.datetime.today()
    start = start.replace(hour = timeOnHours, minute = timeOnMinutes, second = 0)
    fin = datetime.datetime.today()
    fin = fin.replace(hour = timeOffHours, minute = timeOffMinutes, second = 0)

duration = 3600*24-(fin-start).seconds

# base folder
c3poFolder = "/var/www/html/c3po_interface/"

# Dossier contenant les imagettes 
p = c3poFolder+"timeLapsePhotos/diff_subImages"

# change use_diff if specified in options
# # should work with the following but is not :
# try:
#     opts,args = getopt.getopt(sys.argv,"t:")
# except getopt.GetoptError:
#     print('Boucle.py -t <use_diff>')
#     sys.exit(2)
# for opt, arg in opts:
#     if opt == '-t':
#         print('use_diff='+arg)
#         use_diff=arg

# so doing it my way (likely buggy in the long run...)
for iOpt in range(0,len(sys.argv)-1):
    opt = sys.argv[iOpt]
    if opt == '-t':
        arg=sys.argv[iOpt+1]
        print('use_diff='+arg)
        use_diff=arg


today = datetime.datetime.today().strftime("%b-%d-%Y")
if use_diff != "None":
    # load the VGG16 network and initialize the label encoder
    print("[INFO] loading network...")
    model = VGG16(weights="imagenet", include_top=False)


    # load the trained model
    Model1 = joblib.load(c3poFolder+"bin/output/model.cpickle")
    filtre_RL = joblib.load(c3poFolder+"bin/output/RL_annotation_model")
    
    ## CSV contenant les résultats finaux
    csvPath = os.path.sep.join([c3poFolder+"bin/output",
                                    "{}.csv".format("new_images_" + today)])
    
    #tableau = pd.DataFrame(columns = column_name)
    #tableau.to_csv(csvPath, sep ='\t')
    
    
    
    # defines the kernel00.
    kernel = np.ones((blurFact,blurFact),np.uint8)

exec(open(c3poFolder+"/bin/functions.py").read())

## Folder containing timeLapse pictures
subImagesDir = c3poFolder+"timeLapsePhotos/"  
if not os.path.exists(subImagesDir):
    os.makedirs(subImagesDir)

ShouldItStop()
 
## First picture

imageA, timeStamp1 = CapturePi()


# When everything done, release the capture
name1 = subImagesDir+"image_"+timeStamp1+".JPG"  
#name1 = name1.split("/")[-1]

cv2.imwrite(name1,imageA)
print("First image written",name1.split("/")[-1],"\n")
sys.stdout.flush()

# for the SSIM function 
a=cv2.GaussianBlur(imageA,(blurFact,blurFact),sigmaX=0)

if(use_diff == "HSVabsdiff"):
    img1 = cv2.cvtColor(a, cv2.COLOR_BGR2HSV) ## Inutile pour d'autre méthode mais obligatoire dans cet endroit

## afin d'initialiser la boucle pour la méthode HSVabsdiff
#  a,imageA = maskandblur(imageA,landscapeMask,blurFact) for image with lanscapemask

## Folder containing the images of differences
subImagesDir1 = c3poFolder+"timeLapsePhotos/diff_subImages/"
if not os.path.exists(subImagesDir1):
        os.makedirs(subImagesDir1)

## Folder containing diff between A and B besides the drawcontour image
subImagesDir2 = c3poFolder+"timeLapsePhotos/diffdraw/"
if not os.path.exists(subImagesDir2):
        os.makedirs(subImagesDir2)

ShouldItStop()
time.sleep(waitTime)
while 1==1:
    imageB, timeStamp2 = CapturePi()
    name2 = subImagesDir+"image_"+timeStamp2+".JPG"
    cv2.imwrite(name2,imageB)
    print("Picture",name2,"saved.")
            
    exec(open(signalFile).read())
    if(stop):
        print("Boucle.py received stop signal from signal file. Stopping...")
        break
    elif(use_diff=="None"):
        # picture already saved, nothing else needed           
        print("No treatment;")
    else:
        b=cv2.GaussianBlur(imageB,(blurFact,blurFact),sigmaX=0)
        # landscapeMask = cv2.cvtColor(cv2.imread(args["mask"]),cv2.COLOR_BGR2GRAY)
        # b,imageB = maskandblur(imageB,landscapeMask,blurFact) for image with lanscapemask

        exec(open(signalFile).read())
        if(stop):
            print("Boucle.py received stop signal from signal file. Stopping...")
            break

        else:
            print("Starting ", use_diff)
            if(use_diff=="ssim"):
                (score, diff) = compare_ssim(a, b, full=True,multichannel=True)
                diff = (diff * 255).astype("uint8")
                blur = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#                    cv2.imwrite("ablur.jpg",blur)
                thresh = cv2.threshold(src=blur, thresh= 210,
                                       maxval=255,type=cv2.THRESH_BINARY_INV)[1]

            elif(use_diff=="HSVabsdiff"):

                img2 = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
                absDiff2 = cv2.absdiff(img1, img2)

                diff = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(src=absDiff2[:,:,0], thresh=40,
                              maxval=255,type=cv2.THRESH_BINARY)[1]
                img1 = img2
            elif(use_diff=="diff"):
                diff1 = a-b # pareil que (b,a)
#                diff2 = b-a # pareil que (b,a)
                # cv2.imwrite("adiffb.jpg",absDiff2)

                diff = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(src=diff, thresh=4,
                           maxval=255,type=cv2.THRESH_BINARY)[1]
    # thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)


            elif(use_diff=="absdiff"): # ~ 20 s sur le pi3 pour l'ensemble du script
                absDiff2 = cv2.absdiff(a,b) # pareil que (b,a)
    # cv2.imwrite("adiffb.jpg",absDiff2)

                diff = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(src=diff, thresh=10,
                           maxval=255,type=cv2.THRESH_BINARY)[1]
    # thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

            else:
                print("Unknown use_diff")
        # aggregate small changes close from each others
        threshS = cv2.dilate(thresh,(3,3))
        threshS = cv2.erode(threshS,(3,3),iterations=1)
        #    cv2.imwrite(subImagesDir2+"diffsTrAgg-"+timeStamp2+".JPG",threshS)

        # defines corresponding regions of change
        cnts = cv2.findContours(threshS.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        print("Number of diffs to analyze:"+str(len(cnts))) # dispay the number of "imagettes" to be analyzed
        
        
        if(len(cnts)!=0):
            tableau = SaveDrawPi (imageA,imageB,kernel,subImagesDir1,subImagesDir2,
                                  thresh,name1,name2,cnts,maxAnalDL)
            
            if(tableau.shape[0]>0):
                with open(csvPath,'a') as T:
                    tableau.to_csv(T, header = False, index = False)

        a=b   
        imageA=imageB

    sys.stdout.flush()

    time.sleep(waitTime)
    currentTime = datetime.datetime.today()  # sert à vérifier la durée de capture 
    if(not start.time() < currentTime.time() < fin.time()): 
        
        time.sleep(duration)
#            time.sleep(3600*24 - (currentTime - demainOff).seconds - (demainOff - demainOn).seconds)
#    else:
#        time.sleep(waitTime)

cv2.destroyAllWindows()

## flag File to tell the user interface that we stopped
sF = open(flagFile,'w')
sF.write("Start")
sF.close()

print("Boucle.py stopped.")

#Finde la boucle
