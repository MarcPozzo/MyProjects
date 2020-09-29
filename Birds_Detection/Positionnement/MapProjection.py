# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:12:55 2020

@author: mark
"""
import numpy as np
import cv2
import json
from os import chdir

#A set of points to give the function to project
usertest=[[[155],[202]],[[303],[104]],[[254],[345]]]





#For the second and third function
camera_to_map_convertor_file="2020-01-09_16-57-41_GMT_positions.json"
camera_to_map_convertor_file="/mnt/VegaSlowDataDisk/c3po_interface/code/positions/reperes.json"
camera_to_map_convertor_path="/mnt/VegaSlowDataDisk/c3po/codePython/"
chdir(camera_to_map_convertor_path)


#Only 1 function



#Start of Function asking for filename
def Projection(userpoints): 
    print('Enter JSON file')                                   
    
#initializing arrays for storing coordinates    
    mapcoord= []
    shotcoord= []

#Commmented out print commands were for checking that data were being read correctly.                                                                               
#This digs into the json to seperate out the map and shot data.
    with open(input()) as json_file:
        data = json.load(json_file)                                                                                                             
        mapdata = (data['map'])
        mapmarkers = (mapdata['markers'])
        shotdata= (data['shot'])
        shotmarkers=(shotdata['markers'])
        #print (mapmarkers)
       
 #This extracts coordinates from json file and puts them in their respective arrays.       
        for p in mapmarkers:
            #print ('Id: ' + p['id'])
            coords = p['coord']
            for key in coords.items():
                #print (key[0],':', key[1])
                coord=[(key[1])]
                mapcoord.append(coord)
        #print(mapcoord)
        
        for p in shotmarkers:
            coords = p['coord']
            for key in coords.items():
                #print (key[0],':', key[1])
                coord=[(key[1])]
                shotcoord.append(coord)
        #print(shotcoord)
    
# This pairs the lat and lng of the map coordinates
    map_pair=[]
    
    for i in range(0,len(mapcoord),2):
        test=[]
        test.append(mapcoord[i])
        test.append(mapcoord[i+1])
        map_pair.append(test)
    
# This pairs the lat and lng of the shot coordinates    
    shot_pair=[]
    for i in range(0,len(shotcoord),2):
        test=[]
        test.append(shotcoord[i])
        test.append(shotcoord[i+1])
        shot_pair.append(test)
            
#This creates the projection matrix for the transformation       
    projection =  cv2.findHomography(np.float32([shot_pair[:4]]).reshape(4, 1, -1), np.float32([map_pair[:4]]).reshape(4, 1, -1))[0]

#This creates a test projection using the fifth paired map and shot point   
    Testpoint = cv2.perspectiveTransform(np.float32([shot_pair[4]]).reshape(1, 1, -1), projection)
 
#This shows the user the test and the actual point it should map to, so they may check for error  
    print('The following is an actual map point and its projected map point from the shot.\nEnsure that they are acceptable.')
    print(np.float32([map_pair[4]]).reshape(1, 1, -1))
    print(Testpoint)
    
#This projects the user coordinates
    projected_coord=[]
    for i in userpoints:
        user_projection = cv2.perspectiveTransform(np.float32([i]).reshape(1, 1, -1), projection)
        user_projection=user_projection[0]
        user_projection=user_projection[0]
        #print(user_projection)
        projected_coord.extend(user_projection)
        
#This repairs the projected coordinates
    projection_pairs=[]
    for i in range(0,len(projected_coord),2):
        test=[]
        test.append(projected_coord[i])
        test.append(projected_coord[i+1])
        projection_pairs.append(test)    
   
    
    return(projection_pairs)

#Running function
test=Projection(usertest)

#This shows the user the projected map points they inserted    
print('These are the projected map points from your points:')
print(test)

#cd /mnt/VegaSlowDataDisk/c3po/codePython/preSelectModel.R
#preSelectModel.R
#/mnt/VegaSlowDataDisk/c3po/codePython/preSelectModel.R





#Now separate the function in 2 parts








#Functions definitions


#Create a projection matrix to learn how to convert camera coordonates into map coordonates
def get_Projection(camera_to_map_convertor_file): 
                                    
#initializing arrays for storing coordinates    
    mapcoord= []
    shotcoord= []

#Commmented out print commands were for checking that data were being read correctly.                                                                               
#This digs into the json to seperate out the map and shot data.
    with open(camera_to_map_convertor_file) as json_file:
        data = json.load(json_file)                                                                                                             
        mapdata = (data['map'])
        mapmarkers = (mapdata['markers'])
        shotdata= (data['shot'])
        shotmarkers=(shotdata['markers'])
        #print (mapmarkers)
       
 #This extracts coordinates from json file and puts them in their respective arrays.       
        for p in mapmarkers:
            #print ('Id: ' + p['id'])
            coords = p['coord']
            for key in coords.items():
                #print (key[0],':', key[1])
                coord=[(key[1])]
                mapcoord.append(coord)
        #print(mapcoord)
        
        for p in shotmarkers:
            coords = p['coord']
            for key in coords.items():
                #print (key[0],':', key[1])
                coord=[(key[1])]
                shotcoord.append(coord)
        #print(shotcoord)
    
# This pairs the lat and lng of the map coordinates
    map_pair=[]
    
    for i in range(0,len(mapcoord),2):
        test=[]
        test.append(mapcoord[i])
        test.append(mapcoord[i+1])
        map_pair.append(test)
    
# This pairs the lat and lng of the shot coordinates    
    shot_pair=[]
    for i in range(0,len(shotcoord),2):
        test=[]
        test.append(shotcoord[i])
        test.append(shotcoord[i+1])
        shot_pair.append(test)
            
#This creates the projection matrix for the transformation       
    projection =  cv2.findHomography(np.float32([shot_pair[:4]]).reshape(4, 1, -1), np.float32([map_pair[:4]]).reshape(4, 1, -1))[0]

    return  projection
    
    
    
#Apply the projection  to a set of points from the camera coordonate to get the map coordonates
def apply_projection(userpoints,projection):
#This projects the user coordinates
    projected_coord=[]
    for i in userpoints:
        user_projection = cv2.perspectiveTransform(np.float32([i]).reshape(1, 1, -1), projection)
        user_projection=user_projection[0]
        user_projection=user_projection[0]
        #print(user_projection)
        projected_coord.extend(user_projection)
        
#This repairs the projected coordinates
    projection_pairs=[]
    for i in range(0,len(projected_coord),2):
        test=[]
        test.append(projected_coord[i])
        test.append(projected_coord[i+1])
        projection_pairs.append(test)    
   
    
    return(projection_pairs)

#Running function
projection=get_Projection(camera_to_map_convertor_file)
test=apply_projection(usertest,projection)


#This shows the user the projected map points they inserted    
print('These are the projected map points from your points:')
print(test)








