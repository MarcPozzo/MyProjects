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
usertest=[  [[155],[202]],[[303],[104]],[[254],[345]]   ]




camera_to_map_convertor_path="/home/marcpozzo/Desktop/c3po_interface_mark/Materiels/positions/"

chdir(camera_to_map_convertor_path)

camera_to_map_convertor_file=camera_to_map_convertor_path+"fichier_reference.json"

#Only 1 function
dic={"id":"1","coord":{"lat":14,"lng":16}},{"id":"2","coord":{"lat":138,"lng":11}},{"id":"3","coord":{"lat":288,"lng":10}},{"id":"4","coord":{"lat":20,"lng":89}},{"id":"5","coord":{"lat":148,"lng":104}},{"id":"6","coord":{"lat":304,"lng":94}},{"id":"7","coord":{"lat":17,"lng":185}},{"id":"8","coord":{"lat":154,"lng":170}},{"id":"9","coord":{"lat":317,"lng":185}},{"id":"10","coord":{"lat":23,"lng":281}},{"id":"11","coord":{"lat":164,"lng":301}},{"id":"C","coord":{"lat":234,"lng":331}},{"id":"12","coord":{"lat":324,"lng":302}},{"id":"C","coord":{"lat":90,"lng":286}},{"id":"13","coord":{"lat":30,"lng":384}},{"id":"14","coord":{"lat":174,"lng":392}},{"id":"15","coord":{"lat":327,"lng":396}},{"id":"16","coord":{"lat":28,"lng":524}},{"id":"17","coord":{"lat":174,"lng":523}},{"id":"18","coord":{"lat":323,"lng":528}}

dic={"id":"1","coord":{"lat":131,"lng":416.7356403567942}},{"id":"C","coord":{"lat":192,"lng":376.74671301200704}},{"id":"2","coord":{"lat":193,"lng":384.37128038315115}},{"id":"3","coord":{"lat":203,"lng":422.360761360699}},{"id":"4","coord":{"lat":141.5,"lng":371.87474058790514}},{"id":"5","coord":{"lat":111.5,"lng":376.37349491419366}},{"id":"6","coord":{"lat":101.5,"lng":420.3613149934596}},{"id":"7","coord":{"lat":171.5,"lng":441.35550184947283}},{"id":"8","coord":{"lat":171.5,"lng":405.86532883097425}},{"id":"C","coord":{"lat":154.5,"lng":471.84705894987314}},{"id":"9","coord":{"lat":121.5,"lng":471.84705894987314}}


liste_coords=[]
for point in range(len(dic)):
    Couple=[]
    lat=[dic[point]["coord"]["lat"]]
    long=[dic[point]["coord"]["lng"]]
    Couple.append(lat)
    Couple.append(long)
    liste_coords.append(Couple)
    
    
usertest=liste_coords
    
chdir("/mnt/VegaSlowDataDisk/c3po_interface_mark/Photo_test/")




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


{"lat":293.25,"lng":203}},{"id":"2","coord":{"lat":347.25,"lng":353}},{"id":"3","coord":{"lat":243.25,"lng":450




user_test2=[ [ [293.25],[293.25] ] ]



"""
dic={"id":"1","coord":{"lat":14,"lng":16}},{"id":"2","coord":{"lat":138,"lng":11}},{"id":"3","coord":{"lat":288,"lng":10}},{"id":"4","coord":{"lat":20,"lng":89}},{"id":"5","coord":{"lat":148,"lng":104}},{"id":"6","coord":{"lat":304,"lng":94}},{"id":"7","coord":{"lat":17,"lng":185}},{"id":"8","coord":{"lat":154,"lng":170}},{"id":"9","coord":{"lat":317,"lng":185}},{"id":"10","coord":{"lat":23,"lng":281}},{"id":"11","coord":{"lat":164,"lng":301}},{"id":"C","coord":{"lat":234,"lng":331}},{"id":"12","coord":{"lat":324,"lng":302}},{"id":"C","coord":{"lat":90,"lng":286}},{"id":"13","coord":{"lat":30,"lng":384}},{"id":"14","coord":{"lat":174,"lng":392}},{"id":"15","coord":{"lat":327,"lng":396}},{"id":"16","coord":{"lat":28,"lng":524}},{"id":"17","coord":{"lat":174,"lng":523}},{"id":"18","coord":{"lat":323,"lng":528}}


[[-92.71159, 112.8649],
 [15.9440365, 92.17985],
 [130.15607, 75.3847],
 [-64.500114, 181.80005],
 [37.037773, 174.97835],
 [140.21361, 146.76538],
 [-43.28771, 254.70601],
 [48.34069, 222.30563],
 [146.49055, 208.84937],
 [-20.827309, 310.54086],
 [65.024536, 296.88455],
 [103.47776, 300.50183],
 [148.13463, 273.2022],
 [21.149252, 301.44284],
 [-1.3040439, 358.18854],
 [75.65922, 337.10266],
 [148.03195, 315.42844],
 [13.875225, 410.8973],
 [81.677574, 386.16022],
 [144.62209, 365.1894]]


On va faire le lien avec le point à -1
on fait des point toutes autour pour trouver la direction de l'autre axe. En gros en trouve un autre point avec 0 mais avec le deuxième axe différent
"""
