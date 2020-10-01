#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:22:59 2020

@author: marcpozzo
"""

#Import library
from shapely.geometry import Polygon


#This script calculates the proportion area of square amoug a list of sqaure

#Parameters
polygon_to_test1 = Polygon([(0, 0), (0, 0.5), (1, 0.5), (1, 0)])
polygon_to_test2 = Polygon([(0, 0), (0, 0.5),(0.5, 0.5), (0, 0.5) ])
polygon_to_test=[polygon_to_test1,polygon_to_test1]

polygon_ref_1 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
polygon_ref_2 = Polygon([(0, 0), (0, 3), (2, 3), (2, 0)])


nombre_imagette_oiseau=0
for i in polygon_to_test:
    intersection_1=polygon_ref_1.intersection(i)
    intersection_2=polygon_ref_2.intersection(i)
    proportion_1=intersection_1.area/polygon_ref_1.area
    proportion_2=intersection_2.area/polygon_ref_2.area

    max_proportion_test1=max(proportion_1,proportion_2)

    if (max_proportion_test1>0.5) :
        nombre_imagette_oiseau+=1
        


print(nombre_imagette_oiseau)