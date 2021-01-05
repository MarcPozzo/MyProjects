#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:00:59 2021

@author: marcpozzo

"""
        thresh=0.5
        blurFact=17
        chanels=3
        contrast=-8
        #attention mask=True
        imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates,liste_Diff_birds,nb_oiseaux=fn.Evaluate_Lenet_prediction(Images, name_test,name_ref,CNNmodel,data_path=data_path,
                                                                                                                                                           blockSize=blockSize,thresh=0.5,
                                                                                                                                                           blurFact=17,chanels=3,contrast=-8,
                                                                                                                                                           maxAnalDL=maxAnalDL,diff_mod3C=diff_mod3C,
      


         imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates,liste_Diff_birds,nb_oiseaux=fn.Evaluate_Lenet_prediction ( Images , name_test , name_ref  , CNNmodel , maxAnalDL ,data_path , 
                                                                                                                                                                   
                                                                                                                                                                     chanels = chanels,  
                                                                                                                                                                     contrast = contrast , blockSize = blockSize  , blurFact = blurFact ,thresh = thresh,focus = focus,
                                                                                                                                                                     diff_mod3C = diff_mod3C )  
                                                                                                                                                     
        
        
        
        
        
        
        
         imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates,liste_Diff_birds,nb_oiseaux=fn.Evaluate_Lenet_prediction ( Images , name_test , name_ref  , CNNmodel , maxAnalDL ,data_path , 
                                                                                                                                                                     filtre_choice = "No_filtre" ,down_thresh = 25 ,
                                                                                                                                                                     chanels = chanels , numb_classes = 6 , mask = False , 
                                                                                                                                                                     contrast = contrast , blockSize = blockSize  , blurFact = blurFact ,seuil = 210 ,
                                                                                                                                                                     thresh_active = True , index = False ,thresh = thresh,focus = focus,
                                                                                                                                                                     diff_mod3C = diff_mod3C ,diff_mod4C = "HSV")         
        
        
        
        
        
        
        
        
        
        imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates,liste_Diff_birds,nb_oiseaux=fn.Evaluate_Lenet_prediction ( Images , name_test , name_ref  , CNNmodel , maxAnalDL ,data_path , 
                                                                                                                                                                     filtre_choice = "No_filtre" ,down_thresh = 25 ,
                                                                                                                                                                     chanels = 3 , numb_classes = 6 , mask = False , 
                                                                                                                                                                     contrast = - 5 , blockSize = 53 , blurFact = 15 ,seuil = 210 ,
                                                                                                                                                                     thresh_active = True , index = False ,thresh = 0.5,focus = "bird_prob",
                                                                                                                                                                     diff_mod3C = "light" ,diff_mod4C = "HSV")       
        
        
        
