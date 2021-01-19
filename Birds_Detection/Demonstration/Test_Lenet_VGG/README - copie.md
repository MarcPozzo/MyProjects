# Evaluate your algo in real condition

## Introduction

You have already trained your algo in the folder Train_Lenet and evaluate them in your dataset. 
But in real condition you have to extract tiny images and then analyse them, 
you will probably have a lot tiny images extracted without any targets you want to detect.

In this folder you can find the method to get the best parameters of extraction with the best ratio of targets caught 
and (empty) tiny images extracted. 

## Files description

- Extraction_Evaluation.py : Help you to find the best combinaison of blocksize,blurFact,contrast, ...
- Evaluate_Lenet_on_pic.py : You can evaluate different neural networks trained and fix a threshold. 
- functions_Lenet_VGG.py : This is a module with functions usefull in the two others scripts.

## Requirement

- You should already have trained your neural networks and follow the instruction of the Train_Lenet_
- libraries to load : pandas=0.25.3, cv2=3.4.2


## Instruction

- Open the Extraction_Evaluation.py file, change the parameters as your connivence to the best combinaison.
- Open the Evaluate_Lenet_on_pic.py file, use the parameters find in Extraction_Evaluation.py 
change the parameters (thresh, precise, neural network) as your connivence to maximise to TP and minimize the FP



