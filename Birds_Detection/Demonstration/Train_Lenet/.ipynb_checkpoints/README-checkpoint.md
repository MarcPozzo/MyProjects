# Training of tiny images of 3 chanels with Lenet Neural Network

## Introduction

- Les images entières peuvent contenir des objets de petites taille à identifier, l'objectif de cette section est d'extraire ces objets à identifier en une base de données de petites images
- L'algorithme Lenet est ensuite entraîné sur cette base, puis évalué. 


## Files description

- generate_3C_tiny_images.py : generate tiny images depending of the coordonates in Images tables make sur Tiny_images exits
- Train_your_tiny_images.ipynb: script allow you to train Lenet with your data and save neural network
- evaluate_Lenet3C_on_tiny_pic.ipynb: display the performance of Lenet corss_tab, precision, recall, ... .
- Lenet_training_functions.py contient les fonctions supports pour les scripts précédendants


Train_4th_Chanel_Pic : in this folder we add a 4th chanel to the image and train it, we will explain it soon

## Requirement

- Les objets doivent être annotés dans un fichier csv selon leur classe et leur emplacement dans l'image. Se référer au fichier csv pour voir les informations que la DataFrame doit contenir.
- Vos images se trouvent dans un fichier à l'exterieur du répertoire s'appellant data_pic
- libraries : X,X


## Instruction
- type python3 generate_3C_tiny_images.py to generate tiny images
- then train your tiny images with Train_your_tiny_images_in_progress.ipynb
- finaly evaluate the performances of your neural network with evaluate_Lenet3C_on_tiny_pic

## Note
- please feel free to change any parameters as your conviniance.