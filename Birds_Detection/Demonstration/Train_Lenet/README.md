# Training of tiny images of 3 chanels with Lenet Neural Network

## Introduction

- Whole images may contain small objects to be identified, the objective of this section is to extract these objects to be identified into a database of small images
- The Lenet algorithm is then trained on this basis, then evaluated.

## Files description

- generate_3C_tiny_images.py : generate tiny images depending of the coordonates in Images tables make sur Tiny_images exits
- Train_your_tiny_images.ipynb : script allow you to train Lenet with your data and save neural network as drop_out.50
- evaluate_Lenet3C_on_tiny_pic.ipynb : display the performance of Lenet corss_tab, precision, recall, ... .
- Lenet_training_functions.py : contains support functions for previous scripts
- drop_out.50 is a neural network trained in Train_your_tiny_images.ipynb 
	and Evaluate in evaluate_Lenet3C_on_tiny_pic.ipynb (after evaluating neural network please put it in ../../Materiels/Models)

Train_4th_Chanel_Pic : in this folder we add a 4th chanel to the image and train it.

## Requirement

- The objects must be annotated in a csv file according to their class and their location in the image. 
- Refer to the csv file to see the information that the DataFrame should contain.
- Your images are in a file outside the directory called 'Pic_dataset/' this folder must be at '../../../../Pic_dataset/'
- libraries to load : pandas=0.25.3, cv2=3.4.2


## Instruction

- Type python3 generate_3C_tiny_images.py to generate tiny images eventually with a zoom or a dezoom
- Then train your tiny images with Train_your_tiny_images.ipynb file
- Finaly evaluate the performances of your neural network with evaluate_Lenet3C_on_tiny_pic.ipynb file

## Note

- Please feel free to change any parameters as your conviniance.