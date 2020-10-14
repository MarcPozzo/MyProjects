This folder gathers the sump up of essential scripts

For all original images where we see an animal we annote this picture and extract a smaller image containing only the animal called "annoted imagette".
In the future an image refers to the image in the original format and imagette regers to an area (from annotation or not) extract from a bigger image. 
The first amout of data comes from "anoted imagettes". Then we will generate new imagettes thanks to birds recognition system. 


#Train_Lenet : training of Lenet neural networks for 3d chanels or with an additionnal chanel for the tempory dimensions 
	on annoted images or images caught (false or trup positive in previous models)

Transfer_learning_VGG16 : load pre-trained weights of VGG16 and keep the last layer which is train with rf,lr,tree ... .

Test_Lenet_VGG : script to evaluation neural networks trainned with VGG16 or Lenet architecture directly on pictures. 

Train_evaluate_YoloV2 : train YoloV2 on images and then evaluate directly in image.