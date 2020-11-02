This folder gathers models train with Lenet architectures.

- To generate tiny images please create repository in the same foler as data_pic and type  to generate image:
- python3 generate_3C_tiny_images.py
- generate_3C_tiny_images : generate tiny images depending of the coordonates in Images tables make sur Tiny_images exits
- Train_your_tiny_images_in_progress: script allow you to train Lenet with your data and save neural network
- evaluate_Lenet3C_on_tiny_pic: display the performance of Lenet corss_tab, precision, recall, ... .


Lenet_train: script of the training of Lenet containing a generator (zoom, preprocessing function,...)
Train_4th_Chanel_Pic : in this folder we add a 4th chanel to the image