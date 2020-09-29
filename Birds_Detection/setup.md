# this is largely outdated since we installed on the pi
# please check diffetplus/additionalInstall.md

# Use 
#=============
# to launch the php-web server
# in this folder
sudo docker-compose up
# (be patient, wait for "ready to handle connections")



# then can access the site:
http://localhost:8080/
# and the code is in code/

# don't forget to also launch the python bash server
python3 code/bashServer.py



# Useful commands
#=================
# to list active containers
sudo docker container ls 

# to stop a container 
sudo docker stop <first number in ls above> 

# to ssh into a container
sudo docker exec -it <first number in ls above> /bin/bash

# then simply send post to the python server from anywhere (risk de sécurité majeure si exposé)
curl -d '{"cmd":"touch ~/truc"}' 172.17.0.1:8082
# need to think a bit more about the general architecture to avoid at least issues on the development machines
#=> this can be handled directly by the javascript => simpler, just restrict the commands than can be launched
#   need to start the bashServer.py at startup


# Installation
#==============
# install docker
curl -fsSL get.docker.com -o get-docker.sh && sh get-docker.sh
# then check 
sudo docker run hello-world

git clone https://github.com/mikechernev/dockerised-php.git
# to install docker-compose : 
sudo curl -L "https://github.com/docker/compose/releases/download/1.23.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# more secure : use ssh, add ssh on the container, save it...
ssh -l cbarbu 172.17.0.1 "touch ~/thing"
# but need specific user than can only execute some commands and set ssh keys
#=> needs to be handled by php

# motion 
sudo apt install motion
# motion setup : careful to change the /home/cbarbu/ in ./motion.conf
# set up /dev/video0 pour accès par motion
sudo modprobe bcm2835-v4l2
# to make it permanent, add 
bcm2835
# to /etc/modules

# auto launch bashServer.py at boot
# add to /etc/rc.local
mkdir /var/www/html/c3po_interface/log
python3 /var/www/html/c3po_interface/code/bashServer.py > /var/www/html/c3po_interface/log/bashServer.log &


