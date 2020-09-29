#!/bin/bash
#===============================================================================
#
#          FILE:  stillImage.sh
# 
#         USAGE:  ./stillImage.sh 
# 
#   DESCRIPTION:  just get a simple image from the cam, and make it available for the interface
# 
#       OPTIONS:  ---
#  REQUIREMENTS:  ---
#          BUGS:  ---
#         NOTES:  ---
#        AUTHOR:   (), 
#       COMPANY:  
#       VERSION:  1.0
#       CREATED:  16/03/2019 04:55:03 CET
#      REVISION:  ---
#===============================================================================

pkill motion
while pkill -0 motion; do 
    sleep 0.1
done

if (( $(uname -a | grep piFird | wc -l) == 1 )); then 
	# on pi
	# see -ts to add timestamp
	raspistill -t 1 -rot 180 -o photos/stillCurrent.jpeg
else
	# on ubuntu 
	streamer -s 1280x720 -o photos/stillCurrent.jpeg
fi

