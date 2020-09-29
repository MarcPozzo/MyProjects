#!/bin/bash
# turn on usb after ~/bin/turnOffusb.sh turned it off
# should be run as sudo 
echo 1 > /sys/devices/platform/soc/3f980000.usb/buspower
sudo service networking start

