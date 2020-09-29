#!/bin/bash
# 5/10 mA shut down X (doesn't prevent to use ssh -X 
sudo service lightdm stop
# 20 mA power off screen/hdmi
sudo tvservice --off
# 50-100 mA power off usb and ethernet (also disconnect physically the ethernet cable : 10mA)
sudo /var/www/html/c3po_interface/bin/turnOffusb.sh

echo "Normal mode" > /var/www/html/c3po_interface/code/signalFiles/flagEconomy.txt

# idle, the current should be 0.165-0.18 A
# slightly more if 20s timelapse camera (same when idling, pike to 250 during 1s
