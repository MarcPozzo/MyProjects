#!/bin/bash
# cf bin/economyMode
sudo /var/www/html/c3po_interface/bin/turnOnusb.sh
sudo tvservice -p
sudo service lightdm start
echo "Economy mode" > /var/www/html/c3po_interface/code/signalFiles/flagEconomy.txt
