#!/bin/bash
# turn off usb (saves 50-100mA), also unplug the etherne cable or 10/20mA remain
# should be run as sudo 
# doesn't shut down the wifi, only the ethernet
sudo service networking stop
echo 0 > /sys/devices/platform/soc/3f980000.usb/buspower

