#!/bin/sh

sudo apt-get update
sudo apt-get install wget
cd /usr/local/src/
sudo wget --no-check-certificate https://raw.githubusercontent.com/milq/milq/master/scripts/bash/install-opencv.sh
sudo chmod +x install-opencv.sh
sudo ./install-opencv.sh
curl -kL https://bootstrap.pypa.io/get-pip.py | sudo python3
sudo apt-get install caffe-cpu

# optional but frequently used
sudo apt-get install git
sudo apt-get install tmux
