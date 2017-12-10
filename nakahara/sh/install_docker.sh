#!/bin/sh

# install docker
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get -y install docker-ce

# install nvidia-docker
wget https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i nvidia-docker_1.0.1-1_amd64.deb
sudo docker pull nvidia/cuda:8.0-cudnn7-runtime

# test command
sudo nvidia-docker run --rm nvidia/cuda:8.0-cudnn7-runtime nvidia-smi