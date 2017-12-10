FROM nvidia/cuda:7.5-cudnn5-devel

MAINTAINER Kenichi Nakahara <nakahara@akg.t.u-tokyo.ac.jp>

RUN apt-get -y update & apt-get -y upgrade
RUN apt-get install \
    python3-pip curl git cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev\
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
RUN alias python='python3'
RUN alias pip='pip3'

COPY . /home/AIL_team24
RUN pip install -U pip & pip install -r /home/AIL_team24/requirements.txt