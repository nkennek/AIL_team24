#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
import sys

import cv2

from face_seg import FaceSegmentor


OUT = './results/'
OUT_SUBDIR = ['raw', 'mask', 'masked']

if not os.path.exists(OUT):
    os.mkdir(OUT)
    for subdir in OUT_SUBDIR:
        os.mkdir(OUT+subdir)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='execute batch segmentation')
    argparser.add_argument('-f', type=str)
    args = argparser.parse_args()

    file_dir = args.f
    filenames = os.listdir(file_dir)
    filepaths = [os.path.join(file_dir, f) for f in filenames]

    segmentor = FaceSegmentor()
    imgs, masks, masked_images = segmentor(filepaths)

    for filename, img, mask, masked_image in zip(filenames, imgs, masks, masked_images):
        cv2.imwrite(os.path.join(OUT, OUT_SUBDIR[0], filename), img)
        cv2.imwrite(os.path.join(OUT, OUT_SUBDIR[1], filename), mask)
        cv2.imwrite(os.path.join(OUT, OUT_SUBDIR[2], filename), masked_image)
