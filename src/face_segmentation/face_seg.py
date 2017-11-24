#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""

face segmentation implemented and trained at:
    https://github.com/YuvalNirkin/face_segmentation

"""

from datetime import datetime as dt

import caffe
import cv2
import numpy as np


class FaceSegmentor(object):
    def __init__(self):
        self.caffe_net = caffe.Net(
            '../../models/face_seg/face_seg_fcn8s_deploy.prototxt',
            '../../models/face_seg/face_seg_fcn8s.caffemodel',
            caffe.TEST
        )
        self.input_shape = (3, 500, 500)

    def __call__(self, X):
        """
        :param X: paths to images or array of images
        """

        print('{}:starting predict'.format(dt.now()))

        if isinstance(X, list):
            X = np.array(X)

        if X.ndim == 1: # paths information
            X = np.array([self.read(path) for path in X])

        elif X.ndim == 3: # if single img
            X = np.array([X])

        preprocessed_X = [self.preprocess(img) for img in X]

        print('{}:preprocess & read finished'.format(dt.now()))
        masks = self.forward(preprocessed_X)
        print('{}:forward finished'.format(dt.now()))
        masked_images = np.array(
            [self.masked_image(img, mask) for img, mask in zip(X, masks)]
        )

        return X, masks, masked_images

    def forward(self, X, preprocessed=True):
        """
        forward the network

        :param X: array of (preprocessed) image
        """

        if not preprocessed:
            X = [self.preprocess(img) for img in X]

        # shape for input (data blob is N x C x H x W), set data
        self.caffe_net.blobs['data'].reshape(len(X), *self.input_shape)
        self.caffe_net.blobs['data'].data[...] = X

        # run net and take argmax for prediction
        self.caffe_net.forward()
        out = self.caffe_net.blobs['score'].data.argmax(axis=1)

        return out

    def read(self, img_path):
        """
        read an image

        returns : read image
        """

        img = cv2.imread(img_path)
        #resize while keeping aspect
        max_length = max(img.shape)
        center = int(max_length/2)
        half_height = int(img.shape[0]/2)
        half_width = int(img.shape[1]/2)
        anchor=[
            [center-half_height, center-half_height+img.shape[0]],
            [center-half_width, center-half_width+img.shape[1]],
        ]

        canvas = np.zeros(shape=(max_length, max_length, 3))
        canvas[anchor[0][0]:anchor[0][1], anchor[1][0]:anchor[1][1]] = img
        img = cv2.resize(canvas, self.input_shape[1:])
        return img

    def preprocess(self, img):
        """
        :param img: numpy array

        returns: network feedable numpy array
        """

        img = np.array(img, dtype=np.float32)
        img = img[:,:,::-1]
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img = img.transpose((2,0,1))

        return img

    def masked_image(self, img, img_mask):
        img_mask[img_mask > 0] = 1

        masked = (img * img_mask[:,:,np.newaxis]).astype(np.uint8)
        return masked
