#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""

face segmentation implemented and trained at:
    https://github.com/YuvalNirkin/face_segmentation

"""

import caffe
import numpy as np
from PIL import Image


class FaceSegmentor(object):
    def __init__(self):
        self.caffe_net = caffe.Net(
            '../../models/face_seg/face_seg_fcn8s_deploy.prototxt',
            '../../data/face_seg_fcn8s.caffemodel',
            caffe.TEST
        )
        self.input_shape = (3, 500, 500)

    def __call__(self, X, preprocessed=True):
        """
        forward the network

        :param X: paths to images
        """

        if not preprocessed:
            X = [self.preprocess(img) for img in X]

        # shape for input (data blob is N x C x H x W), set data
        self.caffe_net.blobs['data'].reshape(1, *in_.shape)
        self.caffe_net.blobs['data'].data[...] = in_

        # run net and take argmax for prediction
        self.caffe_net.forward()
        out = self.caffe_net.blobs['score'].data.argmax(axis=1)

        return out

    def read(self, img_path):
        """
        read an image

        returns : PIL.Image
        """

        img = Image.open(img_path)
        img = img.resize(self.input_shape[1:])
        return img

    def preprocess(self, img):
        """
        :param img: PIL.Image
        """

        img = np.array(img, dtype=np.float32)
        img = img[:,:,::-1]
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img = img.transpose((2,0,1))

        return img

    def masked_image(self, img, img_mask):
        img_mask[img_mask > 0] = 1

        return img * img_mask
