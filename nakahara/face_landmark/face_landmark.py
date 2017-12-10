#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""

https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

"""

import copy
import os

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models/face_landmark/shape_predictor_68_face_landmarks.dat')


class FaceNotFoundError(Exception):
    def __init__(self):
        pass


class LandmarkDetector(object):
    def __init__(self, model=model_path):
        # initialize detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    def __call__(self, img):
        """
        analyze image

        :param img: filepath or cv2.image
        :returns face rectangle (x, y, w, h), and landmark points [(x1, y1), (x2, y2), ... ]
        """

        if isinstance(img, str) and os.path.isfile(img):
            img = cv2.imread(img)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect faces in grayscale image
        rects = self.detector(img_gray, 1)
        if len(rects) == 0:
            raise FaceNotFoundError()
        if len(rects) > 0:
            #keep largest one
            largest_rect = None
            largest_size = 0
            for idx, rect in enumerate(rects):
                _, _, w, h = self._cast_rectangle(rect)
                size = w*h
                if size > largest_size:
                    largest_rect = rect

            rect = largest_rect
        else:
            rect = rects[0]

        shape = self.predictor(img_gray, rect)
        np_rect = self._cast_rectangle(rect)
        np_landmarks = self._cast_shape(shape)
        return np_rect, np_landmarks

    def plot(self, img):
        (x, y, w, h), landmarks = self(img)
        img = cv2.imread(img)
        img_campus = copy.copy(img)
        cv2.rectangle(img_campus, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for (x, y) in landmarks:
            cv2.circle(img_campus, (x, y), 1, (0, 0, 255), -1)

        plt.imshow(img_campus)
        return img, img_campus

    def _cast_shape(self, shape, dtype=np.int):
        """
        :param shape: dlib.dlib.full_object_detection
        """

    	# initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

    	# loop over the 68 facial landmarks and convert them
    	# to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

    	# return the list of (x, y)-coordinates
        return coords

    def _cast_rectangle(self, rect, dtype=np.int):
        """
        :param rect: dlib.dlib.rectangle
        """

        # take a bounding predicted by dlib and convert it
    	# to the format (x, y, w, h) as we would normally do
    	# with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)