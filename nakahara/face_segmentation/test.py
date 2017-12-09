#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from PIL import Image

from face_seg import FaceSegmentor

if __name__ == '__main__':
    segmentor = FaceSegmentor()
    raw_img = segmentor.read('./test_image/Alison_Lohman_0001.jpg')
    img = segmentor.preprocess(raw_img)
    mask = segmentor.forward(np.array([img]))[0]
    masked_img = segmentor.masked_image(raw_img, mask)

    masked_img.save('./test_image/result.jpg')
