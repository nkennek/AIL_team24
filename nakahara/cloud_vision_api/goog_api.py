#!/usr/bin/env python
# -*- coding:utf-8 -*-

API_KEY = "AIzaSyCsiXklTqf0_dUZr-o2jkPKYTJlYyTIPpw"
GOOGLE_CLOUD_VISION_API_URL = 'https://vision.googleapis.com/v1/images:annotate?key='

import base64
import json

import requests
import urllib


def goog_cloud_vision(img_path, kind='LABEL_DETECTION'):
    """
    :param img_path: path to img
    """
    request_types = (
        "LABEL_DETECTION",
        "TEXT_DETECTION",
        "FACE_DETECTION",
        "LANDMARK_DETECTION",
        "LOGO_DETECTION",
        "IMAGE_PROPERTIES"
    )
    if kind not in request_types:
        raise ValueError("invalid request type: {}".format(kind))

    with open(img_path, 'rb') as img:
        base64_image = base64.b64encode(img.read()).decode('utf-8')

    data = {
        'requests': [{
            'image': {
                'content': base64_image,
            },
            'features': [{
                'type': kind,
                'maxResults': 100,
            }]
        }]
    }
    header = {'Content-Type': 'application/json'}

    # リクエスト送信
    response = requests.post(GOOGLE_CLOUD_VISION_API_URL + API_KEY,  json.dumps(data), header)

    if response.status_code != 200:
        print("err: \n{}".format(response.json()))

    return response.status_code, response.json()