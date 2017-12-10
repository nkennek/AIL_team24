#!/usr/bin/env python
# -*- coding:utf-8 -*-

API_KEY = "AIzaSyCsiXklTqf0_dUZr-o2jkPKYTJlYyTIPpw"
GOOGLE_CLOUD_VISION_API_URL = 'https://vision.googleapis.com/v1/images:annotate?key='

import base64
import json
import os

import pandas as pd
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


def parse_result(res, name=None):
    descriptions = res[1]['responses'][0]['labelAnnotations']
    descriptions_pd = pd.DataFrame(descriptions).set_index('description').score
    if name is not None:
        descriptions_pd = descriptions_pd.rename(name)
    return descriptions_pd


if __name__ == '__main__':
    import time
    from tqdm import tqdm

    data_dir = '../../data/download_all/'
    files = os.listdir(data_dir)

    result_all = None

    for f in tqdm(files):
        try:
            filepath = data_dir + f
            response = goog_cloud_vision(filepath)
            desc = parse_result(response, name=f)
            if result_all is None:
                result_all = desc
                continue

            result_all = pd.concat([result_all, desc], axis = 1)
            time.sleep(0.5)
        except KeyboardInterrupt:
            break

    result_all = result_all.T.fillna(0)
    result_all.to_csv('goog_cloud_vision_labels.csv')