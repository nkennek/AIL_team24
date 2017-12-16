 # -*- coding: utf-8 -*-
# https://qiita.com/bluemooninc/items/075a658f0d2c7ac62efc

# python2 is mandatory
import requests
import json
import base64
import os
import argparse


GOOGLE_CLOUD_VISION_API_URL = 'https://vision.googleapis.com/v1/images:annotate?key='
API_KEY = 'YOUR-GOOGLE-CLOUD-VISION-API-KEY'

def goog_cloud_vison (image_content):
    api_url = GOOGLE_CLOUD_VISION_API_URL + API_KEY
    req_body = json.dumps({
        'requests': [{
            'image': {
                'content': image_content
            },
            'features': [
                {
                'type': 'FACE_DETECTION'
                #'maxResults': 10,
                },
                {
                'type': 'LANDMARK_DETECTION'
                #'maxResults': 10,
                },
                {
                'type': 'LABEL_DETECTION'
                #'maxResults': 10,
                }
            ]
        }]
    })
    res = requests.post(api_url, data=req_body)
    return res.json()

def img_to_base64(filepath):
    with open(filepath, 'rb') as img:
        img_byte = img.read()
    return base64.b64encode(img_byte)



##
## main
##


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect face by google cloud vision api')
    args = parser.add_argument('imgfiles', nargs='+', help='image files')
    #args = parser.add_argument('--debug', action='store_true', help='debug flag')
    #args = parser.add_argument('--mkdir', action='store_true', help='mkdir outfile if not exist')
    #args = parser.add_argument('--overwrite', action='store_true', help='overwrite outfilefile if exist')
    args = parser.add_argument('--category', default='test')
    args = parser.parse_args()

    for file in args.imgfiles:
        img = img_to_base64(file)
        res_json = goog_cloud_vison(img)
        out = res_json['responses'][0]
        out['fileCategory'] = args.category
        out['fileBase'] = os.path.basename(file)

        print(json.dumps(out))
