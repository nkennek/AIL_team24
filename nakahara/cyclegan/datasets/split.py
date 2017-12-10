#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""

Google Cloud Vision APIの診断結果からデータセットを限定・分割する

"""

import sys

from sklearn.datasets import train_test_split

sys.path.append('..')

import config

def bronze_or_not():
    # label not made up yet
    pass