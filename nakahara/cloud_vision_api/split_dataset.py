#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import shutil
import sys

from sklearn.cross_validation import train_test_split


sys.path.append('../cyclegan')
from config import datasets_basepath


def split_dataset(label, master, test_size=0.1, seed=None, threshold_up=0.7, threshold_down=0.3, file_origin=os.path.join(datasets_basepath, 'download_all')):
    pos_files = master[master[label] >= threshold_up].index
    pos_labels = [1]*len(pos_files)
    neg_files = master[master[label] <= threshold_down].index
    neg_labels = [1]*len(neg_files)

    splitted_dataset_root = os.path.join(datasets_basepath, label)
    if os.path.exists(splitted_dataset_root):
        shutil.rmtree(splitted_dataset_root)

    os.mkdir(splitted_dataset_root)
    for files, labels, kind in [(neg_files, neg_labels, 'A'), (pos_files, pos_labels, 'B')]:
        os.mkdir(os.path.join(splitted_dataset_root, 'train'+kind))
        os.mkdir(os.path.join(splitted_dataset_root, 'test'+kind))
        train_X, test_X, _, _ = train_test_split(files, labels, test_size=test_size, random_state=seed)
        for filename in train_X:
            shutil.copy2(os.path.join(file_origin, filename), os.path.join(splitted_dataset_root, 'train'+kind, filename))

        for filename in test_X:
            shutil.copy2(os.path.join(file_origin, filename), os.path.join(splitted_dataset_root, 'test'+kind, filename))

if __name__ == '__main__':
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--master', default='goog_cloud_vision_labels.csv')
    parser.add_argument('--test_size', default=0.1)
    parser.add_argument('--random_state', default=1234)
    parser.add_argument('--label', type=str, help='a property to split data with')

    args = parser.parse_args()
    print(args)
    master = pd.read_csv(args.master, index_col=0)

    split_dataset(args.label, master, args.test_size, args.random_state)