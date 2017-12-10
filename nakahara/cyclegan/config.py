#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import pandas as pd

datasets_basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/')
#dataset_master = pd.read_csv(os.path.join(datasets_basepath, 'hotpepper_label_master.csv'))
#
#datasets_paths = {}
#dirs = os.listdir(datasets_basepath)
#
#for dataset_name in dirs:
#    datasets_paths[dataset_name] = os.path.join(datasets_basepath, dataset_name)
#
#__all__ = (
#    datasets_basepath,
#    datasets_master,
#    datasets_paths,
#    )