#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project: mri-breast-tumor-segmentation
# File   : DataLoading.py
# Author : Bo Wang
# Date   : 8/23/19

import numpy as np
import csv
from sklearn.model_selection import train_test_split


# #############################################################################
# Disrupt the order of the dataset
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    import numpy
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


# #############################################################################
# Load dataset, the last column is [Target]
# Return: total dataset and all features' name
def load_data(file_path):
    temp = []
    with open(file_path) as csv_file:
        lines = csv.reader(csv_file)
        for line in lines:
            temp.append(line)
    features_name = temp[0]
    temp.remove(temp[0])
    my_data = np.array(temp, dtype=np.float64)
    data = my_data[:, :-1]
    target = my_data[:, -1]  # each sample's label
    target = target.astype(int)
    features_name = features_name[:-1]

    return data, target, features_name


# #############################################################################
# Load dataset and split into training data and testing data, the last column is [Target]
# Return: total dataset, train dataset, test dataset, and all features' name
def load_and_split_data(file_path):
    temp = []
    with open(file_path) as csv_file:
        lines = csv.reader(csv_file)
        for line in lines:
            temp.append(line)
    features_name = temp[0]
    temp.remove(temp[0])
    my_data = np.array(temp, dtype=np.float64)
    data = my_data[:, :-1]
    target = my_data[:, -1]  # each sample's label
    target = target.astype(int)
    features_name = features_name[:-1]
    # X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)
    # X_train, X_test, y_train, y_test = data[:110], data[110:], target[:110], target[110:]
    # y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    new_data, new_target = shuffle_in_unison(data, target)
    X_train, X_test, y_train, y_test = train_test_split(new_data, new_target, random_state=0)
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    return data, target, X_train, y_train, X_test, y_test, features_name
