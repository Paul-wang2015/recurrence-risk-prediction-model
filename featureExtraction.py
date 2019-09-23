#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project: mri-breast-tumor-segmentation
# File   : featureExtraction.py
# Author : Bo Wang
# Date   : 7/11/19

import os
import radiomics
import radiomics.featureextractor as FEE
import pandas as pd
import csv
import SimpleITK as sitk

# Initial the data source settings
base_path_DYN1 = r'./Data/all_nrrd_list/stability_DYN1'
base_path_mask = r'./Data/all_nrrd_list/mask3'
base_path_para = r'./Data'
files_DYN1 = os.listdir(base_path_DYN1)
files_DYN1.sort()
files_mask = os.listdir(base_path_mask)
files_mask.sort()
para_name = r'/Params.yaml'
para_path = base_path_para + para_name


# Function to extract feature for one sample
def one_featureExtraction(ori_path, lab_path, para_path):
    # Initial feature extractor by using para file
    extractor = FEE.RadiomicsFeatureExtractor(para_path)
    # Execute the extractor
    # Features extraction: results is returned in a Python ordered dictionary
    current_result = extractor.execute(ori_path, lab_path)

    return current_result


# Function to save the results as a csv file
def save_to_csv(file_name, current_result):
    # Get the output file's name
    name, extention = os.path.splitext(file_name)
    cvsname = "./Results/featureExtraction/" + name[:3] + ".csv"
    output_csvfile = open(cvsname, "w+")
    try:
        writer = csv.writer(output_csvfile)
        writer.writerow(("features", "value"))
        for key, value in current_result.items():
            writer.writerow((str(key), str(value)))
    except IOError as e:
        print(e)
    finally:
        output_csvfile.close()


# 文件全部路径
print('>>>>>>>>>> Feature extraction beginning <<<<<<<<<<')
for path_DYN1, path_mask in zip(files_DYN1, files_mask):
    full_path_DYN1 = os.path.join(base_path_DYN1, path_DYN1)
    full_path_mask = os.path.join(base_path_mask, path_mask)
    name, extention = os.path.splitext(path_DYN1)
    print('case {} processing...'.format(name[:3]))
    # print(path_DYN1)
    # print(path_mask)
    result = one_featureExtraction(full_path_DYN1, full_path_mask, para_path)
    print('case {} finished.'.format(name[:3]))
    print('>>>>>>>>>>')
    # for key, value in result.items():  # 输出特征
    #     print("\t", key, ":", value)
    save_to_csv(path_DYN1, result)
print('>>>>>>>>>> Feature extraction ending <<<<<<<<<<')


