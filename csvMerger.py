#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project: mri-breast-tumor-segmentation
# File   : csvMerger.py
# Author : Bo Wang
# Date   : 7/19/19

import glob
import csv
import pandas

csv_list = glob.glob('./Results/featureExtraction/*.csv')  # 查看同文件夹下的csv文件数
print(u'共发现%s个CSV文件' % len(csv_list))
print(u'正在处理......')
csv_list.sort()
# 创建表头
with open('./Results/featureExtraction/001.csv', 'rt') as f:
    reader = csv.reader(f)
    column = [row[0] for row in reader][23:]
    # print(type(column))
    # print(column)
    with open('./Results/featureExtraction/result.csv', 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(column)
        fp.close()
# 追加属性值
for i in csv_list:
    print('########################################')
    print(u'当前正在处理文件%s ...' % str(i))
    with open(i, 'rt') as f:
        reader = csv.reader(f)
        column = [row[1] for row in reader][23:]
        # print(type(column))
        # print(column)
        with open('./Results/featureExtraction/result.csv', 'a') as fp:
            writer = csv.writer(fp)
            writer.writerow(column)
            fp.close()
    print(u'%s文件追加完成' % str(i))
print('########################################')
print(u'处理完毕')

