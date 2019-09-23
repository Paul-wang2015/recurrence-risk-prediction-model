#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project: mri-breast-tumor-segmentation
# File   : classificationByEachMethod.py
# Author : Bo Wang
# Date   : 7/24/19

from sklearn.datasets import load_iris
import pandas as pd
import os
import mglearn
import csv
import numpy as np
from DataLoading import load_data, load_and_split_data
from sklearn import tree, svm, naive_bayes, neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
sns.set(color_codes=True)

# Setup the original dataset
iris = load_iris()
base_path = './Data/breastfeature/new/'
file_name = 'radiomics50.csv'
# file_name = 'featureofallresult770.csv'  # Feature: all (770)
# file_name = 'featureofshape.csv'  # Feature: shape (14)
# file_name = 'featureofshape-2-3-1.csv'  # Feature: shape (14)
# file_name = 'featureoffirstorder-2-3-1.csv'  # Feature: first-order (18)
# file_name = 'featureoftextural-2-3-1.csv'  # Feature: textural (68)
# file_name = 'featureofwavelet-2-3.csv'  # Feature: wavelet (670)
# file_name = 'featureofselected25.csv'  # Feature: all (25)
# file_name = 'featureofselected25-2-3-1.csv'  # Feature: all (25)
# file_name = 'clinical_after_sel3.csv'
my_file_path = base_path + file_name
total_data, total_target, X_train, y_train, X_test, y_test, my_features = load_and_split_data(my_file_path)
target_names = ['class1', 'class2', 'class3']

# Load dataset and split into training data and testing data

# ==================== 特征归一化 ====================
# 计算训练集中每个特征的最小值
min_on_train = X_train.min(axis=0)
# 计算训练集中每个特征的范围（最大值-最小值）
range_on_training = (X_train-min_on_train).max(axis=0)
# 范围缩放
X_train_scaled = (X_train-min_on_train)/range_on_training
X_test_scaled = (X_test-min_on_train)/range_on_training
# np.savetxt("./Results/scalertrain.csv", X_train_scaled, delimiter=",")


# # Method 2
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# ==================== 特征标准化 ====================
# # 计算训练集中的每个特征的平均值
# mean_on_train = X_train.mean(axis=0)
# # 计算训练集中每个特征的标准差
# std_on_train = X_train.std(axis=0)
# # 减去平均值，然后乘以标准差的倒数
# X_train_scaled = (X_train-mean_on_train)/std_on_train
# X_test_scaled = (X_test-mean_on_train)/std_on_train
# Method 2
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# import mglearn
# param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1, 10, 100]}

# grid_searsh = GridSearchCV(SVC(gamma='auto'), param_grid, cv=5)
# grid_searsh.fit(X_train, y_train)
# print("Test set score:{:.3f}".format(grid_searsh.score(X_test, y_test)))
# print("Best paraments:{}".format(grid_searsh.best_params_))
# print("Best cross-validation score:{:.4f}".format(grid_searsh.best_score_))
# # 转换为dataframe（数据框）
# results = pd.DataFrame(grid_searsh.cv_results_)
# # display(results.head())
# scores = np.array(results.mean_test_score).reshape(6, 6)
# # 对交叉验证平均分数作图
# mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
#                       ylabel='C', yticklabels=param_grid['C'], cmap='viridis')
# plt.show()

print("训练数据集的样本数和特征维度:", X_train_scaled.shape)
print("测试数据集的样本数和特征维度:", X_test_scaled.shape)
print("Feature names include:", my_features)


# 显示和保存特征的重要性函数
def plot_feature_importances(model, feature_importances_=None):
    if feature_importances_ is None:
        feature_importances_ = model.feature_importances_
    n_features = total_data.shape[1]
    plt.barh(range(n_features), feature_importances_, align='center')
    plt.yticks(np.arange(n_features), my_features)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.savefig('./Results/feature_importance/dt_selected_importances.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== 决策树 ====================
from sklearn.tree import DecisionTreeClassifier
# loop = 100
# mean_of_train, mean_of_test = 0.0, 0.0
# mean_of_feature = np.zeros(X_train_scaled.shape[1])
# for i in range(1, loop):
#     detree = DecisionTreeClassifier(random_state=0).fit(X_train_scaled, y_train.ravel())
#     mean_of_train += detree.score(X_train_scaled, y_train)
#     mean_of_test += detree.score(X_test_scaled, y_test)
#     mean_of_feature = mean_of_feature + detree.feature_importances_
# mean_of_train /= loop
# mean_of_test /= loop
# mean_of_feature /= loop
# print(mean_of_feature)
# # np.savetxt("./Results/feature_importance/wavelet_importances.csv", mean_of_feature, fmt='%.4f', delimiter=",")
# cvsname = "./Results/feature_importance/selected_importances.csv"
# output_csvfile = open(cvsname, "w+")
# writer = csv.writer(output_csvfile)
# writer.writerow(("features", "value"))
# for key, value in zip(my_features, mean_of_feature):
#     writer.writerow((str(key), str(value)))
# print('平均训练集精度{:.2f},平均测试集精度{:.2f}'.format(mean_of_train, mean_of_test))
# plot_feature_importances(detree, mean_of_feature)

# -------------------- 决策树的可视化 --------------------
# from sklearn.tree import export_graphviz
# import graphviz
#
# dot_grap = export_graphviz(detree, out_file=None, class_names=['group1', 'group2', 'group3'],
#                            feature_names=my_features, filled=True, rounded=True, special_characters=True)
# grap = graphviz.Source(dot_grap)
# grap.render('./Results/shapeResults')

# -------------------- 随机森林 --------------------
# forest0 = RandomForestClassifier(n_estimators=50).fit(X_train, y_train.ravel())
# y_score = forest0.predict_proba(X_test)
# y_pred = forest0.predict(X_train[:20])
# print(y_score)
# print(y_test.shape)


from sklearn.metrics import confusion_matrix, classification_report

# y_true = [2, 1, 0, 1, 2, 0]
# y_pred = [2, 0, 0, 1, 2, 1]

# X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, random_state=42)
# forest1 = RandomForestClassifier(n_estimators=50).fit(X_train1, y_train1.ravel())
# y_score1 = forest1.predict_proba(X_test1)
# y_pred1 = forest1.predict(X_test1)
# C = confusion_matrix(y_test1, y_pred1)
# print(C, end='\n\n')
# print(classification_report(y_test1, y_pred1, target_names=target_names))
# sns.heatmap(C, annot=None)

# #############################################################################
# from DataProcessing import plot_confusion_matrix
# labels = target_names
# tick_marks = np.array(range(len(labels))) + 0.5
# cm = confusion_matrix(y_test1, y_pred1)
# np.set_printoptions(precision=2)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print(cm_normalized)
# plt.figure(figsize=(12, 8), dpi=120)
# ind_array = np.arange(len(labels))
# x, y = np.meshgrid(ind_array, ind_array)
#
# for x_val, y_val in zip(x.flatten(), y.flatten()):
#     c = cm_normalized[y_val][x_val]
#     if c > 0.01:
#         plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# # offset the tick
# plt.gca().set_xticks(tick_marks, minor=True)
# plt.gca().set_yticks(tick_marks, minor=True)
# plt.gca().xaxis.set_ticks_position('none')
# plt.gca().yaxis.set_ticks_position('none')
# plt.grid(True, which='minor', linestyle='-')
# plt.gcf().subplots_adjust(bottom=0.15)
#
# plot_confusion_matrix(cm_normalized, labels=labels, title='Normalized confusion matrix')
# # show confusion matrix
# # plt.savefig('../Data/confusion_matrix.png', format='png')
# plt.show()

# score0 = forest0.score(X_train, y_train)
# y0 = forest0.predict(X_train)
# print(y_train.ravel())
# print(y0)
# print(score0)
# score1 = forest0.score(X_test, y_test)
# y1 = forest0.predict(X_test)
# print(y_test.ravel())
# print(y1)
# print(score1)
# print('标准化前\n训练集精确度:{:2.2f}% \n测试集精确度:{:2.2f}%'.format(100*forest0.score(X_train, y_train),
#                                                        100*forest0.score(X_test, y_test)))
print('====================')
# forest = RandomForestClassifier(n_estimators=100).fit(X_train_scaled, y_train.ravel())
# print('标准化后\n训练集精确度:{:2.2f}% \n测试集精确度:{:2.2f}%'.format(100*forest.score(X_train_scaled, y_train),
#                                                        100*forest.score(X_test_scaled, y_test)))
# print(forest.feature_importances_)
# plot_feature_importances(forest)
# # -------------------- 梯度提升机 --------------------
# from sklearn.ensemble import GradientBoostingClassifier
# gbrt = GradientBoostingClassifier(random_state=0, max_depth=1).fit(X_train_scaled, y_train)
# print('测试集精确度{},训练集精确度{}。'.format(gbrt.score(X_train_scaled, y_train), gbrt.score(X_test_scaled, y_test)))
# # plot_feature_importances(gbrt)
# print("Decision function shape:{}".format(gbrt.decision_function(X_test_scaled).shape))
# print("Decision function:\n{}".format(gbrt.decision_function(X_test_scaled)[:6, :]))
# print("argmax of decision function:\n{}".format(np.argmax(gbrt.decision_function(X_test_scaled), axis=1)))
# print("predictions\n{}".format(gbrt.predict(X_test_scaled)))
# print(y_test)
# # 显示predict_proba的前几个元素
# print("predicted probabilities:\n{}".format(gbrt.predict_proba(X_test_scaled)[:6]))
# # 显示每行的和都是1
# print("sums:{}".format(gbrt.predict_proba(X_test_scaled)[:6].sum(axis=1)))
# # ==================== 支持向量机SVM ====================
# svc0 = svm.SVC(gamma='scale', C=1000).fit(X_train, y_train)
# print('标准化前\n训练集精确度:{:2.2f}% \n测试集精确度:{:2.2f}%'.format(100*svc0.score(X_train, y_train),
#                                                        100*svc0.score(X_test, y_test)))
# print('====================')
# svc = svm.SVC(gamma='scale', C=1000).fit(X_train_scaled, y_train)
# print('标准化后\n训练集精确度:{:2.2f}% \n测试集精确度:{:2.2f}%'.format(100*svc.score(X_train_scaled, y_train),
#                                                        100*svc.score(X_test_scaled, y_test)))


def try_different_method(clf_):
    # 分别计算标准化前和标准化后训练集和测试集在不同分类器下的精确度
    # clf_: 分类器方法
    # 返回: 标准化前和标准化后训练集和测试集的精确度
    clf_.fit(X_train, y_train.ravel())
    train_score_before_scaled = clf_.score(X_train, y_train.ravel())
    test_score_before_scaled = clf_.score(X_test, y_test.ravel())
    clf_.fit(X_train_scaled, y_train.ravel())
    train_score_after_scaled = clf_.score(X_train_scaled, y_train.ravel())
    test_score_after_scaled = clf_.score(X_test_scaled, y_test.ravel())
    # print('the score is :{:2.2f}%'.format(100*score))
    return train_score_before_scaled, test_score_before_scaled, train_score_after_scaled, test_score_after_scaled


def mean_score_of_different_method(clf_, loop_):
    # 分别计算标准化前和标准化后训练集和测试集的平均精确度
    # clf_: 分类器
    # loop_: 循环的次数
    # 返回: 标准化前和标准化后训练集和测试集的平均精确度
    sum_of_train_bs_, sum_of_test_bs_, sum_of_train_as_, sum_of_test_as_ = 0.0, 0.0, 0.0, 0.0
    # mean_train_bs_, mean_test_bs_, mean_train_as_, mean_test_as_ = 0.0, 0.0, 0.0, 0.0
    for i in range(0, loop_):
        temp_train_score_bs, temp_test_score_bs, temp_train_score_as, temp_test_score_as = try_different_method(clf_)
        sum_of_train_bs_ = sum_of_train_bs_ + temp_train_score_bs
        sum_of_test_bs_ = sum_of_test_bs_ + temp_test_score_bs
        sum_of_train_as_ = sum_of_train_as_ + temp_train_score_as
        sum_of_test_as_ = sum_of_test_as_ + temp_test_score_as
    mean_train_bs_, mean_test_bs_ = sum_of_train_bs_ / loop, sum_of_test_bs_ / loop
    mean_train_as_, mean_test_as_ = sum_of_train_as_ / loop, sum_of_test_as_ / loop

    return mean_train_bs_, mean_test_bs_, mean_train_as_, mean_test_as_


def print_results(clf_key_, mean_train_bs_, mean_test_bs_, mean_train_as_, mean_test_as_):
    # 输出结果
    print('the classifier is :', clf_key_)
    print('标准化前\n训练集平均精确度:{:2.2f}% \n测试集平均精确度:{:2.2f}%'.format(100 * mean_train_bs_, 100 * mean_test_bs_))
    print('标准化后\n训练集平均精确度:{:2.2f}% \n测试集平均精确度:{:2.2f}%'.format(100 * mean_train_as_, 100 * mean_test_as_))


# 所有的分类方法集合, ['naive_mul': naive_bayes.MultinomialNB(),]
clfs = {'svm': svm.SVC(gamma='auto', C=1000),
#         'decision_tree': tree.DecisionTreeClassifier(),
#         'naive_gaussian': naive_bayes.GaussianNB(),
#         'K_neighbor': neighbors.KNeighborsClassifier(),
#         'bagging_knn': BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
#         'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5),
        'random_forest': RandomForestClassifier(n_estimators=50),
        # 'adaboost': AdaBoostClassifier(n_estimators=50),
        # 'gradient_boost': GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0),
        # 'MLP': MLPClassifier(solver='lbfgs', random_state=0, max_iter=10000)
        }

# 测试代码
loop = 100  # 循环次数,循环 loop 次取平均值
print('训练开始----->')
max_train_bs, max_test_bs, max_train_as, max_test_as = 0.0, 0.0, 0.0, 0.0
max_train_clf_bs, max_test_clf_bs, max_train_clf_as, max_test_clf_as = 'None', 'None', 'None', 'None'
for clf_key in clfs.keys():
    print('=============================')
    clf = clfs[clf_key]
    mean_train_bs, mean_test_bs, mean_train_as, mean_test_as = mean_score_of_different_method(clf, loop)
    print_results(clf_key, mean_train_bs, mean_test_bs, mean_train_as, mean_test_as)
    if mean_train_bs > max_train_bs:
        max_train_bs = mean_train_bs
        max_train_clf_bs = clf_key
    if mean_test_bs > max_test_bs:
        max_test_bs = mean_test_bs
        max_test_clf_bs = clf_key
    if mean_train_as > max_train_as:
        max_train_as = mean_train_as
        max_train_clf_as = clf_key
    if mean_test_as > max_test_as:
        max_test_as = mean_test_as
        max_test_clf_as = clf_key
print('===================================')
print('<-----训练结束')
print('>>>>>>>>>> 最终结果如下 <<<<<<<<<<')
print('(I)标准化前\n训练集平均精确度最大分类器:', max_train_clf_bs)
print('\t\t\t\t最大值:{:2.2f}%'.format(100 * max_train_bs))
print('测试集平均精确度最大分类器:', max_test_clf_bs)
print('\t\t\t\t最大值:{:2.2f}%'.format(100 * max_test_bs))
print('(II)标准化后\n训练集平均精确度最大分类器:', max_train_clf_as)
print('\t\t\t\t最大值:{:2.2f}%'.format(100 * max_train_as))
print('测试集平均精确度最大分类器:', max_test_clf_as)
print('\t\t\t\t最大值:{:2.2f}%'.format(100 * max_test_as))
print('===================================')









