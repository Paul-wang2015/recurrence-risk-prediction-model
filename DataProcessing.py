#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project: mri-breast-tumor-segmentation
# File   : DataProcessing.py
# Author : Bo Wang
# Date   : 8/29/19

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectPercentile, f_classif
import numpy as np
import math


# #############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
# Attributes:
#   scores_: array-like, shape=(n_features,), scores of features.
#   pvalues_: array-like, shape=(n_features,), p-values of feature scores, None if score_func returned only scores.
def univariate_feature_selection(data_, target_):
    # # Some noisy data not correlated, this is an optional data
    # E = np.random.uniform(0, 0.1, size=(len(data_), 20))
    # # Add the noisy data to the informative features
    # X = np.hstack((data_, E))
    X = data_
    y = target_
    X_indices = np.arange(X.shape[-1])
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(X, y)
    np.savetxt("./Results/univarianceResults/univariatePvalues.csv", selector.pvalues_, delimiter=",")

    # Show and save the bar figure of [Feature number vs (p_value))]
    plt.bar(X_indices + 1, selector.pvalues_, width=.2, color='darkorange', edgecolor='black')
    plt.title("Univariate feature selection with F-test")
    plt.xlabel('Feature number')
    plt.ylabel(r'$p_{value}$')
    plt.axis('tight')
    plt.savefig("./Results/univarianceResults/univariatePvalues.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Show the bar figure of [Feature number vs Univariate score (-Log(p_value))]
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    plt.bar(X_indices + 1, scores, width=.2)
    plt.title("Univariate feature selection with F-test")
    plt.xlabel('Feature number')
    plt.ylabel(r'Univariate score ($-Log(p_{value})$)')
    plt.axis('tight')
    plt.savefig("./Results/univarianceResults/univariateScores.png", dpi=300, bbox_inches='tight')
    plt.show()


# #############################################################################
# Show and save the result of feature importance (显示和保存特征的重要性函数)
def plot_feature_importance(model, n_features, features_name, feature_importance_=None):
    if feature_importance_ is None:
        feature_importance_ = model.feature_importances_
    # n_features = total_data.shape[1]
    plt.barh(range(n_features), feature_importance_, align='center')
    plt.yticks(np.arange(n_features), features_name)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.savefig('./Results/featureImportance/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


# #############################################################################
# Plot feature size vs cross-validation AUC figure
def plot_AUC_feature_size(size_, value_):
    plt.plot(size_, value_, lw=2, marker="o")
    plt.grid(True)
    plt.xticks(range(0, 21, 5))
    plt.ylim(0.4, 0.8)
    plt.xlabel('Feature subset size')
    plt.ylabel('Cross_validation AUC')
    plt.title('Size vs AUC')
    plt.savefig('./Results/featureImportance/sizeVSauc.png', dpi=300, bbox_inches='tight')
    plt.show()

# # test data
# feature_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# auc_value = [0.52, 0.49, 0.54, 0.51, 0.59, 0.48, 0.59, 0.63, 0.61, 0.47,
#              0.55, 0.50, 0.52, 0.57, 0.54, 0.45, 0.52, 0.52, 0.47, 0.63]
# plot_AUC_feature_size(feature_size, auc_value)


# #############################################################################
# Plot Confusion_Matrix figure
def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# #############################################################################
# Compute the 95% CI for a set of data
def median_ci(data, confidence=0.95):
    data1 = sorted(data)
    n = len(data1)
    ll = 0.5*n - 0.98*math.sqrt(n)
    ul = 1 + 0.5*n + 0.98*math.sqrt(n)
    low_ = data1[math.ceil(ll) - 1]
    up_ = data1[math.floor(ul) - 1]
    return low_, up_

# # test data
# data = [0.1, 2.4, 0.1, 0.7, 1.4, 0.9, 3.2, 0.2, 0.3, 0.6, 3.2, 5.5]
# l, u = median_ci(data=auc_value)
# print(l, u)


# #############################################################################
# K-fold cross-validation
def k_fold_split(X_, y_, k=5):
    if(len(y_) < k):
        print('The length of original dataset is %d, that is less the k value, '
              'so k value is reset to half of the original length!' % len(y_))
        k = int(len(y_) / 2)
        print('The new k value is %d' % k)
    skf = StratifiedKFold(n_splits=k)
    i = 0
    for train_index, test_index in skf.split(X_, y_):
        print("------------Fold %d------------" % int(i+1))
        # print("TRAIN:", train_index, "\nTEST:", test_index)
        print("TEST:", test_index + 1)
        i += 1


# # test data
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 6], [7, 8]])
# y = np.array([0, 0, 0, 1, 1, 1])
# k_fold_split(X, y, k=10)
# skf = StratifiedKFold(n_splits=3)
# skf.get_n_splits(X, y)
# print(skf)

# #############################################################################
# calculate each rate for 2-classes
# ACC(accracy), PPV(precision), NPV, TPR(SENS), TNR(SPEC), FNR, FPR
def perf_measure(y_actual, y_hat):
    all_number = len(y_hat)
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
           TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
           FP += 1
        if y_actual[i] == y_hat[i] == 0:
           TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
           FN += 1
    accracy = float(TP + TN) / float(all_number)
    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP + FP)
    if FN + TN == 0:
        NPV = 0
    else:
        NPV = float(TN) / float(FN + TN)
    if TP + FN == 0:
        TPR = 0
        FNR = 0
    else:
        TPR = float(TP) / float(TP + FN)
        FNR = float(FN) / float(TP + FN)
    if FP + TN == 0:
        TNR = 0
        FPR = 0
    else:
        TNR = float(TN) / float(FP + TN)
        FPR = float(FP) / float(FP + TN)

    return accracy, precision, NPV, TPR, TNR, FNR, FPR


# #############################################################################
# calculate each rate for multi-classes
# ACC(accracy), PPV(precision), NPV, TPR(SENS), TNR(SPEC), FNR, FPR
def multi_perf_measure(y_actual, y_hat):
    all_number = len(y_hat)
    cm = confusion_matrix(y_actual, y_hat)
    print(cm, end='\n\n')
    TP = np.diag(cm).sum()
    FP = (cm.sum(axis=0) - np.diag(cm)).sum()
    FN = (cm.sum(axis=1) - np.diag(cm)).sum()
    TN = (cm.sum() - (FP + FN + TP)).sum()
    print(TP, FP, FN, TN)

    accracy = (TP + TN) / (all_number)
    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP + FP)
    if FN + TN == 0:
        NPV = 0
    else:
        NPV = float(TN) / float(FN + TN)
    if TP + FN == 0:
        TPR = 0
        FNR = 0
    else:
        TPR = float(TP) / float(TP + FN)
        FNR = float(FN) / float(TP + FN)
    if FP + TN == 0:
        TNR = 0
        FPR = 0
    else:
        TNR = float(TN) / float(FP + TN)
        FPR = float(FP) / float(FP + TN)

    return accracy, precision, NPV, TPR, TNR, FNR, FPR

# test data
# y_true = [2, 1, 0, 1, 2, 0]
# y_pred = [2, 0, 0, 1, 2, 1]
#
# accracy, precision, NPV, TPR, TNR, FNR, FPR = multi_perf_measure(y_true, y_pred)
# print(accracy, precision, NPV, TPR, TNR, FNR, FPR)


# #############################################################################
# Two-classes AUC calculation


# #############################################################################
# Multi-classes AUC calculation


