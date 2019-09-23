#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project: mri-breast-tumor-segmentation
# File   : main.py
# Author : Bo Wang
# Date   : 9/3/19

import numpy as np
import csv
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.feature_selection import SelectPercentile, f_classif, SelectFwe

from DataLoading import load_data
from DataProcessing import k_fold_split, univariate_feature_selection, plot_feature_importance
from DataProcessing import median_ci, perf_measure, multi_perf_measure

# #############################################################################
# Set base file path
# #############################################################################
base_path = './Data/DataTransform/clinical/'
# file_name = 'comb.csv'
# my_file_path = base_path + file_name
target_names = ['class1', 'class2', 'class3']
# total_data, total_target, my_features = load_data(my_file_path)

# #############################################################################
# Step 1: 10-fold split, this procedure is run independently
# #############################################################################
# k_fold_split(X, y, k=10)

# #############################################################################
# Step 2: univariable analysis, this procedure is run independently
# #############################################################################
# univariate_feature_selection(X, y)

# #############################################################################
# Step 3: Feature important selection by RF
# #############################################################################
loop = 100  # loop 100 times to calculate the average value
file_name = 'clinical1.csv'
file_path = base_path + file_name
data, target, _features = load_data(file_path)
mean_of_feature = np.zeros(data.shape[1])
for i in range(1, loop):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(data, target)
    mean_of_feature = mean_of_feature + rf.feature_importances_
mean_of_feature /= loop
# print("Features sorted by their score:")
# print(sorted(zip(map(lambda x: round(x, 4), mean_of_feature), _features), reverse=True))
cvs_name = "./Results/featureImportance/feature_importance_clinical.csv"
output_csv_file = open(cvs_name, "w+")
writer = csv.writer(output_csv_file)
writer.writerow(("features", "value"))
for key, value in zip(_features, mean_of_feature):
    writer.writerow((str(key), str(round(value, 4))))
# plot_feature_importance(rf, mean_of_feature)

# #############################################################################
# Step 4: 100 times 10-fold corss-validation
# #############################################################################
'''
# Run classifier with cross-validation and plot ROC curves
# classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
# classifier = RandomForestClassifier(n_estimators=100)
classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100))

final_tprs_train = []
final_aucs_train = []
final_mean_fpr_train = np.linspace(0, 1, 100)
final_l_train, final_u_train = 0.0, 0.0
final_ACC_train, final_PPV_train, final_NPV_train = 0.0, 0.0, 0.0
final_TPR_train, final_TNR_train, final_FNR_train, final_FPR_train = 0.0, 0.0, 0.0, 0.0

final_tprs_test = []
final_aucs_test = []
final_mean_fpr_test = np.linspace(0, 1, 100)
final_l_test, final_u_test = 0.0, 0.0
final_ACC_test, final_PPV_test, final_NPV_test = 0.0, 0.0, 0.0
final_TPR_test, final_TNR_test, final_FNR_test, final_FPR_test = 0.0, 0.0, 0.0, 0.0

loop = 100
for time in range(loop):
    print("Time %d processing ===>>>" % (time + 1))
    tprs_train = []
    aucs_train = []
    tags_train = []
    pre_tags_train = []
    mean_fpr_train = np.linspace(0, 1, 100)

    tprs_test = []
    aucs_test = []
    tags_test = []
    pre_tags_test = []
    mean_fpr_test = np.linspace(0, 1, 100)

    for i in range(10):
        # read the 10-fold data after feature selection
        train_file_name = 'com_train' + str(i+1) + '.csv'
        train_path = base_path + train_file_name
        train_data, train_target, train_features = load_data(train_path)
        # ###############################################
        # ############### for two classes ###############
        train_data, train_target = train_data[train_target != 0], train_target[train_target != 0]
        # Binarize the label
        train_target = label_binarize(train_target, classes=[1, 2])
        train_target = train_target.ravel()
        # #####################################################
        # ################# for three classes #################
        # train_target = label_binarize(train_target, classes=[0, 1, 2])
        # n_classes = train_target.shape[1]

        test_file_name = 'com_test' + str(i + 1) + '.csv'
        test_path = base_path + test_file_name
        test_data, test_target, test_features = load_data(test_path)
        # ###############################################
        # ############### for two classes ###############
        test_data, test_target = test_data[test_target != 0], test_target[test_target != 0]
        # Binarize the label
        test_target = label_binarize(test_target, classes=[1, 2])
        test_target = test_target.ravel()
        # #################################################
        # ############### for three classes ###############
        # test_target = label_binarize(test_target, classes=[0, 1, 2])
        # n_classes = test_target.shape[1]

        # #####################################################
        # Compute the predict targets for train and test cohort
        # #####################################################
        # split the train cohort into train and validation dataset
        X_train, X_validation, y_train, y_validation = train_test_split(train_data, train_target, random_state=0)
        probas_train = classifier.fit(X_train, y_train).predict_proba(X_validation)  # train cohort
        pre_train = classifier.fit(X_train, y_train).predict(X_validation)
        # probas_train = classifier.fit(X_train, y_train).predict_proba(X_train)  # train cohort
        # pre_train = classifier.fit(X_train, y_train).predict(X_train)
        tags_train = np.append(tags_train, train_target)
        pre_tags_train = np.append(pre_tags_train, pre_train)

        probas_test = classifier.fit(X_train, y_train).predict_proba(test_data)  # test cohort
        pre_test = classifier.fit(X_train, y_train).predict(test_data)
        tags_test = np.append(tags_test, test_target)
        pre_tags_test = np.append(pre_tags_test, pre_test)

        # ####################################################
        # Compute ROC curve and area the curve, every fold (i)
        # ####################################################
        #
        # train cohort
        # ###############################################
        # ############### for two classes ###############
        fpr_train, tpr_train, thresholds_train = roc_curve(y_validation, probas_train[:, 1])
        ACC_train, PPV_train, NPV_train, TPR_train, TNR_train, FNR_train, FPR_train = perf_measure(y_validation, pre_train)
        # fpr_train, tpr_train, thresholds_train = roc_curve(y_train, probas_train[:, 1])
        # ACC_train, PPV_train, NPV_train, TPR_train, TNR_train, FNR_train, FPR_train = perf_measure(y_train, pre_train)

        tprs_train.append(interp(mean_fpr_train, fpr_train, tpr_train))
        tprs_train[-1][0] = 0.0
        roc_auc_train = auc(fpr_train, tpr_train)
        aucs_train.append(roc_auc_train)
        final_aucs_train.append(roc_auc_train)

        # #################################################
        # ############### for three classes ###############
        # fpr_train = dict()
        # tpr_train = dict()
        # roc_auc_train = dict()
        # # for j in range(n_classes):
        # #     fpr_train[j], tprs_train[j], _ = roc_curve(y_validation[:, j], probas_train[:, j])
        # #     roc_auc_train[j] = auc(fpr_train[j], tprs_train[j])
        # # Compute micro-average ROC curve and ROC area
        # fpr_train["micro"], tpr_train["micro"], _ = roc_curve(y_validation.ravel(), probas_train.ravel())
        # roc_auc_train["micro"] = auc(fpr_train["micro"], tpr_train["micro"])
        # tprs_train.append(interp(mean_fpr_train, fpr_train["micro"], tpr_train["micro"]))
        # tprs_train[-1][0] = 0.0
        # aucs_train.append(roc_auc_train["micro"])
        # final_aucs_train.append(roc_auc_train["micro"])
        # # print(accuracy_score(train_target, pre_train))

        # test cohort
        # ###############################################
        # ############### for two classes ###############
        fpr_test, tpr_test, thresholds_test = roc_curve(test_target, probas_test[:, 1])
        ACC_test, PPV_test, NPV_test, TPR_test, TNR_test, FNR_test, FPR_test = perf_measure(test_target, pre_test)

        tprs_test.append(interp(mean_fpr_test, fpr_test, tpr_test))
        tprs_test[-1][0] = 0.0
        roc_auc_test = auc(fpr_test, tpr_test)
        aucs_test.append(roc_auc_test)
        final_aucs_test.append(roc_auc_test)

        # #################################################
        # ############### for three classes ###############
        # fpr_test = dict()
        # tpr_test = dict()
        # roc_auc_test = dict()
        # # for j in range(n_classes):
        # #     fpr_test[j], tprs_test[j], _ = roc_curve(test_target[:, j], probas_test[:, j])
        # #     roc_auc_test[j] = auc(fpr_test[j], tprs_test[j])
        # # Compute micro-average ROC curve and ROC area
        # fpr_test["micro"], tpr_test["micro"], _ = roc_curve(test_target.ravel(), probas_test.ravel())
        # roc_auc_test["micro"] = auc(fpr_test["micro"], tpr_test["micro"])
        # tprs_test.append(interp(mean_fpr_test, fpr_test["micro"], tpr_test["micro"]))
        # tprs_test[-1][0] = 0.0
        # aucs_train.append(roc_auc_test["micro"])
        # final_aucs_test.append(roc_auc_test["micro"])

    # #######################
    # Compute the total rates
    # #######################
    # train cohort
    all_ACC_train, all_PPV_train, all_NPV_train, all_TPR_train, all_TNR_train, all_FNR_train, all_FPR_train = \
        perf_measure(tags_train, pre_tags_train)
    # #################
    # for three classes
    # cm_train = confusion_matrix(tags_train, pre_tags_train)
    # all_ACC_train = np.diag(cm_train).sum() / len(tags_train)
    # #################
    mean_tpr_train = np.mean(tprs_train, axis=0)
    mean_tpr_train[-1] = 1.0
    mean_auc_train = auc(mean_fpr_train, mean_tpr_train)  # mean auc
    std_auc_train = np.std(aucs_train)

    # test cohort
    all_ACC_test, all_PPV_test, all_NPV_test, all_TPR_test, all_TNR_test, all_FNR_test, all_FPR_test = \
        perf_measure(tags_test, pre_tags_test)
    # #################
    # for three classes
    # cm_test = confusion_matrix(tags_test, pre_tags_test)
    # all_ACC_test = np.diag(cm_test).sum() / len(tags_test)
    # #################
    mean_tpr_test = np.mean(tprs_test, axis=0)
    mean_tpr_test[-1] = 1.0
    mean_auc_test = auc(mean_fpr_test, mean_tpr_test)  # mean auc
    std_auc_test = np.std(aucs_test)

    # ###########################
    # calculate 100 times results
    # ###########################
    # train cohort
    final_tprs_train.append(interp(final_mean_fpr_train, mean_fpr_train, mean_tpr_train))
    final_tprs_train[-1][0] = 0.0

    final_ACC_train += all_ACC_train
    final_TPR_train += all_TPR_train
    final_TNR_train += all_TNR_train
    final_PPV_train += all_PPV_train
    final_NPV_train += all_NPV_train

    # test cohort
    final_tprs_test.append(interp(final_mean_fpr_test, mean_fpr_test, mean_tpr_test))
    final_tprs_test[-1][0] = 0.0

    final_ACC_test += all_ACC_test
    final_TPR_test += all_TPR_test
    final_TNR_test += all_TNR_test
    final_PPV_test += all_PPV_test
    final_NPV_test += all_NPV_test

# ###########################
# calculate 100 times results
# ###########################
# train cohort
final_ACC_train /= loop
final_TPR_train /= loop
final_TNR_train /= loop
final_PPV_train /= loop
final_NPV_train /= loop
final_l_train, final_u_train = median_ci(final_aucs_train)
final_mean_tpr_train = np.mean(final_tprs_train, axis=0)
final_mean_tpr_train[-1] = 1.0
final_mean_auc_train = auc(final_mean_fpr_train, final_mean_tpr_train)
final_std_auc_train = np.std(final_aucs_train)
# np.savetxt('./Results/ROCdata/clinicalResults/class-3_fm_fpr_train.txt', final_mean_fpr_train, fmt='%0.4f')
# np.savetxt('./Results/ROCdata/clinicalResults/class-3_fm_tpr_train.txt', final_mean_tpr_train, fmt='%0.4f')

# test cohort
final_ACC_test /= loop
final_TPR_test /= loop
final_TNR_test /= loop
final_PPV_test /= loop
final_NPV_test /= loop
final_l_test, final_u_test = median_ci(final_aucs_test)
final_mean_tpr_test = np.mean(final_tprs_test, axis=0)
final_mean_tpr_test[-1] = 1.0
final_mean_auc_test = auc(final_mean_fpr_test, final_mean_tpr_test)
final_std_auc_test = np.std(final_aucs_test)
# np.savetxt('./Results/ROCdata/clinicalResults/class-3_fm_fpr_test.txt', final_mean_fpr_test, fmt='%0.4f')
# np.savetxt('./Results/ROCdata/clinicalResults/class-3_fm_tpr_test.txt', final_mean_tpr_test, fmt='%0.4f')

# #######################
# print 100 times results
# #######################
print("==================== End loop, final results ====================")
print("train cohort:")
print("mean AUC of train: %.3f, std: %.3f" % (final_mean_auc_train, final_std_auc_train))
print("95%% CI: (%0.3f - %0.3f)" % (final_l_train, final_u_train))
print("ACC: {:2.2f}%, SENS: {:2.2f}%, SPEC: {:2.2f}%, PPV: {:2.2f}%, NPV: {:2.2f}%".format(
    final_ACC_train * 100, final_TPR_train * 100, final_TNR_train * 100, final_PPV_train * 100, final_NPV_train * 100))
print("-----------------------")
print("test cohort:")
print("mean AUC of test: %.3f, std: %.3f" % (final_mean_auc_test, final_std_auc_test))
print("95%% CI: (%0.3f - %0.3f)" % (final_l_test, final_u_test))
print("ACC: {:2.2f}%, SENS: {:2.2f}%, SPEC: {:2.2f}%, PPV: {:2.2f}%, NPV: {:2.2f}%".format(
    final_ACC_test * 100, final_TPR_test * 100, final_TNR_test * 100, final_PPV_test * 100, final_NPV_test * 100))

# ############################
# show and save the ROC figure
# ############################
plt.plot(final_mean_fpr_train, final_mean_tpr_train, color='b', label=r'(AUC = %0.3f)' % final_mean_auc_train, lw=2, alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Reference line', alpha=.8)
plt.xlim([-0.0, 1.0])
plt.ylim([-0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) of Train cohort')
plt.legend(loc="lower right")
# plt.savefig('./Results/ROCdata/clinicalResults/class-3_train.png', dpi=300, bbox_inches='tight')
plt.show()

plt.plot(final_mean_fpr_test, final_mean_tpr_test, color='b', label=r'(AUC = %0.3f)' % final_mean_auc_test, lw=2, alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Reference line', alpha=.8)
plt.xlim([-0.0, 1.0])
plt.ylim([-0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) of Test cohort')
plt.legend(loc="lower right")
# plt.savefig('./Results/ROCdata/clinicalResults/class-3_test.png', dpi=300, bbox_inches='tight')
plt.show()
'''



