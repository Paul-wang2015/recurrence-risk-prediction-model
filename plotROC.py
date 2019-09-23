#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project: mri-breast-tumor-segmentation
# File   : plotROC.py
# Author : Bo Wang
# Date   : 9/20/19

import matplotlib.pyplot as plt
import numpy as np

# ########################################################
# clinical model
# load the fpr and tpr values from [Data/rocdata/clinical]

# train cohort
cl_1v2_train_fpr = np.loadtxt('./Data/rocdata/clinical/class1vs2_fm_fpr_train.txt', delimiter=',')
cl_1v2_train_tpr = np.loadtxt('./Data/rocdata/clinical/class1vs2_fm_tpr_train.txt', delimiter=',')
cl_1v2_train_auc = 0.818

cl_1v3_train_fpr = np.loadtxt('./Data/rocdata/clinical/class1vs3_fm_fpr_train.txt', delimiter=',')
cl_1v3_train_tpr = np.loadtxt('./Data/rocdata/clinical/class1vs3_fm_tpr_train.txt', delimiter=',')
cl_1v3_train_auc = 0.934

cl_2v3_train_fpr = np.loadtxt('./Data/rocdata/clinical/class2vs3_fm_fpr_train.txt', delimiter=',')
cl_2v3_train_tpr = np.loadtxt('./Data/rocdata/clinical/class2vs3_fm_tpr_train.txt', delimiter=',')
cl_2v3_train_auc = 0.873

cl_1v23_train_fpr = np.loadtxt('./Data/rocdata/clinical/class1vs23_fm_fpr_train.txt', delimiter=',')
cl_1v23_train_tpr = np.loadtxt('./Data/rocdata/clinical/class1vs23_fm_tpr_train.txt', delimiter=',')
cl_1v23_train_auc = 0.844

cl_3_train_fpr = np.loadtxt('./Data/rocdata/clinical/class-3_fm_fpr_train.txt', delimiter=',')
cl_3_train_tpr = np.loadtxt('./Data/rocdata/clinical/class-3_fm_tpr_train.txt', delimiter=',')
cl_3_train_auc = 0.688

plt.figure()
plt.plot(cl_1v2_train_fpr, cl_1v2_train_tpr, color='b', label=r'(class [1, 2] AUC = %0.3f)' % cl_1v2_train_auc, lw=1, alpha=.8)
plt.plot(cl_1v3_train_fpr, cl_1v3_train_tpr, color='y', label=r'(class [1, 3] AUC = %0.3f)' % cl_1v3_train_auc, lw=1, alpha=.8)
plt.plot(cl_2v3_train_fpr, cl_2v3_train_tpr, color='g', label=r'(class [2, 3] AUC = %0.3f)' % cl_2v3_train_auc, lw=1, alpha=.8)
plt.plot(cl_1v23_train_fpr, cl_1v23_train_tpr, color='c', label=r'(class [1, 2/3] AUC = %0.3f)' % cl_1v23_train_auc, lw=2, alpha=.8)
plt.plot(cl_3_train_fpr, cl_3_train_tpr, color='m', label='(class [1, 2, 3] AUC = %0.3f)' % cl_3_train_auc, lw=2, alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Reference line', alpha=.8)
plt.xlim([-0.0, 1.0])
plt.ylim([-0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of training cohort by clinical model')
plt.legend(loc="lower right")
plt.savefig('./Results/rocResults/clinical_train.png', dpi=300, bbox_inches='tight')
plt.show()

# test cohort
cl_1v2_test_fpr = np.loadtxt('./Data/rocdata/clinical/class1vs2_fm_fpr_test.txt', delimiter=',')
cl_1v2_test_tpr = np.loadtxt('./Data/rocdata/clinical/class1vs2_fm_tpr_test.txt', delimiter=',')
cl_1v2_test_auc = 0.635

cl_1v3_test_fpr = np.loadtxt('./Data/rocdata/clinical/class1vs3_fm_fpr_test.txt', delimiter=',')
cl_1v3_test_tpr = np.loadtxt('./Data/rocdata/clinical/class1vs3_fm_tpr_test.txt', delimiter=',')
cl_1v3_test_auc = 0.800

cl_2v3_test_fpr = np.loadtxt('./Data/rocdata/clinical/class2vs3_fm_fpr_test.txt', delimiter=',')
cl_2v3_test_tpr = np.loadtxt('./Data/rocdata/clinical/class2vs3_fm_tpr_test.txt', delimiter=',')
cl_2v3_test_auc = 0.731

cl_1v23_test_fpr = np.loadtxt('./Data/rocdata/clinical/class1vs23_fm_fpr_test.txt', delimiter=',')
cl_1v23_test_tpr = np.loadtxt('./Data/rocdata/clinical/class1vs23_fm_tpr_test.txt', delimiter=',')
cl_1v23_test_auc = 0.702

cl_3_test_fpr = np.loadtxt('./Data/rocdata/clinical/class-3_fm_fpr_test.txt', delimiter=',')
cl_3_test_tpr = np.loadtxt('./Data/rocdata/clinical/class-3_fm_tpr_test.txt', delimiter=',')
cl_3_test_auc = 0.695

plt.figure()
plt.plot(cl_1v2_test_fpr, cl_1v2_test_tpr, color='b', label=r'(class [1, 2] AUC = %0.3f)' % cl_1v2_test_auc, lw=1, alpha=.8)
plt.plot(cl_1v3_test_fpr, cl_1v3_test_tpr, color='y', label=r'(class [1, 3] AUC = %0.3f)' % cl_1v3_test_auc, lw=1, alpha=.8)
plt.plot(cl_2v3_test_fpr, cl_2v3_test_tpr, color='g', label=r'(class [2, 3] AUC = %0.3f)' % cl_2v3_test_auc, lw=1, alpha=.8)
plt.plot(cl_1v23_test_fpr, cl_1v23_test_tpr, color='c', label=r'(class [1, 2/3] AUC = %0.3f)' % cl_1v23_test_auc, lw=2, alpha=.8)
plt.plot(cl_3_test_fpr, cl_3_test_tpr, color='m', label='(class [1, 2, 3] AUC = %0.3f)' % cl_3_test_auc, lw=2, alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Reference line', alpha=.8)
plt.xlim([-0.0, 1.0])
plt.ylim([-0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of testing cohort by clinical model')
plt.legend(loc="lower right")
plt.savefig('./Results/rocResults/clinical_test.png', dpi=300, bbox_inches='tight')
plt.show()

# ########################################################
# radiomics model
# load the fpr and tpr values from [Data/rocdata/radiomics]

# train cohort
ra_1v2_train_fpr = np.loadtxt('./Data/rocdata/radiomics/class1vs2_fm_fpr_train.txt', delimiter=',')
ra_1v2_train_tpr = np.loadtxt('./Data/rocdata/radiomics/class1vs2_fm_tpr_train.txt', delimiter=',')
ra_1v2_train_auc = 0.627

ra_1v3_train_fpr = np.loadtxt('./Data/rocdata/radiomics/class1vs3_fm_fpr_train.txt', delimiter=',')
ra_1v3_train_tpr = np.loadtxt('./Data/rocdata/radiomics/class1vs3_fm_tpr_train.txt', delimiter=',')
ra_1v3_train_auc = 0.677

ra_2v3_train_fpr = np.loadtxt('./Data/rocdata/radiomics/class2vs3_fm_fpr_train.txt', delimiter=',')
ra_2v3_train_tpr = np.loadtxt('./Data/rocdata/radiomics/class2vs3_fm_tpr_train.txt', delimiter=',')
ra_2v3_train_auc = 0.598

ra_1v23_train_fpr = np.loadtxt('./Data/rocdata/radiomics/class1vs23_fm_fpr_train.txt', delimiter=',')
ra_1v23_train_tpr = np.loadtxt('./Data/rocdata/radiomics/class1vs23_fm_tpr_train.txt', delimiter=',')
ra_1v23_train_auc = 0.613

ra_3_train_fpr = np.loadtxt('./Data/rocdata/radiomics/class-3_fm_fpr_train.txt', delimiter=',')
ra_3_train_tpr = np.loadtxt('./Data/rocdata/radiomics/class-3_fm_tpr_train.txt', delimiter=',')
ra_3_train_auc = 0.660

plt.figure()
plt.plot(ra_1v2_train_fpr, ra_1v2_train_tpr, color='b', label=r'(class [1, 2] AUC = %0.3f)' % ra_1v2_train_auc, lw=1, alpha=.8)
plt.plot(ra_1v3_train_fpr, ra_1v3_train_tpr, color='y', label=r'(class [1, 3] AUC = %0.3f)' % ra_1v3_train_auc, lw=1, alpha=.8)
plt.plot(ra_2v3_train_fpr, ra_2v3_train_tpr, color='g', label=r'(class [2, 3] AUC = %0.3f)' % ra_2v3_train_auc, lw=1, alpha=.8)
plt.plot(ra_1v23_train_fpr, ra_1v23_train_tpr, color='c', label=r'(class [1, 2/3] AUC = %0.3f)' % ra_1v23_train_auc, lw=2, alpha=.8)
plt.plot(ra_3_train_fpr, ra_3_train_tpr, color='m', label='(class [1, 2, 3] AUC = %0.3f)' % ra_3_train_auc, lw=2, alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Reference line', alpha=.8)
plt.xlim([-0.0, 1.0])
plt.ylim([-0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of training cohort by radiomics model')
plt.legend(loc="lower right")
plt.savefig('./Results/rocResults/radiomics_train.png', dpi=300, bbox_inches='tight')
plt.show()

# test cohort
ra_1v2_test_fpr = np.loadtxt('./Data/rocdata/radiomics/class1vs2_fm_fpr_test.txt', delimiter=',')
ra_1v2_test_tpr = np.loadtxt('./Data/rocdata/radiomics/class1vs2_fm_tpr_test.txt', delimiter=',')
ra_1v2_test_auc = 0.563

ra_1v3_test_fpr = np.loadtxt('./Data/rocdata/radiomics/class1vs3_fm_fpr_test.txt', delimiter=',')
ra_1v3_test_tpr = np.loadtxt('./Data/rocdata/radiomics/class1vs3_fm_tpr_test.txt', delimiter=',')
ra_1v3_test_auc = 0.599

ra_2v3_test_fpr = np.loadtxt('./Data/rocdata/radiomics/class2vs3_fm_fpr_test.txt', delimiter=',')
ra_2v3_test_tpr = np.loadtxt('./Data/rocdata/radiomics/class2vs3_fm_tpr_test.txt', delimiter=',')
ra_2v3_test_auc = 0.511

ra_1v23_test_fpr = np.loadtxt('./Data/rocdata/radiomics/class1vs23_fm_fpr_test.txt', delimiter=',')
ra_1v23_test_tpr = np.loadtxt('./Data/rocdata/radiomics/class1vs23_fm_tpr_test.txt', delimiter=',')
ra_1v23_test_auc = 0.576

ra_3_test_fpr = np.loadtxt('./Data/rocdata/radiomics/class-3_fm_fpr_test.txt', delimiter=',')
ra_3_test_tpr = np.loadtxt('./Data/rocdata/radiomics/class-3_fm_tpr_test.txt', delimiter=',')
ra_3_test_auc = 0.643

plt.figure()
plt.plot(ra_1v2_test_fpr, ra_1v2_test_tpr, color='b', label=r'(class [1, 2] AUC = %0.3f)' % ra_1v2_test_auc, lw=1, alpha=.8)
plt.plot(ra_1v3_test_fpr, ra_1v3_test_tpr, color='y', label=r'(class [1, 3] AUC = %0.3f)' % ra_1v3_test_auc, lw=1, alpha=.8)
plt.plot(ra_2v3_test_fpr, ra_2v3_test_tpr, color='g', label=r'(class [2, 3] AUC = %0.3f)' % ra_2v3_test_auc, lw=1, alpha=.8)
plt.plot(ra_1v23_test_fpr, ra_1v23_test_tpr, color='c', label=r'(class [1, 2/3] AUC = %0.3f)' % ra_1v23_test_auc, lw=2, alpha=.8)
plt.plot(ra_3_test_fpr, ra_3_test_tpr, color='m', label='(class [1, 2, 3] AUC = %0.3f)' % ra_3_test_auc, lw=2, alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Reference line', alpha=.8)
plt.xlim([-0.0, 1.0])
plt.ylim([-0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of testing cohort by radiomics model')
plt.legend(loc="lower right")
plt.savefig('./Results/rocResults/radiomics_test.png', dpi=300, bbox_inches='tight')
plt.show()

# ########################################################
# combination model
# load the fpr and tpr values from [Data/rocdata/comb]

# train cohort
com_1v2_train_fpr = np.loadtxt('./Data/rocdata/comb/class1vs2_fm_fpr_train.txt', delimiter=',')
com_1v2_train_tpr = np.loadtxt('./Data/rocdata/comb/class1vs2_fm_tpr_train.txt', delimiter=',')
com_1v2_train_auc = 0.653

com_1v3_train_fpr = np.loadtxt('./Data/rocdata/comb/class1vs3_fm_fpr_train.txt', delimiter=',')
com_1v3_train_tpr = np.loadtxt('./Data/rocdata/comb/class1vs3_fm_tpr_train.txt', delimiter=',')
com_1v3_train_auc = 0.799

com_2v3_train_fpr = np.loadtxt('./Data/rocdata/comb/class2vs3_fm_fpr_train.txt', delimiter=',')
com_2v3_train_tpr = np.loadtxt('./Data/rocdata/comb/class2vs3_fm_tpr_train.txt', delimiter=',')
com_2v3_train_auc = 0.648

com_1v23_train_fpr = np.loadtxt('./Data/rocdata/comb/class1vs23_fm_fpr_train.txt', delimiter=',')
com_1v23_train_tpr = np.loadtxt('./Data/rocdata/comb/class1vs23_fm_tpr_train.txt', delimiter=',')
com_1v23_train_auc = 0.674

com_3_train_fpr = np.loadtxt('./Data/rocdata/comb/class-3_fm_fpr_train.txt', delimiter=',')
com_3_train_tpr = np.loadtxt('./Data/rocdata/comb/class-3_fm_tpr_train.txt', delimiter=',')
com_3_train_auc = 0.717

plt.figure()
plt.plot(com_1v2_train_fpr, com_1v2_train_tpr, color='b', label=r'(class [1, 2] AUC = %0.3f)' % com_1v2_train_auc, lw=1, alpha=.8)
plt.plot(com_1v3_train_fpr, com_1v3_train_tpr, color='y', label=r'(class [1, 3] AUC = %0.3f)' % com_1v3_train_auc, lw=1, alpha=.8)
plt.plot(com_2v3_train_fpr, com_2v3_train_tpr, color='g', label=r'(class [2, 3] AUC = %0.3f)' % com_2v3_train_auc, lw=1, alpha=.8)
plt.plot(com_1v23_train_fpr, com_1v23_train_tpr, color='c', label=r'(class [1, 2/3] AUC = %0.3f)' % com_1v23_train_auc, lw=2, alpha=.8)
plt.plot(com_3_train_fpr, com_3_train_tpr, color='m', label='(class [1, 2, 3] AUC = %0.3f)' % com_3_train_auc, lw=2, alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Reference line', alpha=.8)
plt.xlim([-0.0, 1.0])
plt.ylim([-0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of training cohort by combination model')
plt.legend(loc="lower right")
plt.savefig('./Results/rocResults/comb_train.png', dpi=300, bbox_inches='tight')
plt.show()

# test cohort
com_1v2_test_fpr = np.loadtxt('./Data/rocdata/comb/class1vs2_fm_fpr_test.txt', delimiter=',')
com_1v2_test_tpr = np.loadtxt('./Data/rocdata/comb/class1vs2_fm_tpr_test.txt', delimiter=',')
com_1v2_test_auc = 0.612

com_1v3_test_fpr = np.loadtxt('./Data/rocdata/comb/class1vs3_fm_fpr_test.txt', delimiter=',')
com_1v3_test_tpr = np.loadtxt('./Data/rocdata/comb/class1vs3_fm_tpr_test.txt', delimiter=',')
com_1v3_test_auc = 0.748

com_2v3_test_fpr = np.loadtxt('./Data/rocdata/comb/class2vs3_fm_fpr_test.txt', delimiter=',')
com_2v3_test_tpr = np.loadtxt('./Data/rocdata/comb/class2vs3_fm_tpr_test.txt', delimiter=',')
com_2v3_test_auc = 0.599

com_1v23_test_fpr = np.loadtxt('./Data/rocdata/comb/class1vs23_fm_fpr_test.txt', delimiter=',')
com_1v23_test_tpr = np.loadtxt('./Data/rocdata/comb/class1vs23_fm_tpr_test.txt', delimiter=',')
com_1v23_test_auc = 0.638

com_3_test_fpr = np.loadtxt('./Data/rocdata/comb/class-3_fm_fpr_test.txt', delimiter=',')
com_3_test_tpr = np.loadtxt('./Data/rocdata/comb/class-3_fm_tpr_test.txt', delimiter=',')
com_3_test_auc = 0.677

plt.figure()
plt.plot(com_1v2_test_fpr, com_1v2_test_tpr, color='b', label=r'(class [1, 2] AUC = %0.3f)' % com_1v2_test_auc, lw=1, alpha=.8)
plt.plot(com_1v3_test_fpr, com_1v3_test_tpr, color='y', label=r'(class [1, 3] AUC = %0.3f)' % com_1v3_test_auc, lw=1, alpha=.8)
plt.plot(com_2v3_test_fpr, com_2v3_test_tpr, color='g', label=r'(class [2, 3] AUC = %0.3f)' % com_2v3_test_auc, lw=1, alpha=.8)
plt.plot(com_1v23_test_fpr, com_1v23_test_tpr, color='c', label=r'(class [1, 2/3] AUC = %0.3f)' % com_1v23_test_auc, lw=2, alpha=.8)
plt.plot(com_3_test_fpr, com_3_test_tpr, color='m', label='(class [1, 2, 3] AUC = %0.3f)' % com_3_test_auc, lw=2, alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Reference line', alpha=.8)
plt.xlim([-0.0, 1.0])
plt.ylim([-0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of testing cohort by combination model')
plt.legend(loc="lower right")
plt.savefig('./Results/rocResults/comb_test.png', dpi=300, bbox_inches='tight')
plt.show()
