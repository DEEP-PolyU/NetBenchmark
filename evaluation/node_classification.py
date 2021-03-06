from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import f1_score

def node_classifcation(feature, labels):
    labels = labels.reshape(-1)
    shape = len(labels.shape)
    if shape == 2:
        labels = np.argmax(labels, axis=1)
    f1_mac = []
    f1_mic = []
    kf = KFold(n_splits=5, shuffle=True,random_state=0)
    for train_index, test_index in kf.split(feature):
        train_X, train_y = feature[train_index], labels[train_index]
        test_X, test_y = feature[test_index], labels[test_index]
        clf = svm.SVC(kernel='rbf')
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)

        micro = f1_score(test_y, preds, average='micro')
        macro = f1_score(test_y, preds, average='macro')
        f1_mac.append(macro)
        f1_mic.append(micro)
    f1_mic = np.array(f1_mic)
    f1_mac = np.array(f1_mac)
    f1_mic = np.mean(f1_mic)
    f1_mac = np.mean(f1_mac)
    print('Testing based on svm: ',
          'f1_micro=%.4f' % f1_mic,
          'f1_macro=%.4f' % f1_mac)

def node_classifcation_test(feature, labels):
    labels = labels.reshape(-1)
    shape = len(labels.shape)
    if shape == 2:
        labels = np.argmax(labels, axis=1)
    f1_mac = []
    f1_mic = []
    kf = KFold(n_splits=5, shuffle=True,random_state=0)
    for train_index, test_index in kf.split(feature):
        train_X, train_y = feature[train_index], labels[train_index]
        test_X, test_y = feature[test_index], labels[test_index]
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)

        micro = f1_score(test_y, preds, average='micro')
        macro = f1_score(test_y, preds, average='macro')
        f1_mac.append(macro)
        f1_mic.append(micro)
    f1_mic = np.array(f1_mic)
    f1_mac = np.array(f1_mac)
    f1_mic = np.mean(f1_mic)
    f1_mac = np.mean(f1_mac)
    return (f1_mic + f1_mac) / 2