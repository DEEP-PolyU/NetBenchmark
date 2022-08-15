from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import f1_score
from tqdm import tqdm
def node_classifcation_10_time(feature, labels):
    pbar = tqdm(total=10, position=0, leave=True)
    labels = labels.reshape(-1)
    shape = len(labels.shape)
    if shape == 2:
        labels = np.argmax(labels, axis=1)
    i = 0
    f1_mac = []
    f1_mic = []
    for i in range(10):
        f1_mic_fold = []
        f1_mac_fold = []
        kf = KFold(n_splits=5, shuffle=True,random_state=0)
        for train_index, test_index in kf.split(feature):
            train_X, train_y = feature[train_index], labels[train_index]
            test_X, test_y = feature[test_index], labels[test_index]
            clf = svm.SVC(kernel='rbf')
            clf.fit(train_X, train_y)
            preds = clf.predict(test_X)

            micro = f1_score(test_y, preds, average='micro')
            macro = f1_score(test_y, preds, average='macro')
            f1_mac_fold.append(macro)
            f1_mic_fold.append(micro)

        f1_mic_fold = np.array(f1_mic_fold)
        f1_mac_fold = np.array(f1_mac_fold)
        f1_mic.append(np.mean(f1_mic_fold))
        f1_mac.append(np.mean(f1_mac_fold))
        tqdm.update(1)

    f1_mac = np.array(f1_mac)
    f1_mic = np.array(f1_mic)

    print('Testing based on svm: ',
          'f1_micro=%.4f' % np.mean(f1_mic),
          'f1_macro=%.4f' % np.mean(f1_mac))
    return np.mean(f1_mic),np.mean(f1_mac)


def node_classifcation_end2end(feature, labels):
    labels = labels.reshape(-1)
    shape = len(labels.shape)
    if shape == 2:
        labels = np.argmax(labels, axis=1)
    kf = KFold(n_splits=5, shuffle=True,random_state=0)
    for train_index, test_index in kf.split(feature):
        val_index_index = np.random.choice(train_index.shape[0], int(train_index.shape[0]/10), replace=False)
        val_index = train_index[val_index_index]
        train_index=np.delete(train_index,val_index_index)
        train_X, train_y = feature[train_index], labels[train_index]
        val_X, val_y = feature[val_index], labels[val_index]
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(train_X, train_y)
        preds = clf.predict(val_X)
        micro = f1_score(val_y, preds, average='micro')
        macro = f1_score(val_y, preds, average='macro')
        return (micro+macro)/2