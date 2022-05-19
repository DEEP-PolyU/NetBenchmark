from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from .GCN_package.utils import load_data, accuracy, load_citation,load_citationANEmat_gac,load_webANEmat_gac,F1_score

from .GCN_package.models import GCN_original
from .model import *
from preprocessing.preprocessing import load_normalized_format
from hyperparameters.public_hyper import SPACE_TREE
# This is the graphsage version of GCN_package paper

class GCN(Models):

    @classmethod
    def is_preprocessing(cls):
        return False
    @classmethod
    def is_deep_model(cls):
        return False
    @classmethod
    def is_end2end(cls):
        return True

    def check_train_parameters(self):
        space_dtree={}
        space_dtree['lr'] = SPACE_TREE['lr']
        space_dtree["dropout"]=SPACE_TREE["dropout"]
        space_dtree["nb_epochs"]=SPACE_TREE["nb_epochs"]
        return space_dtree

    def train_model(self, **kwargs):
        seed=42
        hidden=128
        dropout=kwargs["dropout"]
        lr = kwargs["lr"]
        weight_decay=0
        epochs=int(kwargs["nb_epochs"])
        semi_rate=0.6

        np.random.seed(seed)

        adj, features, labels, idx_train, idx_val, idx_test=load_normalized_format(datasets=self.mat_content,semi_rate=semi_rate)

        if self.use_gpu:
            device = self.device
            torch.cuda.manual_seed(42)
            adj = adj.to(device)
            labels = labels.to(device)
            features = features.to(device)
        else:
            device = self.device
            # print("--> No GPU")
        # Model and optimizer
        model = GCN_original(nfeat=features.shape[1],
                    nhid=hidden,
                    nclass=labels.max().item() + 1,
                    dropout=dropout)
        optimizer = optim.Adam(model.parameters(),
                               lr=lr, weight_decay=weight_decay)

        model.to(device)
        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)

        # idx_train = idx_train.to(device)
        # idx_val = idx_val.to(device)
        # idx_test = idx_test.to(device)

        def train(epoch, idx_train):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            # if not fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            #    model.eval()
            #    output = model(features, adj)

            # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            # acc_val = accuracy(output[idx_val], labels[idx_val])
            # print('Epoch: {:04d}'.format(epoch + 1),
            #       'loss_train: {:.4f}'.format(loss_train.item()),
            #       'acc_train: {:.4f}'.format(acc_train.item()),
            #       # 'loss_val: {:.4f}'.format(loss_val.item()),
            #       # 'acc_val: {:.4f}'.format(acc_val.item()),
            #       'time: {:.4f}s'.format(time.time() - t))

        def test(idx_test, labels):
            model.eval()
            output = model(features, adj)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            micro, macro = F1_score(output[idx_test], labels[idx_test])
            return micro, macro

        # if __name__ == '__main__':
        # Train model
        kf = KFold(n_splits=5, shuffle=True,random_state=0)
        t_total = time.time()
        F1_mic_tot = []
        F1_mac_tot = []
        for train_index, test_index in kf.split(features):
            val_index_index = np.random.choice(train_index.shape[0], int(train_index.shape[0] / 10), replace=False)
            val_index = train_index[val_index_index]
            train_index = np.delete(train_index, val_index_index)

            train_index = torch.LongTensor(train_index)
            val_index = torch.LongTensor(val_index)
            train_index.to(device)
            val_index.to(device)
            for epoch in range(epochs):
                train(epoch, train_index)
            # print("Optimization Finished!")
            # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            # Testing
            F1_mic, F1_mac = test(val_index, labels)
            return (F1_mic+F1_mic)/2
    def get_best_result(self, **kwargs):
        seed=42
        hidden=128
        dropout=kwargs["dropout"]
        # lr=np.power(10,-4*kwargs["lr"])
        lr = kwargs["lr"]

        weight_decay=0
        epochs=int(kwargs["nb_epochs"])
        semi_rate=0.6

        np.random.seed(seed)

        adj, features, labels, idx_train, idx_val, idx_test=load_normalized_format(datasets=self.mat_content,semi_rate=semi_rate)

        if self.use_gpu:
            device = self.device
            torch.cuda.manual_seed(42)
            adj = adj.to(device)
            labels = labels.to(device)
            features = features.to(device)
        else:
            device = self.device
            print("--> No GPU")
        # Model and optimizer
        model = GCN_original(nfeat=features.shape[1],
                    nhid=hidden,
                    nclass=labels.max().item() + 1,
                    dropout=dropout)
        optimizer = optim.Adam(model.parameters(),
                               lr=lr, weight_decay=weight_decay)

        model.to(device)
        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)

        # idx_train = idx_train.to(device)
        # idx_val = idx_val.to(device)
        # idx_test = idx_test.to(device)

        def train(epoch, idx_train):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()


        def test(idx_test, labels):
            model.eval()
            output = model(features, adj)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            micro, macro = F1_score(output[idx_test], labels[idx_test])
            return micro, macro


        kf = KFold(n_splits=5, shuffle=True,random_state=0)
        t_total = time.time()
        F1_mic_tot = []
        F1_mac_tot = []
        for train_index, test_index in kf.split(features):
            train_index = torch.LongTensor(train_index)
            test_index = torch.LongTensor(test_index)
            train_index.to(device)
            test_index.to(device)
            for epoch in range(epochs):
                train(epoch, train_index)
            print("Optimization Finished!")
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            # Testing
            F1_mic, F1_mac = test(test_index, labels)
            F1_mic_tot.append(F1_mic)
            F1_mac_tot.append(F1_mac)
        return np.mean(F1_mic_tot),np.mean(F1_mac_tot)
