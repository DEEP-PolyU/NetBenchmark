from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from .GCN_package.utils import load_data, accuracy, load_citation,load_citationANEmat_gac,load_webANEmat_gac,F1_score
from .GCN_package.input_graph_feed import GraphInput
from .GAT_package.models import GAT, SpGAT
from .model import *
from preprocessing.preprocessing import load_normalized_format
import os
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
# This is the graphsage version of GCN_package paper

class GATModel(Models):

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
        space_dtree = {

            'batch_size': hp.uniformint('batch_size', 1, 100),
            'nb_epochs': hp.uniformint('nb_epochs', 100, 120),
            'lr': hp.loguniform('lr', np.log(0.05), np.log(0.2)), # walk_length,window_size
            'dropout': hp.uniform('dropout', 0, 0.75),
            # 'nb_heads': hp.uniformint('nb_heads',1,10),
            'evaluation': str(self.evaluation)
        }

        return space_dtree

    def train_model(self, **kwargs):

        semi=0
        seed=42
        hidden=128
        dropout=kwargs["dropout"]
        lr=kwargs["lr"]
        weight_decay=0
        epochs=int(kwargs["nb_epochs"])
        semi_rate=0.6
        nb_heads = 8 #int(kwargs['nb_heads'])
        alpha = 0.2

        sparse = False
        np.random.seed(seed)

        if self.use_gpu:
            device = self.device
            torch.cuda.manual_seed(42)
        else:
            device = self.device
            print("--> No GPU")
        fastmode = False
        # Load data
        # adj, features, labels, idx_train, idx_val, idx_test = load_data()
        adj, features, labels, idx_train, idx_val, idx_test = load_normalized_format(datasets=self.mat_content,semi_rate=semi_rate)

        # Model and optimizer
        if sparse == True:
            model = SpGAT(nfeat=features.shape[1],
                          nhid=hidden,
                          nclass=int(labels.max()) + 1,
                          dropout=dropout,
                          nheads=nb_heads,
                          alpha=alpha)
        else:
            model = GAT(nfeat=features.shape[1],
                        nhid=hidden,
                        nclass=int(labels.max()) + 1,
                        dropout=dropout,
                        nheads=nb_heads,
                        alpha=alpha)
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=weight_decay)

        model.to(device)
        features = features.to(device)
        adj = adj.to_dense().to(device)
        labels = labels.to(device)


        # idx_train = idx_train.to(device)
        # idx_val = idx_val.to(device)
        # idx_test = idx_test.to(device)
        features, adj, labels = Variable(features), Variable(adj), Variable(labels)

        def train(epochs, idx_train):
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
        kf = KFold(n_splits=5, random_state=seed, shuffle=True)
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
        F1_mic_tot = np.array(F1_mic_tot)
        F1_mac_tot = np.array(F1_mac_tot)
        F1_mic_mean = np.mean(F1_mic_tot)
        F1_mac_mean = np.mean(F1_mac_tot)
        print('F1_mic:', F1_mic_mean)
        print('F1_mac:', F1_mac_mean)


        return F1_mic_mean,F1_mac_mean
