from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from .GCN_package.utils import load_data, accuracy, load_citation,load_citationANEmat_gac,load_webANEmat_gac,F1_score
from .GCN_package.input_graph_feed import GraphInput
from .GCN_package.models import GCN_original
from .model import *
from preprocessing.preprocessing import load_normalized_format
import os
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
# This is the graphsage version of GCN_package paper

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')


class GCN(Models):

    @classmethod
    def is_preprocessing(cls):
        return False

    @classmethod
    def is_deep_model(cls):
        return True

    def deep_algo(self,stop_time):

        semi=0
        seed=42
        hidden=128
        dropout=0.5
        lr=0.01
        weight_decay=0
        epochs=200
        semi_rate=0.6
        cuda=use_gpu

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        fastmode = False
        # Load data
        # adj, features, labels, idx_train, idx_val, idx_test = load_data()
        adj, features, labels, idx_train, idx_val, idx_test=load_normalized_format(datasets=self.mat_content,semi_rate=semi_rate,cuda=cuda)

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
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  # 'loss_val: {:.4f}'.format(loss_val.item()),
                  # 'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

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
        start_time = time.time()
        for train_index, test_index in kf.split(features):
            train_index = torch.LongTensor(train_index)
            test_index = torch.LongTensor(test_index)
            train_index.to(device)
            test_index.to(device)
            for epoch in range(epochs):
                train(epoch, train_index)
                if (time.time() - start_time) >= stop_time:  # Change in Time stoping
                    print('times up,Time setting is: {:.2f}'.format(time.time() - start_time))
                    break
            print("Optimization Finished!")
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            # Testing
            F1_mic, F1_mac = test(test_index, labels)
            F1_mic_tot.append(F1_mic)
            F1_mac_tot.append(F1_mac)
        print(F1_mic_tot)
        print(F1_mac_tot)
        print("Optimization Finished!")
        output = model(features, adj)


        if use_gpu:
            output = output.cpu()

        return output.data.numpy()
