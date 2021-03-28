from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from .GCN_package.utils import load_data, accuracy, load_citation,sparse_mx_to_torch_sparse_tensor
from .GCN_package.input_graph_feed import GraphInput
from .GCN_package.models import GCN_original
from .model import *
from preprocessing.preprocessing import load_normalized_format
import os
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

        seed=42
        hidden=64
        dropout=0.5
        lr=0.01
        weight_decay=5e-4
        epochs=2000
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
        adj=adj.to(device)
        features=features.to(device)
        labels=labels.to(device)

        def train(epoch):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if not fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output = model(features, adj)

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

        start_time = time.time()
        for epoch in range(epochs):
            train(epoch)
            # early stop by time
            if (time.time() - start_time) >= stop_time:  # Change in Time stoping
                print('times up,Time setting is: {:.2f}'.format(time.time() - start_time))
                break
        print("Optimization Finished!")
        output = model(features, adj)
        # roc_score, ap_score =  self.get_roc_score(test_edges, test_edges_false, preds_sub_u)
        # roc_score_a, ap_score_a =  self.get_roc_score_a(test_feas, test_feas_false, preds_sub_a)

        if use_gpu:
            output = output.cpu()

        return output.data.numpy()
