from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
import time
import os
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import scipy.io
import argparse
from .CAN_package.optimizer import  OptimizerCAN
from .CAN_package.model import CAN_original
from .CAN_package.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, mask_test_feas
from .model import *


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
use_gpu = torch.cuda.is_available()
device = torch.device('cpu')



class CAN(Models):

    num_nodes=None
    adj_orig=None
    num_features=None
    features_orig=None

    @classmethod
    def is_preprocessing(cls):
        return False

    @classmethod
    def is_deep_model(cls):
        return True

    def get_roc_score(self,edges_pos, edges_neg, preds_sub_u):
        def sigmoid(x):
            x = np.clip(x, -500, 500)
            return 1.0 / (1 + np.exp(-x))

        # Predict on test set of edges
        # adj_rec = sess.run(model.reconstructions[0], feed_dict=feed_dict).reshape([self.num_nodes, self.num_nodes])
        if (use_gpu):
            adj_rec = preds_sub_u.view(self.num_nodes, self.num_nodes).cpu().data.numpy()
        else:
            adj_rec = preds_sub_u.view(self.num_nodes, self.num_nodes).data.numpy()
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(self.adj_orig[e[0], e[1]])
        # print(np.min(adj_rec))
        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(self.adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg]).astype(np.float)
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_roc_score_a(self,feas_pos, feas_neg, preds_sub_a):


        def sigmoid(x):
            x = np.clip(x, -500, 500)
            return 1.0 / (1 + np.exp(-x))

        # Predict on test set of edges
        # fea_rec = sess.run(model.reconstructions[1], feed_dict=feed_dict).reshape([self.num_nodes, self.num_features])
        if (use_gpu):
            fea_rec = preds_sub_a.view(self.num_nodes, self.num_features).cpu().data.numpy()
        else:
            fea_rec = preds_sub_a.view(self.num_nodes, self.num_features).data.numpy()
        preds = []
        pos = []
        for e in feas_pos:
            preds.append(sigmoid(fea_rec[e[0], e[1]]))
            pos.append(self.features_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in feas_neg:
            preds_neg.append(sigmoid(fea_rec[e[0], e[1]]))
            neg.append(self.features_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def weighted_cross_entropy_with_logits(self, logits, targets, pos_weight):
        logits = logits.clamp(-10, 10)
        return targets * -torch.log(torch.sigmoid(logits)) * pos_weight + (1 - targets) * -torch.log(
            1 - torch.sigmoid(logits))

    def deep_algo(self,stop_time):
        learning_rate=0.01
        hidden1=256
        hidden2=128
        dropout=0
        epochs=200
        seed=42

        np.random.seed(seed)
        torch.manual_seed(seed)

        adj=self.mat_content['Network']
        features=self.mat_content['Attributes']
        self.adj_orig = adj
        self.adj_orig = self.adj_orig - sp.dia_matrix((self.adj_orig.diagonal()[np.newaxis, :], [0]), shape=self.adj_orig.shape)
        self.adj_orig.eliminate_zeros()
        print('--->Generate train/valid links for unsupervised learning...')
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false = mask_test_feas(features)

        adj = adj_train
        self.features_orig = features
        features = sp.lil_matrix(features)


        adj_norm = preprocess_graph(adj)

        self.num_nodes = adj.shape[0]
        features = sparse_to_tuple(features.tocoo())
        self.num_features = features[2][1]
        features_nonzero = features[1].shape[0]
        print(features[1].shape)
        # Create model
        print('--->Create model...')
        # args can be one parameter
        # 创建model这里的参数是传到CAN的构造函数(init)处
        model = CAN_original(hidden1, hidden2, self.num_features, self.num_nodes, features_nonzero, dropout).to(device)
        pos_weight_u = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm_u = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        pos_weight_a = float(features[2][0] * features[2][1] - len(features[1])) / len(features[1])
        norm_a = features[2][0] * features[2][1] / float((features[2][0] * features[2][1] - len(features[1])) * 2)
        # Optimizer

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        cost_val = []
        acc_val = []
        val_roc_score = []

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        features_label = sparse_to_tuple(self.features_orig)
        '''sparse_to_tuple返回三个参数，第1个的index，即指明哪一行哪一列有指，第2个就是具体的value，第三个是矩阵的维度
        这三个参数刚好是torch.sparse将稠密矩阵转成稀疏矩阵的参数，具体可以查官网文档'''
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0]).t(), torch.FloatTensor(features[1]),
                                            features[2]).to(device)
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].astype(np.int32)).t(),
                                            torch.FloatTensor(adj_norm[1]), adj_norm[2]).to(device)
        # Train model
        print('--->Train model...')
        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            t = time.time()
            optimizer.zero_grad()
            # Get result
            '''train model这里参数传到CAN的forward处，features即论文模型图中的X矩阵(结点属性矩阵)，adj_norm是模型中的A矩阵，即邻接矩阵'''
            preds_sub_u, preds_sub_a, z_u_mean, z_u_log_std, z_a_mean, z_a_log_std = model(features, adj_norm)

            labels_sub_u = torch.from_numpy(self.adj_orig.toarray()).flatten().float().to(device)
            labels_sub_a = torch.from_numpy(self.features_orig.toarray()).flatten().float().to(device)
            cost_u = norm_u * torch.mean(
                self.weighted_cross_entropy_with_logits(logits=preds_sub_u, targets=labels_sub_u, pos_weight=pos_weight_u))
            cost_a = norm_a * torch.mean(
                self.weighted_cross_entropy_with_logits(logits=preds_sub_a, targets=labels_sub_a, pos_weight=pos_weight_a))

            # Latent loss
            log_lik = cost_u + cost_a
            kl_u = (0.5) * torch.mean(
                (1 + 2 * z_u_log_std - z_u_mean.pow(2) - (torch.exp(2 * z_u_log_std))).sum(1)) / self.num_nodes
            kl_a = (0.5) * torch.mean(
                (1 + 2 * z_a_log_std - (z_a_mean.pow(2)) - (torch.exp(2 * z_a_log_std))).sum(1)) / self.num_features
            kl = kl_u + kl_a
            cost = log_lik - kl
            correct_prediction_u = torch.sum(
                torch.eq(torch.ge(torch.sigmoid(preds_sub_u), 0.5).float(), labels_sub_u).float()) / len(labels_sub_u)
            correct_prediction_a = torch.sum(
                torch.eq(torch.ge(torch.sigmoid(preds_sub_a), 0.5).float(), labels_sub_a).float()) / len(labels_sub_a)
            accuracy = torch.mean(correct_prediction_u + correct_prediction_a)

            # Compute average loss
            avg_cost = cost
            avg_accuracy = accuracy
            log_lik = log_lik
            kl = kl
            roc_curr, ap_curr = self.get_roc_score(val_edges, val_edges_false, preds_sub_u)
            roc_curr_a, ap_curr_a =  self.get_roc_score_a(val_feas, val_feas_false, preds_sub_a)
            val_roc_score.append(roc_curr)



            # early stop by time
            if (time.time() - start_time) >= stop_time:  # Change in Time stoping
                print('times up,Time setting is: {:.2f}'.format(time.time() - start_time))
                break


            # Run backward
            avg_cost.backward()
            optimizer.step()

            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost),
                  "log_lik=", "{:.5f}".format(log_lik),
                  "KL=", "{:.5f}".format(kl),
                  "train_acc=", "{:.5f}".format(avg_accuracy),
                  "val_edge_roc=", "{:.5f}".format(val_roc_score[-1]),
                  "val_edge_ap=", "{:.5f}".format(ap_curr),
                  "val_attr_roc=", "{:.5f}".format(roc_curr_a),
                  "val_attr_ap=", "{:.5f}".format(ap_curr_a),
                  "time=", "{:.5f}".format(time.time() - t))

        #print("Optimization Finished!")

        preds_sub_u, preds_sub_a, z_u_mean, z_u_log_std, z_a_mean, z_a_log_std = model(features, adj_norm)
        # roc_score, ap_score =  self.get_roc_score(test_edges, test_edges_false, preds_sub_u)
        # roc_score_a, ap_score_a =  self.get_roc_score_a(test_feas, test_feas_false, preds_sub_a)

        if use_gpu:
            z_u_mean = z_u_mean.cpu()

        return z_u_mean.data.numpy()

