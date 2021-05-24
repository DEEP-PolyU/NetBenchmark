from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import f1_score
# from optimizer import loss_function_entropy
from models.dgi_package import process
from preprocessing.preprocessing import mask_test_edges_fast
from hyperopt import hp
from models.GAE_package import GCN,GCNTra
from .model import *





def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_roc_score(net, features, adj, edges_pos, edges_neg):

    net.eval()
    emb = net(features, adj)
    emb = emb.data.cpu().numpy()

    adj_rec = np.matmul(emb, emb.T)

    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def mat_import(mat):

    adj, features, labels, idx_train, idx_val, idx_test = process.load_citationmat_feature(mat)
    # features = process.sparse_mx_to_torch_sparse_tensor(features).float()
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = process.mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing and transfer to tensor
    adj_norm = process.row_normalize(adj + sp.eye(adj_train.shape[0]))
    adj_norm = process.sparse_mx_to_torch_sparse_tensor(adj_norm)

    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    adj_label = adj_train + sp.eye(adj_train.shape[0])

    return features, adj_norm, adj_label, val_edges, val_edges_false, test_edges, test_edges_false, norm, pos_weight

def train(features, adj, adj_label, val_edges, val_edges_false, device, pos_weight, norm,hid1,hid2,dropout,lr,weight_decay,epochs,**kwargs):

    model = GCNTra(nfeat=features.shape[1],
                nhid=hid1,
                nhid2=hid2,
                dropout=dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    pos_weight = pos_weight.to(device)
    b_xent = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    adj_label = process.sparse_mx_to_torch_sparse_tensor(adj_label).to_dense()
    adj_label = adj_label.to(device)

    max_auc = 0.0
    max_ap = 0.0
    best_epoch = 0
    cnt_wait = 0
    start_time = time.time()
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        emb = model(features, adj)
        logits = model.pred_logits(emb)

        loss = b_xent(logits, adj_label)
        loss = loss * norm
        loss_train = loss.item()

        auc_, ap_ = get_roc_score(model, features, adj, val_edges, val_edges_false)
        if auc_ > max_auc:
            max_auc = auc_
            max_ap = ap_
            best_epoch = epoch
            cnt_wait = 0
            # torch.save(model.state_dict(), save_path)
        else:
            cnt_wait += 1

        print('Epoch %d / %d' % (epoch, epochs),
              'current_best_epoch: %d' % best_epoch,
              'train_loss: %.4f' % loss_train,
              'valid_acu: %.4f' % auc_,
              'valid_ap: %.4f' % ap_)

        if cnt_wait == 2800 and best_epoch != 0:
            print('Early stopping!')
            break




        loss.backward()
        optimizer.step()

    # model.load_state_dict(torch.load(save_path))
    model.eval()
    emb = model(features, adj)


    return emb.data.cpu().numpy()


def save_emb(emb, adj, label, save_emb_path):
    emb = sp.csr_matrix(emb)
    sio.savemat(save_emb_path, {'feature': emb, 'label': label, 'adj': adj})
    print('---> Embedding saved on %s' % save_emb_path)


def print_configuration(args):
    print('--> Experiment configuration')
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))




class GAE(Models):

    def is_preprocessing(cls):
        return False


    @classmethod
    def is_epoch(cls):
        return False
    @classmethod
    def is_deep_model(cls):
        return True


    def check_train_parameters(self):

        space_dtree = {

            'batch_size': hp.uniformint('batch_size', 1, 100),
            'epochs': hp.uniformint('epochs', 100, 5000),
            'lr': hp.loguniform('lr', np.log(0.05), np.log(0.2)),
            'dropout': hp.uniform('dropout', 0, 1),
            'evaluation': str(self.evaluation)
        }

        return space_dtree

    def train_model(self, **kwargs):

        if self.use_gpu:
            device = self.device
            torch.cuda.manual_seed(42)
        else:
            device = self.device
            print("--> No GPU")

        # link_predic_result_file = "result/GAE_{}.res".format('datasets')
        # embedding_node_mean_result_file = "result/GAE_{}_n_mu.emb".format('datasets')
        # embedding_attr_mean_result_file = "result/GAE_{}_a_mu.emb".format('datasets')
        # embedding_node_var_result_file = "result/GAE_{}_n_sig.emb".format('datasets')
        # embedding_attr_var_result_file = "result/GAE_{}_a_sig.emb".format('datasets')

        #dropout =0.6 ,lr = 0.001,weight_decay = 0,epochs = 1000
        dropout = 0
        # lr = 0.001
        weight_decay = 0
        # epochs = 2000


        features, adj_norm, adj_label, val_edges, val_edges_false, test_edges, test_edges_false, norm, pos_weight = mat_import(self.mat_content)
        #features, adj, adj_label, val_edges, val_edges_false, save_path, device, pos_weight, norm,hid1,hid2,dropout,lr,weight_decay,epochs
        embeding = train(features=features, adj = adj_norm, adj_label = adj_label, val_edges = val_edges, val_edges_false = val_edges_false,  device=device, pos_weight=pos_weight, norm=norm,
                         hid1= 256,hid2 = 128,weight_decay = weight_decay,**kwargs)



        return embeding







