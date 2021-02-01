from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import f1_score

from preprocessing.utils import normalize, sigmoid, load_citation, sparse_mx_to_torch_sparse_tensor, load_citationmat
from model.GCN.ZX_GCN import GCNTra
from preprocessing.preprocessing import mask_test_edges

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='0', help='specify cuda devices')
parser.add_argument('--dataset',type=str,default='cora',
                    #default="cora",
                    help='BlogCatalog.')
parser.add_argument('--model_type', type=str, default="gcn",
                    help='Dataset to use.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--use_feat', type=int, default=1,
                    help='Use attribute or not')
parser.add_argument('--patience', type=int, default=200,
                    help='Use attribute or not')
parser.add_argument('--use_cpu', type=int, default=0,
                    help='Use attribute or not')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                    choices=['AugNormAdj'],
                    help='Normalization method for the adjacency matrix.')


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


def train(features, adj, adj_label, val_edges, val_edges_false, save_path, device, args, pos_weight, norm):
    model = GCNTra(nfeat=features.shape[1],
                nhid=args.hidden1,
                nhid2=args.hidden2,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = pos_weight.to(device)
    b_xent = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_label).to_dense()
    adj_label = adj_label.to(device)

    max_auc = 0.0
    max_ap = 0.0
    best_epoch = 0
    cnt_wait = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        emb = model(features, adj)
        logits = model.pred_logits(emb)
        # loss = loss_function_entropy(preds=logits, labels=adj_label,
        #                              norm=norm, pos_weight=pos_weight)

        loss = b_xent(logits, adj_label)
        loss = loss * norm
        loss_train = loss.item()

        auc_, ap_ = get_roc_score(model, features, adj, val_edges, val_edges_false)
        if auc_ > max_auc:
            max_auc = auc_
            max_ap = ap_
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            cnt_wait += 1

        print('Epoch %d / %d' % (epoch, args.epochs),
              'current_best_epoch: %d' % best_epoch,
              'train_loss: %.4f' % loss_train,
              'valid_acu: %.4f' % auc_,
              'valid_ap: %.4f' % ap_)

        if cnt_wait == args.patience and best_epoch != 0:
            print('Early stopping!')
            break

        loss.backward()
        optimizer.step()

    print('!!! Training finished',
          'best_epoch: %d' % best_epoch,
          'best_auc: %.4f' % max_auc,
          'best_ap: %.4f' % max_ap)

    model.load_state_dict(torch.load(save_path))
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


def test_classify(feature, labels, args):
    shape = len(labels.shape)
    if shape == 2:
        labels = np.argmax(labels, axis=1)
    f1_mac = []
    f1_mic = []
    kf = KFold(n_splits=5, random_state=args.seed, shuffle=True)
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
    print('Testing based on svm: ',
          'f1_micro=%.4f' % f1_mic,
          'f1_macro=%.4f' % f1_mac)


if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        cuda_name = 'cuda:' + args.cuda
        device = torch.device(cuda_name)
        print('--> Use GPU %s' % args.cuda)
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")
        print("--> No GPU")

    if args.use_cpu:
        device = torch.device("cpu")

    print('---> Loading dataset...')
    if args.dataset == 'BlogCatalog':
        adj, features, labels, idx_train, idx_val, idx_test = load_citationmat(args.dataset, args.normalization,
                                                                            args.use_feat,
                                                                            args.cuda)
    elif args.dataset == 'Flickr':
        adj, features, labels, idx_train, idx_val, idx_test = load_citationmat(args.dataset, args.normalization,
                                                                               args.use_feat, args.cuda)
    else:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.use_feat,
                                                                            args.cuda)
    print('--->Generate train/valid links for unsupervised learning...')
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing and transfer to tensor
    adj_norm = normalize(adj + sp.eye(adj_train.shape[0]))
    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)

    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    save_path = "weights/%s_" % args.model_type + args.dataset + '_%d_' % args.hidden1 + '_%d_' % args.hidden2 + '.pth'
    save_path_emb = "./emb/%s_" % args.model_type + args.dataset + '_%d_' % args.hidden1 + '_%d_' % args.hidden2 + '.mat'
    print_configuration(args)
    node_emb = train(features, adj_norm, adj_label, val_edges, val_edges_false, save_path, device, args, pos_weight, norm)

    # save_emb(node_emb, adj_orig + sp.eye(adj_orig.shape[0]), labels, save_path_emb)
    test_classify(node_emb, labels, args)
    print('!!! Finish')







