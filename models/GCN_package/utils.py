import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
from .normalization import fetch_normalization, row_normalize
import sys
import networkx as nx
import os
import scipy.io as scio
import math
from sklearn.metrics import f1_score


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print('Finish loading dataset')

    return adj, features, labels, idx_train, idx_val, idx_test


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def load_citation(dataset_str="cora", normalization="AugNormAdj", use_feat=True, cuda=True,semi=0):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        adj, features = preprocess_citation(adj, features, normalization)

    elif dataset_str == 'nell.0.001':
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/{}.features.npz".format(dataset_str)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                          dtype=np.int32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended)
            print("Done!")
            save_sparse_csr("data/{}.features".format(dataset_str), features)
        else:
            features = load_sparse_csr("data/{}.features.npz".format(dataset_str))
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = adj.astype(float)
        features = features.astype(float)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        adj, features = preprocess_citation(adj, features, normalization)

    else:
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    # features = torch.FloatTensor(np.array(features.todense())).float()
    if use_feat:
        features = sparse_mx_to_torch_sparse_tensor(features).float()
    else:
        features = create_sparse_eye_tensor(features.shape).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    if semi == 0:
        if dataset_str == 'cora':
            idx_train, idx_val, idx_test = range(1208), range(1208,1708), range(1708, 2708)
        elif dataset_str == 'citeseer':
            idx_train, idx_val, idx_test = range(1812), range(1812, 2312), range(2312, 3312)
        elif dataset_str == 'pubmed':
            idx_train, idx_val, idx_test = range(18217), range(18217,18717), range(18717, 19717)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    cuda = False
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))

    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def F1_score(output,labels):
    preds = output.max(1)[1].type_as(labels)
    preds=preds.data.cpu().numpy()
    labels=labels.data.cpu().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro,macro


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def create_sparse_eye_tensor(shape):
    row = np.array(range(shape[0])).astype(np.int64)
    col = np.array(range(shape[0])).astype(np.int64)
    value_ = np.ones(shape[0]).astype(float)
    indices = torch.from_numpy(np.vstack((row, col)))
    values = torch.from_numpy(value_)
    shape = torch.Size(shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citation_agcr(dataset_str="cora", normalization="AugNormAdj", use_feat=True, cuda=True):
    """
    Load Citation Networks Datasets. Prepare data for gcn
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        raw_feat = features.tocoo()

        adj, features = preprocess_citation(adj, features, normalization)

    elif dataset_str == 'nell.0.001':
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/{}.features.npz".format(dataset_str)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                          dtype=np.int32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended)
            print("Done!")
            save_sparse_csr("data/{}.features".format(dataset_str), features)
        else:
            features = load_sparse_csr("data/{}.features.npz".format(dataset_str))
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = adj.astype(float)
        features = features.astype(float)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        raw_feat = features.tocoo()

        adj, features = preprocess_citation(adj, features, normalization)

    else:
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        raw_feat = features.tocoo()

        adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    # features = torch.FloatTensor(np.array(features.todense())).float()
    # if use_feat:
    #     features = sparse_mx_to_torch_sparse_tensor(features).float()
    # else:
    #     features = create_sparse_eye_tensor(features.shape).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    edges = np.vstack((adj.row, adj.col)).astype(np.int64)
    values = adj.data
    features_ = features
    labels_ = labels.numpy()
    idx_train_ = idx_train.numpy()
    idx_val_ = idx_val.numpy()
    idx_test_ = idx_test.numpy()

    # save_path = 'data/' + dataset_str + '/{}_bipartite.pkl'.format(dataset_str)
    # with open(save_path, 'wb') as f:  # Python 3: open(..., 'wb')
    #     pkl.dump([edges, values, features_, raw_feat, labels_, idx_train_, idx_val_, idx_test_], f)

    return edges, values, features_, labels_, idx_train_, idx_val_, idx_test_

def load_citationANEmat_gac(dataset_str="BlogCatalog", semi_rate=0.1, normalization="AugNormAdj",cuda=True):
    data_file = 'data/{}/{}'.format(dataset_str, dataset_str) + '.mat'
    data = scio.loadmat(data_file)
    if dataset_str == 'ACM':
        features = data['Features']
    else:
        features = data['Attributes']
    labels = data['Label'].reshape(-1)
    adj = data['Network']
    adj, features = preprocess_citation(adj, features, normalization)
    #features = row_normalize(features)

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1

    train_idx_file = 'data/' + dataset_str + '/' + dataset_str + '_train_{}'.format(semi_rate) + '.pickle'
    valid_idx_file = 'data/' + dataset_str + '/' + dataset_str + '_valid_{}'.format(semi_rate) + '.pickle'
    test_idx_file = 'data/' + dataset_str + '/' + dataset_str + '_test_{}'.format(semi_rate) + '.pickle'
    if os.path.isfile(train_idx_file):
        with open(test_idx_file, 'rb') as f:
            idx_test = pkl.load(f)
        with open(valid_idx_file, 'rb') as f:
            idx_val = pkl.load(f)
        with open(train_idx_file, 'rb') as f:
            idx_train = pkl.load(f)
    else:
        mask = np.unique(labels)
        label_count = [np.sum(labels == v) for v in mask]
        idx_train = []
        idx_val = []
        idx_test = []
        for i, v in enumerate(mask):
            cnt = label_count[i]
            idx_all = np.where(labels == v)[0]
            np.random.shuffle(idx_all)
            idx_all = idx_all.tolist()
            test_len = math.ceil(cnt * 0.2)
            valid_len = math.ceil(cnt * 0.2)
            train_len = math.ceil(cnt * semi_rate)
            idx_test.extend(idx_all[-test_len:])
            idx_val.extend(idx_all[-(test_len + valid_len):-test_len])
            train_len_ = min(train_len, cnt - test_len - valid_len)
            idx_train.extend(idx_all[:train_len_])

        idx_train = np.array(idx_train)
        idx_val = np.array(idx_val)
        idx_test = np.array(idx_test)

        with open(train_idx_file, 'wb') as pfile:
            pkl.dump(idx_train, pfile, pkl.HIGHEST_PROTOCOL)
        with open(test_idx_file, 'wb') as pfile:
            pkl.dump(idx_test, pfile, pkl.HIGHEST_PROTOCOL)
        with open(valid_idx_file, 'wb') as pfile:
            pkl.dump(idx_val, pfile, pkl.HIGHEST_PROTOCOL)
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    features = torch.FloatTensor(np.array(features.todense())).float()
    #features = sparse_mx_to_torch_sparse_tensor(features)
    labels = torch.LongTensor(labels)
    # labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_webANEmat_gac(dataset_str="texas", normalization="AugNormAdj", semi=1, semi_rate=0.1):
    data_file = 'data/{}/{}'.format(dataset_str, dataset_str) + '.mat'
    file_train = 'data/{}/{}_train'.format(dataset_str, dataset_str) + '.pickle'
    file_valid = 'data/{}/{}_valid'.format(dataset_str, dataset_str) + '.pickle'
    file_test = 'data/{}/{}_test'.format(dataset_str, dataset_str) + '.pickle'
    data = scio.loadmat(data_file)
    features = data['Attributes']
    labels = data['Label'].reshape(-1)
    adj = data['Network']
    #features = row_normalize(features)
    adj, features = preprocess_citation(adj, features, normalization)

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1

    with open(file_test, 'rb') as f:
        idx_test = pkl.load(f)
    with open(file_valid, 'rb') as f:
        idx_val = pkl.load(f)
    with open(file_train, 'rb') as f:
        idx_train = pkl.load(f)
    if semi == 1:
        train_idx_file = 'data/' + dataset_str + '/' + dataset_str + '_train_{}'.format(semi_rate) + '.pickle'
        valid_idx_file = 'data/' + dataset_str + '/' + dataset_str + '_valid_{}'.format(semi_rate) + '.pickle'
        test_idx_file = 'data/' + dataset_str + '/' + dataset_str + '_test_{}'.format(semi_rate) + '.pickle'
        if os.path.isfile(train_idx_file):
            with open(test_idx_file, 'rb') as f:
                idx_test = pkl.load(f)
            with open(valid_idx_file, 'rb') as f:
                idx_val = pkl.load(f)
            with open(train_idx_file, 'rb') as f:
                idx_train = pkl.load(f)
        else:
            mask = np.unique(labels)
            label_count = [np.sum(labels == v) for v in mask]
            idx_train = []
            idx_val = []
            idx_test = []
            for i, v in enumerate(mask):
                cnt = label_count[i]
                idx_all = np.where(labels == v)[0]
                np.random.shuffle(idx_all)
                idx_all = idx_all.tolist()
                test_len = math.ceil(cnt * 0.2)
                valid_len = math.ceil(cnt * 0.2)
                train_len = math.ceil(cnt * semi_rate)
                idx_test.extend(idx_all[-test_len:])
                idx_val.extend(idx_all[-(test_len + valid_len):-test_len])
                train_len_ = min(train_len, cnt - test_len - valid_len)
                idx_train.extend(idx_all[:train_len_])

            idx_train = np.array(idx_train)
            idx_val = np.array(idx_val)
            idx_test = np.array(idx_test)

            with open(train_idx_file, 'wb') as pfile:
                pkl.dump(idx_train, pfile, pkl.HIGHEST_PROTOCOL)
            with open(test_idx_file, 'wb') as pfile:
                pkl.dump(idx_test, pfile, pkl.HIGHEST_PROTOCOL)
            with open(valid_idx_file, 'wb') as pfile:
                pkl.dump(idx_val, pfile, pkl.HIGHEST_PROTOCOL)

    features = sparse_mx_to_torch_sparse_tensor(features).float()
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = torch.LongTensor(labels)

    return adj, features, labels, idx_train, idx_val, idx_test

def induct_adj(adj, index):
    node, _ = adj.shape
    #index = index.data.cpu().numpy()
    adj=adj.toarray()
    adj_train = np.zeros(shape=[node, node], dtype=float)
    for i, idx_ in enumerate(index):
        #temp_adj = adj[idx_, index].toarray()
        temp_adj = adj[idx_, index]
        adj_train[idx_, index] = temp_adj
    adj_train = sp.csr_matrix(adj_train)
    return adj_train

def load_webANEmat_gac_induct(dataset_str="texas", normalization="AugNormAdj", semi=1, semi_rate=0.1):
    data_file = 'data/{}/{}'.format(dataset_str, dataset_str) + '.mat'
    file_train = 'data/{}/{}_train'.format(dataset_str, dataset_str) + '.pickle'
    file_valid = 'data/{}/{}_valid'.format(dataset_str, dataset_str) + '.pickle'
    file_test = 'data/{}/{}_test'.format(dataset_str, dataset_str) + '.pickle'
    data = scio.loadmat(data_file)
    features = data['Attributes']
    labels = data['Label'].reshape(-1)
    adj = data['Network']
    #features = row_normalize(features)
    adj, features = preprocess_citation(adj, features, normalization)

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1

    with open(file_test, 'rb') as f:
        idx_test = pkl.load(f)
    with open(file_valid, 'rb') as f:
        idx_val = pkl.load(f)
    with open(file_train, 'rb') as f:
        idx_train = pkl.load(f)
    if semi == 1:
        train_idx_file = 'data/' + dataset_str + '/' + dataset_str + '_train_{}'.format(semi_rate) + '.pickle'
        valid_idx_file = 'data/' + dataset_str + '/' + dataset_str + '_valid_{}'.format(semi_rate) + '.pickle'
        test_idx_file = 'data/' + dataset_str + '/' + dataset_str + '_test_{}'.format(semi_rate) + '.pickle'
        if os.path.isfile(train_idx_file):
            with open(test_idx_file, 'rb') as f:
                idx_test = pkl.load(f)
            with open(valid_idx_file, 'rb') as f:
                idx_val = pkl.load(f)
            with open(train_idx_file, 'rb') as f:
                idx_train = pkl.load(f)
        else:
            mask = np.unique(labels)
            label_count = [np.sum(labels == v) for v in mask]
            idx_train = []
            idx_val = []
            idx_test = []
            for i, v in enumerate(mask):
                cnt = label_count[i]
                idx_all = np.where(labels == v)[0]
                np.random.shuffle(idx_all)
                idx_all = idx_all.tolist()
                test_len = math.ceil(cnt * 0.2)
                valid_len = math.ceil(cnt * 0.2)
                train_len = math.ceil(cnt * semi_rate)
                idx_test.extend(idx_all[-test_len:])
                idx_val.extend(idx_all[-(test_len + valid_len):-test_len])
                train_len_ = min(train_len, cnt - test_len - valid_len)
                idx_train.extend(idx_all[:train_len_])

            idx_train = np.array(idx_train)
            idx_val = np.array(idx_val)
            idx_test = np.array(idx_test)

            with open(train_idx_file, 'wb') as pfile:
                pkl.dump(idx_train, pfile, pkl.HIGHEST_PROTOCOL)
            with open(test_idx_file, 'wb') as pfile:
                pkl.dump(idx_test, pfile, pkl.HIGHEST_PROTOCOL)
            with open(valid_idx_file, 'wb') as pfile:
                pkl.dump(idx_val, pfile, pkl.HIGHEST_PROTOCOL)
    adj_train = induct_adj(adj, idx_train)
    features = sparse_mx_to_torch_sparse_tensor(features).float()
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    adj_train=sparse_mx_to_torch_sparse_tensor(adj_train).float()


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = torch.LongTensor(labels)

    return adj, adj_train,features, labels, idx_train, idx_val, idx_test

def load_citationANEmat_gac_induct(dataset_str="BlogCatalog", semi_rate=0.1, normalization="AugNormAdj",cuda=True):
    data_file = 'data/{}/{}'.format(dataset_str, dataset_str) + '.mat'
    data = scio.loadmat(data_file)
    if dataset_str == 'ACM':
        features = data['Features']
    else:
        features = data['Attributes']
    labels = data['Label'].reshape(-1)
    adj = data['Network']
    adj, features = preprocess_citation(adj, features, normalization)
    #features = row_normalize(features)

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1

    train_idx_file = 'data/' + dataset_str + '/' + dataset_str + '_train_{}'.format(semi_rate) + '.pickle'
    valid_idx_file = 'data/' + dataset_str + '/' + dataset_str + '_valid_{}'.format(semi_rate) + '.pickle'
    test_idx_file = 'data/' + dataset_str + '/' + dataset_str + '_test_{}'.format(semi_rate) + '.pickle'
    if os.path.isfile(train_idx_file):
        with open(test_idx_file, 'rb') as f:
            idx_test = pkl.load(f)
        with open(valid_idx_file, 'rb') as f:
            idx_val = pkl.load(f)
        with open(train_idx_file, 'rb') as f:
            idx_train = pkl.load(f)
    else:
        mask = np.unique(labels)
        label_count = [np.sum(labels == v) for v in mask]
        idx_train = []
        idx_val = []
        idx_test = []
        for i, v in enumerate(mask):
            cnt = label_count[i]
            idx_all = np.where(labels == v)[0]
            np.random.shuffle(idx_all)
            idx_all = idx_all.tolist()
            test_len = math.ceil(cnt * 0.2)
            valid_len = math.ceil(cnt * 0.2)
            train_len = math.ceil(cnt * semi_rate)
            idx_test.extend(idx_all[-test_len:])
            idx_val.extend(idx_all[-(test_len + valid_len):-test_len])
            train_len_ = min(train_len, cnt - test_len - valid_len)
            idx_train.extend(idx_all[:train_len_])

        idx_train = np.array(idx_train)
        idx_val = np.array(idx_val)
        idx_test = np.array(idx_test)

        with open(train_idx_file, 'wb') as pfile:
            pkl.dump(idx_train, pfile, pkl.HIGHEST_PROTOCOL)
        with open(test_idx_file, 'wb') as pfile:
            pkl.dump(idx_test, pfile, pkl.HIGHEST_PROTOCOL)
        with open(valid_idx_file, 'wb') as pfile:
            pkl.dump(idx_val, pfile, pkl.HIGHEST_PROTOCOL)
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    adj_train = induct_adj(adj, idx_train)
    features = torch.FloatTensor(np.array(features.todense())).float()
    #features = sparse_mx_to_torch_sparse_tensor(features)
    labels = torch.LongTensor(labels)
    # labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    adj_train=sparse_mx_to_torch_sparse_tensor(adj_train).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, adj_train,features, labels, idx_train, idx_val, idx_test

def load_citation_induct(dataset_str="cora", normalization="AugNormAdj", use_feat=True, cuda=True,semi=0):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        adj, features = preprocess_citation(adj, features, normalization)

    elif dataset_str == 'nell.0.001':
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/{}.features.npz".format(dataset_str)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                          dtype=np.int32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended)
            print("Done!")
            save_sparse_csr("data/{}.features".format(dataset_str), features)
        else:
            features = load_sparse_csr("data/{}.features.npz".format(dataset_str))
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = adj.astype(float)
        features = features.astype(float)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        adj, features = preprocess_citation(adj, features, normalization)

    else:
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    # features = torch.FloatTensor(np.array(features.todense())).float()
    if use_feat:
        features = sparse_mx_to_torch_sparse_tensor(features).float()
    else:
        features = create_sparse_eye_tensor(features.shape).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]


    if semi == 0:
        if dataset_str == 'cora':
            idx_train, idx_val, idx_test = range(1208), range(1208,1708), range(1708, 2708)
        elif dataset_str == 'citeseer':
            idx_train, idx_val, idx_test = range(1812), range(1812, 2312), range(2312, 3312)
        elif dataset_str == 'pubmed':
            idx_train, idx_val, idx_test = range(18217), range(18217,18717), range(18717, 19717)

    adj_train = induct_adj(adj, idx_train)

    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    adj_train=sparse_mx_to_torch_sparse_tensor(adj_train).float()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    cuda = False
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, adj_train,features, labels, idx_train, idx_val, idx_test
