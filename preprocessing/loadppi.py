import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
import json
from networkx.readwrite import json_graph
import pdb
from sklearn.preprocessing import StandardScaler


sys.setrecursionlimit(99999)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum == 0) * 1 + rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# adapted from PetarV/GAT
def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        # for v in range(nb_nodes):
        for v in adj[u, :].nonzero()[1]:
            # if adj[u,v]== 1:
            run_dfs(adj, msk, v, ind, nb_nodes)


def dfs_split(adj):
    # Assume adj is of shape [nb_nodes, nb_nodes]
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0

    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1

    return ret


def test(adj, mapping):
    nb_nodes = adj.shape[0]
    for i in range(nb_nodes):
        # for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] != mapping[j]:
                #  if adj[i,j] == 1:
                return False
    return True


def find_split(adj, mapping, ds_label):
    nb_nodes = adj.shape[0]
    dict_splits = {}
    for i in range(nb_nodes):
        # for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] == 0 or mapping[j] == 0:
                dict_splits[0] = None
            elif mapping[i] == mapping[j]:
                if ds_label[i]['val'] == ds_label[j]['val'] and ds_label[i]['test'] == ds_label[j]['test']:

                    if mapping[i] not in dict_splits.keys():
                        if ds_label[i]['val']:
                            dict_splits[mapping[i]] = 'val'

                        elif ds_label[i]['test']:
                            dict_splits[mapping[i]] = 'test'

                        else:
                            dict_splits[mapping[i]] = 'train'

                    else:
                        if ds_label[i]['test']:
                            ind_label = 'test'
                        elif ds_label[i]['val']:
                            ind_label = 'val'
                        else:
                            ind_label = 'train'
                        if dict_splits[mapping[i]] != ind_label:
                            print('inconsistent labels within a graph exiting!!!')
                            return None
                else:
                    print('label of both nodes different, exiting!!')
                    return None
    return dict_splits


def index_to_mask(index, size):
    mask = torch.full((size,), False, dtype=torch.bool)
    mask[index] = True
    return mask

def load_saintdata(dataset_name):


    adj_full = sp.load_npz('data/{}/adj_full.npz'.format(dataset_name)).astype(np.bool)
    adj_train = sp.load_npz('data/{}/adj_train.npz'.format(dataset_name)).astype(np.bool)
    role = json.load(open('data/{}/role.json'.format(dataset_name)))
    feats = np.load('data/{}/feats.npy'.format(dataset_name))

    class_map = json.load(open('data/{}/class_map.json'.format(dataset_name)))

    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]

    # for key, value in id_map.items():
    #     id_map[key] = [value]
    # print (len(id_map))




    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k, v in class_map.items():
            class_arr[k][v - offset] = 1

    new_label = np.argmax(class_arr, axis=1)


    adj_full = sp.csc_matrix(adj_full)
    feats = sp.csc_matrix(feats)

    return adj_full, adj_train, feats, new_label, role