import numpy as np
import scipy.sparse as sp
import torch
import random

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def mask_test_edges_net(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    num_node = adj.shape[0]
    sample_size = int((len(train_edges) * 1.0) / num_node)
    sample_size = sample_size + 10

    neg_list = []
    all_candiate = set(range(adj.shape[0]))
    for i in range(0, num_node):
        non_zeros = adj[i].nonzero()[1]
        neg_candi = np.array(list(all_candiate.difference(set(non_zeros))))
        if len(neg_candi) >= sample_size:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=False)
        elif len(neg_candi) == 0:
            pass
        else:
            neg_candi = neg_candi

        neg_candi = [[i, j] for j in neg_candi]
        # print('len_: %d' % len(neg_candi))
        neg_list.extend(neg_candi)
    # neg_list = np.array(neg_list)

    train_edges_false = np.array(random.sample(neg_list, len(train_edges)))

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def mask_test_edges_sample(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    # test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    # test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    # train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    train_edges = np.delete(edges, val_edge_idx, axis=0)

    sample_size = int((len(train_edges) * 1.0) / adj.shape[0])
    sample_size = sample_size * 10

    neg_list = []
    all_candiate = np.array(range(adj.shape[1]))
    for i in range(adj.shape[0]):
        non_zeros = adj[i].nonzero()[1]
        neg_candi = np.delete(all_candiate, non_zeros.tolist() + [i], axis=0)
        if len(neg_candi) < sample_size:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=True)
        else:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=False)
        neg_candi = [[i, j] for j in neg_candi]
        neg_list.extend(neg_candi)
    neg_list = np.array(neg_list)

    # def ismember(a, b, tol=5):
    #     rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    #     return np.any(rows_close)

    if neg_list.shape[0] < len(train_edges):
        idx = np.random.choice(len(neg_list), len(train_edges), replace=True)
    else:
        idx = np.random.choice(len(neg_list), len(train_edges), replace=False)
    idx_test = np.random.choice(len(neg_list), len(val_edges), replace=False)
    # train_edges_false = np.array(random.sample(neg_list, len(train_edges) * neg_num))
    # val_edges_false = np.array(random.sample(neg_list, len(val_edges)))
    train_edges_false = np.array(neg_list[idx])
    val_edges_false = np.array(neg_list[idx_test])
    # while len(val_edges_false) < len(val_edges):
    #     idx_i = np.random.randint(0, adj.shape[0])
    #     idx_j = np.random.randint(0, adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], train_edges):
    #         continue
    #     if ismember([idx_j, idx_i], train_edges):
    #         continue
    #     if ismember([idx_i, idx_j], val_edges):
    #         continue
    #     if ismember([idx_j, idx_i], val_edges):
    #         continue
    #     if val_edges_false:
    #         if ismember([idx_j, idx_i], np.array(val_edges_false)):
    #             continue
    #         if ismember([idx_i, idx_j], np.array(val_edges_false)):
    #             continue
    #     val_edges_false.append([idx_i, idx_j])
    #
    # # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    test_edges = []
    test_edges_false = []

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, [train_edges, np.array(train_edges_false)], val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_edges_bipart(adj, num_node):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    # test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    # test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    # train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    train_edges = np.delete(edges, val_edge_idx, axis=0)

    sample_size = int((len(train_edges) * 1.0) / num_node)
    sample_size = sample_size + 10

    neg_list = []
    all_candiate = set(range(num_node, adj.shape[0]))
    for i in range(0, num_node):
        non_zeros = adj[i].nonzero()[1]
        neg_candi = np.array(list(all_candiate.difference(set(non_zeros))))
        if len(neg_candi) >= sample_size:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=False)
        elif len(neg_candi) == 0:
            pass
        else:
            neg_candi = neg_candi

        neg_candi = [[i, j] for j in neg_candi]
        # print('len_: %d' % len(neg_candi))
        neg_list.extend(neg_candi)
    # neg_list = np.array(neg_list)

    train_edges_false = np.array(random.sample(neg_list, len(train_edges)))
    val_edges_false = np.array(random.sample(neg_list, len(val_edges)))
    # while len(val_edges_false) < len(val_edges):
    #     idx_i = np.random.randint(0, adj.shape[0])
    #     idx_j = np.random.randint(0, adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], train_edges):
    #         continue
    #     if ismember([idx_j, idx_i], train_edges):
    #         continue
    #     if ismember([idx_i, idx_j], val_edges):
    #         continue
    #     if ismember([idx_j, idx_i], val_edges):
    #         continue
    #     if val_edges_false:
    #         if ismember([idx_j, idx_i], np.array(val_edges_false)):
    #             continue
    #         if ismember([idx_i, idx_j], np.array(val_edges_false)):
    #             continue
    #     val_edges_false.append([idx_i, idx_j])
    #
    # # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    test_edges = []
    test_edges_false = []

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, [train_edges, np.array(train_edges_false)], val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_edges_bipartall(adj, num_node):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    # original_adj edges
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    # test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    # test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    # train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    train_edges = np.delete(edges, val_edge_idx, axis=0)

    sample_size = int((len(train_edges) * 1.0) / num_node)
    sample_size = sample_size + 10

    neg_list = []
    all_candiate = set(range(num_node, adj.shape[0]))
    for i in range(0, num_node):
        non_zeros = adj[i].nonzero()[1]
        neg_candi = np.array(list(all_candiate.difference(set(non_zeros))))
        if len(neg_candi) >= sample_size:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=False)
        elif len(neg_candi) == 0:
            pass
        else:
            neg_candi = neg_candi

        neg_candi = [[i, j] for j in neg_candi]
        # print('len_: %d' % len(neg_candi))
        neg_list.extend(neg_candi)
    # neg_list = np.array(neg_list)

    train_edges_false = np.array(random.sample(neg_list, len(train_edges)))
    val_edges_false = np.array(random.sample(neg_list, len(val_edges)))
    # while len(val_edges_false) < len(val_edges):
    #     idx_i = np.random.randint(0, adj.shape[0])
    #     idx_j = np.random.randint(0, adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], train_edges):
    #         continue
    #     if ismember([idx_j, idx_i], train_edges):
    #         continue
    #     if ismember([idx_i, idx_j], val_edges):
    #         continue
    #     if ismember([idx_j, idx_i], val_edges):
    #         continue
    #     if val_edges_false:
    #         if ismember([idx_j, idx_i], np.array(val_edges_false)):
    #             continue
    #         if ismember([idx_i, idx_j], np.array(val_edges_false)):
    #             continue
    #     val_edges_false.append([idx_i, idx_j])
    #
    # # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    test_edges = []
    test_edges_false = []

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, [train_edges, np.array(train_edges_false)], val_edges, val_edges_false, test_edges, test_edges_false

def mask_test_edges_fast(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    # test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    # test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    # train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    train_edges = np.delete(edges, val_edge_idx, axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    # test_edges_false = []
    # while len(test_edges_false) < len(test_edges):
    #     idx_i = np.random.randint(0, adj.shape[0])
    #     idx_j = np.random.randint(0, adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], edges_all):
    #         continue
    #     if test_edges_false:
    #         if ismember([idx_j, idx_i], np.array(test_edges_false)):
    #             continue
    #         if ismember([idx_i, idx_j], np.array(test_edges_false)):
    #             continue
    #     test_edges_false.append([idx_i, idx_j])

    # val_edges_false = []
    # while len(val_edges_false) < len(val_edges):
    #     idx_i = np.random.randint(0, adj.shape[0])
    #     idx_j = np.random.randint(0, adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], train_edges):
    #         continue
    #     if ismember([idx_j, idx_i], train_edges):
    #         continue
    #     if ismember([idx_i, idx_j], val_edges):
    #         continue
    #     if ismember([idx_j, idx_i], val_edges):
    #         continue
    #     if val_edges_false:
    #         if ismember([idx_j, idx_i], np.array(val_edges_false)):
    #             continue
    #         if ismember([idx_i, idx_j], np.array(val_edges_false)):
    #             continue
    #     val_edges_false.append([idx_i, idx_j])

    num_node = adj.shape[0]
    sample_size = int((len(val_edges) * 1.0) / num_node)
    sample_size = sample_size + 5

    neg_list = []
    all_candiate = set(range(adj.shape[0]))
    for i in range(0, num_node):
        non_zeros = adj[i].nonzero()[1]
        neg_candi = np.array(list(all_candiate.difference(set(non_zeros))))
        if len(neg_candi) >= sample_size:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=False)
        elif len(neg_candi) == 0:
            pass
        else:
            neg_candi = neg_candi

        neg_candi = [[i, j] for j in neg_candi]
        # print('len_: %d' % len(neg_candi))
        neg_list.extend(neg_candi)
    # neg_list = np.array(neg_list)

    # train_edges_false = np.array(random.sample(neg_list, len(train_edges)))
    val_edges_false = np.array(random.sample(neg_list, len(val_edges)))

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    test_edges = []
    test_edges_false = []

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

from .normalization import *

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

import math
def load_normalized_format(datasets, semi_rate=0.1, normalization="AugNormAdj",cuda=True):

    features = datasets['Attributes']
    labels = datasets['Label'].reshape(-1)
    adj = datasets['Network']
    adj, features = preprocess_citation(adj, features, normalization)
    #features = row_normalize(features)

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1

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

    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    features = torch.FloatTensor(np.array(features.todense())).float()
    #features = sparse_mx_to_torch_sparse_tensor(features)
    labels = torch.LongTensor(labels)
    # labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    # adj = torch.FloatTensor(np.array(adj.todense()))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_normalized_agrc(datasets, semi_rate=0.1, normalization="AugNormAdj",cuda=True):

    features = datasets['Attributes']
    labels = datasets['Label'].reshape(-1)
    adj = datasets['Network']
    adj, features = preprocess_citation(adj, features, normalization)
    #features = row_normalize(features)

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1

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

    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    labels = torch.LongTensor(labels)
    # labels = torch.max(labels, dim=1)[1]
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

    return edges, values, features_, labels_, idx_train_, idx_val_, idx_test_


def load_normalized_Not_tensor(datasets, semi_rate=0.1, normalization="AugNormAdj",cuda=True):

    features = datasets['Attributes']
    labels = datasets['Label'].reshape(-1)
    adj = datasets['Network']
    adj, features = preprocess_citation(adj, features, normalization)
    #features = row_normalize(features)

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1

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

    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    features = torch.FloatTensor(np.array(features.todense())).float()
    #features = sparse_mx_to_torch_sparse_tensor(features)
    labels = torch.LongTensor(labels)
    # labels = torch.max(labels, dim=1)[1]
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    # adj = torch.FloatTensor(np.array(adj.todense()))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
def load_citationmat_new( datasets, normalization="AugNormAdj"):
    """
    Load Citation Networks Datasets and obtain 4/5 as training set.
    """

    features = datasets['Attributes']
    labels = datasets['Label'].reshape(-1)
    adj = datasets['Network']
    adj, features = preprocess_citation(adj, features, normalization)
    # features = row_normalize(features)

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1

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
        valid_len = math.ceil(cnt * 0.8 * 0.1)
        idx_test.extend(idx_all[-test_len:])
        idx_val.extend(idx_all[-(test_len + valid_len):-test_len])
        train_len_ = cnt - test_len - valid_len
        idx_train.extend(idx_all[:train_len_])

    idx_train = np.array(idx_train)
    idx_val = np.array(idx_val)
    idx_test = np.array(idx_test)

    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    features = torch.FloatTensor(np.array(features.todense())).float()
    # features = sparse_mx_to_torch_sparse_tensor(features)
    labels = torch.LongTensor(labels)
    # labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    # adj = torch.FloatTensor(np.array(adj.todense()))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
