import scipy.io as sio
import preprocessing.preprocessing as pre
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from preprocessing.utils import normalize, sigmoid,sparse_mx_to_torch_sparse_tensor, load_citationmat
from preprocessing.loadCora import load_citation
from models.deepwalk import deepwalk_fun
def get_roc_score(emb_name, edges_pos, edges_neg):
    matr = sio.loadmat(emb_name)

    data = matr['Deepwalk']
    emb=data
    #emb = np.transpose(data)

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
    print("roc_score=",roc_score)
    print("ap_score=",ap_score)


adj, features, labels, idx_train, idx_val, idx_test = load_citation('cora', 'AugNormAdj',1,'0')
net_emb=deepwalk_fun(adj,number_walks=10,representation_size=10)
sio.savemat('deepWalk_cora.mat', {"Deepwalk": net_emb})
# matr = sio.loadmat('deepWalk_cora.mat')
# # matr_emb=sio.loadmat('Deepwalk_Embedding.mat')
# net_emb= matr['Network']
# print(adj.shape)
# print(matr_emb['Deepwalk'].shape)
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_test_edges(adj)
get_roc_score(emb_name="deepWalk_cora.mat",edges_pos=val_edges,edges_neg = val_edges_false)
