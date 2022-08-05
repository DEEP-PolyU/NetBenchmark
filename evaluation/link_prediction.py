import scipy.io as sio
import preprocessing.preprocessing as pre
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from preprocessing.utils import normalize, sigmoid, load_citation, sparse_mx_to_torch_sparse_tensor, load_citationANEmat

def link_prediction(emb,edges_pos, edges_neg):

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
    # print("roc_score=",roc_score)
    # print("ap_score=",ap_score)
    return roc_score,ap_score


def link_prediction_10_time(best, Graph,model):
    roc_score=[]
    ap_score=[]
    for i in range(10):
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_test_edges(Graph['Network'])
        model.replace_mat_content(adj_train)
        emb=model.train_model(**best)
        roc, ap = link_prediction(emb, edges_pos=val_edges, edges_neg=val_edges_false)
        roc_score.append(np.array(roc))
        ap_score.append(np.array(ap))
        # print(roc_score,ap_score)
    print("roc_score=",np.mean(roc_score))
    print("ap_score=",np.mean(ap_score))
    return np.mean(roc_score), np.mean(ap_score)



def link_prediction_Automatic_tuning(emb, edges_pos, edges_neg):
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
    return (roc_score+ap_score)/2