import scipy.io as sio
import preprocessing.preprocessing as pre
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from preprocessing.utils import normalize, sigmoid, load_citation, sparse_mx_to_torch_sparse_tensor, load_citationANEmat
import copy
import multiprocessing
from tqdm import tqdm
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

def split_dataset(Graph):
    temp_Graph = copy.deepcopy(Graph)
    adj_train, train_edges, val_edges, val_edges_false = pre.mask_test_edges(temp_Graph['Network'])
    return adj_train, train_edges, val_edges, val_edges_false


def link_prediction_10_time(best, Graph,model):
    total_time=10
    roc_score=[]
    ap_score=[]
    results_list=[]
    pool = multiprocessing.Pool(processes=4)
    for i in range(total_time):
        result_per_read = pool.apply_async(split_dataset, (Graph,))
        results_list.append(result_per_read)
    pool.close()
    pbar = tqdm(total=total_time, position=0, leave=True)
    for result_df in results_list:
        temp = result_df.get()
        if temp is not None:
            adj_train, train_edges, val_edges, val_edges_false=temp
            model.replace_mat_content(adj_train)
            emb = model.train_model(**best)
            roc, ap = link_prediction(emb, edges_pos=val_edges, edges_neg=val_edges_false)
            roc_score.append(roc)
            ap_score.append(ap)
        pbar.update(1)
    pool.join()
    print("roc_score=",np.mean(roc_score),"± %.4f" % np.std(roc_score))
    print("ap_score=",np.mean(ap_score),"± %.4f" % np.std(ap_score))
    return np.mean(roc_score), np.mean(ap_score),np.std(roc_score), np.std(ap_score)

def link_prediction_10_time_old(best, Graph, model):
    total_time = 5
    roc_score = []
    ap_score = []
    pbar = tqdm(total=total_time, position=0, leave=True)
    for i in range(total_time):
        temp_Graph = copy.deepcopy(Graph)
        adj_train, train_edges, val_edges, val_edges_false= pre.mask_test_edges(temp_Graph['Network'])
        model.replace_mat_content(adj_train)
        emb=model.train_model(**best)
        roc, ap = link_prediction(emb, edges_pos=val_edges, edges_neg=val_edges_false)
        roc_score.append(np.array(roc))
        ap_score.append(np.array(ap))
        pbar.update(1)
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