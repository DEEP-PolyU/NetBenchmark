from .node_classification import node_classifcation
import scipy.io as sio
import preprocessing.preprocessing as pre
from .link_prediction import link_prediction
import numpy as np

def evaluation(emb,Graph,evaluation):

    if evaluation == "node_classification":
        f1_mic,f1_mac=node_classifcation(np.array(emb), Graph['Label'])
        return f1_mic, f1_mac
    elif evaluation == "link_prediction":
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_test_edges(Graph['Network'])
        roc_score,ap_score=link_prediction(emb, edges_pos=test_edges ,edges_neg=test_edges_false)
        return roc_score, ap_score
