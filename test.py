import scipy.io as sio
import preprocessing.preprocessing as pre
from evaluation.link_prediction import link_prediction




matr = sio.loadmat('data/BlogCatalog/BlogCatalog.mat')
matr_emb=sio.loadmat('Deepwalk_Embedding.mat')
adj=matr['Network']
print(matr["Network"].shape)
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_test_edges(adj)
link_prediction(emb_name="Deepwalk_Embedding.mat",variable_name="Deepwalk",edges_pos=val_edges,edges_neg = val_edges_false)
