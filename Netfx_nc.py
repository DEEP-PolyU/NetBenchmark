import numpy as np
import scipy.io as sio
import time
from models.NetMF import netmf
from evaluation.SVM import node_classify
mat_contents = sio.loadmat('data/BlogCatalog/BlogCatalog.mat')
start_time = time.time()
netmf().netmf_small(rootdir='data/BlogCatalog/BlogCatalog.mat',variable_name="Network")
print("time elapsed: {:.2f}s".format(time.time() - start_time))
Label = mat_contents["Label"]
matr2 = sio.loadmat('netmf_Embedding.mat')
netmf= matr2['NetMF']
labels = Label.reshape(-1)
node_classify(np.array(netmf),labels)