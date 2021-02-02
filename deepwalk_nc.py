import numpy as np
import scipy.io as sio
import time
from models.deepwalk import deepwalk_c
from evaluation.node_classification import SVM


mat_contents = sio.loadmat('data/ACM/ACM.mat')
start_time = time.time()

deepwalk_c().deepwalkgood(rootdir='data/BlogCatalog/BlogCatalog.mat',variable_name="Network",number_walks=10,representation_size=10)
print("time elapsed: {:.2f}s".format(time.time() - start_time))
Label = mat_contents["Label"]
matr2 = sio.loadmat('deepwalk_Embedding.mat')
Deepwalk= matr2['Deepwalk']
labels = Label.reshape(-1)
SVM(np.array(Deepwalk),labels)