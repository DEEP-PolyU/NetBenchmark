import numpy as np
import scipy.io as sio
import time
from models.FeatWalk import featurewalk
from models.FeatWalk_S import featurewalk_S
from evaluation.node_classification import node_classifcation
'''################# Load data  #################'''
# adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str="cora", normalization="AugNormAdj", use_feat=1, cuda=True)
mat_contents = sio.loadmat('data/Flickr/Flickr_SDM.mat')
number_walks = 8  # 'Number of random walks to start at each instance'
walk_length = 25  # 'Length of the random walk started at each instance'
win_size = 5  # 'Window size of skipgram models.'

'''################# Experimental Settings #################'''
d = 100  # the dimension of the embedding representation
X1 = mat_contents["Attributes"]
X2 = mat_contents["Network"]
Label = mat_contents["Label"]
del mat_contents
n = X1.shape[0]
Indices = np.random.randint(25, size=n)+1  # 5-fold cross-validation indices
Group1 = []
Group2 = []
[Group1.append(x) for x in range(0, n) if Indices[x] <= 20]  # 2 for 10%, 5 for 25%, 20 for 100% of training group
[Group2.append(x) for x in range(0, n) if Indices[x] >= 21]  # test group
n1 = len(Group1)  # num of instances in training group
n2 = len(Group2)  # num of instances in test group
CombX1 = X1[Group1+Group2, :]
CombX2 = X2[Group1+Group2, :][:, Group1+Group2]
start_time = time.time()
H_FeatWalk = featurewalk_S(featur1=CombX1, alpha1=.62, featur2=None, alpha2=0, Net=CombX2, beta=0, num_paths=number_walks, path_length=walk_length, dim=d, win_size=win_size).function()
print("time elapsed: {:.2f}s".format(time.time() - start_time))
sio.savemat('Featwalk_Flickr_Embedding.mat', {"H_FeatWalk": H_FeatWalk})
import scipy.io as io
matr = io.loadmat('Featwalk_Flickr_Embedding.mat')
H_FeatWalk_emb = matr['H_FeatWalk']
node_classifcation(np.array(H_FeatWalk_emb),Label)

