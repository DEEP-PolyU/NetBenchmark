import numpy as np
import scipy.io as sio
import time
from models.FeatWalk import featurewalk
from evaluation.SVM import node_classify
'''################# Load data  #################'''
mat_contents = sio.loadmat('data/ACM/ACM.mat')
number_walks = 35  # 'Number of random walks to start at each instance'
walk_length = 25  # 'Length of the random walk started at each instance'
win_size = 5  # 'Window size of skipgram models.'

'''################# Experimental Settings #################'''
d = 100  # the dimension of the embedding representation
X1 = mat_contents["Features"]
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
H_FeatWalk = featurewalk(featur1=CombX1, alpha1=.97, featur2=None, alpha2=0, Net=CombX2, beta=0, num_paths=number_walks, path_length=walk_length, dim=d, win_size=win_size).function()
print("time elapsed: {:.2f}s".format(time.time() - start_time))
sio.savemat('ACM_Embedding.mat', {"H_FeatWalk": H_FeatWalk})
import scipy.io as io
matr = io.loadmat('ACM_Embedding.mat')

H_FeatWalk = matr['H_FeatWalk']
labels = Label.reshape(-1)
# label_min = np.min(labels)
# if label_min != 0:
#     labels = labels - 1
# max_class = np.max(labels) + 1
# class_one = np.eye(max_class)
# labels = class_one[labels]

node_classify(np.array(H_FeatWalk),labels)