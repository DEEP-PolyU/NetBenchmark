import numpy as np
import numpy.random as npr
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize
from math import ceil
from gensim.models import Word2Vec


class featurewalk_S:
    def __init__(self, featur1, alpha1, featur2, alpha2, Net, beta, num_paths, path_length, dim, win_size):
        self.n = featur1.shape[0]  # number of instance
        if alpha1+alpha2 < 1:  # Embed Network
            Net = csc_matrix(Net)
            self.path_list_Net = []
            self.idx_Net = []
            for i in range(self.n):
                self.path_list_Net.append(Net.getcol(i).nnz)
                self.idx_Net.append(Net.getcol(i).indices)
            weight = float(1 - alpha1 - alpha2) * num_paths * self.n / np.sum(self.path_list_Net)
            self.path_list_Net = [int(ceil(i * weight)) for i in self.path_list_Net]

        if alpha1 > 0:  # Embed Feature 1
            self.featur1 = featur1
        if alpha2 > 0:  # Embed Feature 2
            self.featur2 = featur2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.num_paths = num_paths
        self.path_length = path_length
        self.d = dim
        self.win_size = win_size





    def function(self):
        sentenceList = []
        # Embed Net first, such that we could del Net earlier
        if self.alpha1 + self.alpha2 < 1:  # Embed Network
            allidx = np.nonzero(self.path_list_Net)[0]
            if len(allidx) != self.n:
                for i in np.where(np.asarray(self.path_list_Net) == 0)[0]:
                    sentenceList.append([str(i)])

            for i in allidx:
                for j in range(self.path_list_Net[i]):
                    sentence = [npr.choice(self.idx_Net[i]), i]
                    for tmp in range(self.path_length - 2):
                        sentence.append(npr.choice(self.idx_Net[sentence[-1]]))
                    sentenceList.append([str(word) for word in sentence])
            del self.idx_Net, self.path_list_Net

        for alphaidx in range(2):  # Embed Feature
            if alphaidx == 0:
                alpha = self.alpha1
                if alpha > 0:
                    featur = normalize(csc_matrix(self.featur1), norm='l2')
                    del self.featur1

            if alphaidx == 1:
                alpha = self.alpha2
                if alpha > 0:
                    featur = normalize(csc_matrix(self.featur2), norm='l2')
                    del self.featur2

            if alpha > 0:
                if self.beta > 0:
                    featur = featur.multiply(featur > self.beta * (featur.sum() / featur.nnz))
                featur = featur[:, np.where((featur != 0).sum(axis=0) > 1)[1]].T

                self.path_list = []
                for i in range(self.n):
                    self.path_list.append(featur.getcol(i).nnz)

                featur = normalize(featur, norm='l1', axis=0)
                self.JListrow, self.qListrow, self.idx = walk_setup(normalize(featur.T, norm='l1', axis=0) * featur)
                del featur

                weight = float(alpha * self.num_paths * self.n) / np.sum(self.path_list)
                self.path_list = [ceil(i * weight) for i in self.path_list]

                allidx = np.nonzero(self.path_list)[0]
                if alpha == 1 and len(allidx) != self.n:
                    for i in np.where(np.asarray(self.path_list) == 0)[0]:
                        sentenceList.append([str(i)])

                for i in allidx:
                    for j in range(self.path_list[i]):
                        current = i  # ahead
                        sentence = [self.idx[i][alias_draw(self.JListrow[i], self.qListrow[i])], i]
                        for tmp in range(self.path_length - 2):
                            current = self.idx[current][alias_draw(self.JListrow[current], self.qListrow[current])]
                            sentence.append(current)
                        sentenceList.append([str(word) for word in sentence])
                del self.JListrow, self.qListrow, self.idx

        word_vectors = Word2Vec(sentenceList, size=self.d, window=self.win_size, min_count=0).wv
        del sentenceList
        H = np.zeros((self.n, self.d))
        for nodei in range(self.n):
            H[nodei] = word_vectors[str(nodei)]
        return H


#  Compute utility lists for non-uniform sampling from discrete distributions.
#  Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def walk_setup(sx):
    qListrow = []  # for each instance
    JListrow = []
    idx = []
    for ni in range(sx.shape[1]):   # for each instance
        coli = sx.getcol(ni)
        J, q = alias_setup(coli.data)
        JListrow.append(J)
        qListrow.append(q)
        idx.append(coli.indices)

    return JListrow, qListrow, idx

def alias_draw(J, q):
    K = len(J)
    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand() * K))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]


