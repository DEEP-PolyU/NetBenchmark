from models.deepwalk_package import graph, walks as serialized_walks
from models.deepwalk_package.skipgram import Skipgram
from gensim.models import Word2Vec
import random
import os
import numpy as np
import time
import scipy.io as sio
from evaluation.node_classification import node_classifcation_test
from .model import *
from hyperopt import fmin, tpe, hp, space_eval,Trials, partial



def deepwalk_fun(CombG, d, number_walks, walk_length,window_size,evaluation):

    max_memory_data_size = 466000000.0
    seed = 0
    vertex_freq_degree = False

    G = graph.from_numpy(CombG)
    #print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * number_walks

    #print("Number of walks: {}".format(num_walks))

    data_size = num_walks * walk_length

    #print("Data size (walks*length): {}".format(data_size))


    if data_size < max_memory_data_size:
        # print("Walking...")
        start_time = time.time()
        walks = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                            path_length=walk_length, alpha=0, rand=random.Random(seed))
        # print("time elapsed: {:.2f}s".format(time.time() - start_time))
        # print("Training...")
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, size=d, window=window_size, min_count=0, workers=os.cpu_count())
        # print("time elapsed: {:.2f}s".format(time.time() - start_time))
    else:
        # print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, max_memory_data_size))
        # print("Walking...")

        walks_filebase = "deepwalk_cache" + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=number_walks,
                                                          path_length=walk_length, alpha=0,
                                                          rand=random.Random(seed),
                                                          num_workers=os.cpu_count())

        print("Counting vertex frequency...")
        if not vertex_freq_degree:
            vertex_counts = serialized_walks.count_textfiles(walk_files, 1)
        else:
            # use degree distribution for frequency in tree
            vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")
        model = Skipgram(sentences=serialized_walks.combine_files_iter(walk_files), vocabulary_counts=vertex_counts,
                         size=d,
                         window=window_size, min_count=0, workers=os.cpu_count())

    word_vectors = model.wv
    H = np.zeros((CombG.shape[0], d))
    H[:,0] = 1
    for nodei in G.nodes():
        H[nodei] = word_vectors[str(nodei)]
    return H

## start





class deepwalk(Models):


    def check_train_parameters(self):

        space_dtree = {

            'number_walks': hp.uniformint('number_walks', 5, 80),
            'walk_length': hp.uniformint('walk_length', 5, 50),
            'window_size': hp.uniformint('window_size', 5, 50), #walk_length,window_size
            'evaluation': hp.choice('evaluation', self.evaluation)
        }


        return space_dtree

    @classmethod
    def is_preprocessing(cls):
        return False

    @classmethod
    def is_deep_model(cls):
        return False

    def train_model(self, **kwargs): #(self,rootdir,variable_name,number_walks):


        ComG = self.mat_content['Network']

        embbeding = deepwalk_fun(ComG, d = 128,**kwargs)

        # sio.savemat('Deepwalk_Embedding.mat', {"Deepwalk": embbeding})
        #
        # return 'Deepwalk_Embedding.mat',"Deepwalk"
        return embbeding

    # def get_score(self, params):
    #     ComG = self.mat_content['Network']
    #
    #     embbeding = deepwalk_fun(ComG, d=128, **params)
    #
    #     Label = self.mat_content["Label"]
    #     score=node_classifcation_test(np.array(embbeding),Label)
    #
    #     return -score
