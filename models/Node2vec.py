
"""
Graph Utility functions

Author: Apoorva Vinod Gorur
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from time import perf_counter
from datetime import timedelta
import argparse
import numpy as np
import networkx as nx
import scipy.io
from collections import defaultdict
from time import perf_counter
from datetime import timedelta
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from multiprocessing import cpu_count, process
from .model import *


def timer(msg):
    def inner(func):
        def wrapper(*args, **kwargs):
            t1 = perf_counter()
            ret = func(*args, **kwargs)
            t2 = perf_counter()
            print("Time elapsed for "+msg+" ----> "+str(timedelta(seconds=t2-t1)))
            print("\n---------------------------------------\n")
            return ret
        return wrapper
    return inner

def read_graph(input_path, directed=False):



    edges = list()
    nodes = input_path.tolil()
    G = nx.DiGraph()
    for start_node, end_nodes in enumerate(nodes.rows, start=0):
        for end_node in end_nodes:
            edges.append((start_node, end_node))

    G.add_edges_from(edges)



    if not directed:
        G = G.to_undirected()

    probs = defaultdict(dict)
    for node in G.nodes():
        probs[node]['probabilities'] = dict()

    print(nx.info(G) + "\n---------------------------------------\n")
    return G, probs


def generate_embeddings(corpus, dimensions, window_size, num_workers, p, q, input_file, output_file):
    model = Word2Vec(corpus, size=dimensions, window=window_size, min_count=0, sg=1, workers=num_workers)
    # model.wv.most_similar('1')
    w2v_emb = model.wv

    # print('Saved embeddings at : ',output_file)
    # w2v_emb.save_word2vec_format(output_file)

    return model, w2v_emb

def newprocess(input, directed, p, q, d, walks, length, workers, window, output,content):
    Graph, init_probabilities = read_graph(input, directed)
    G = HGraph(Graph, init_probabilities, p, q, walks, length, workers)
    G.compute_probabilities()
    walks = G.generate_random_walks()
    model, embeddings = generate_embeddings(walks, d, window, workers, p, q, input, output)

    #mat = scipy.io.loadmat(input)

    H = np.zeros((input.shape[0], d))
    H[:, 0] = 1
    for nodei in Graph.nodes():
        H[nodei] = embeddings[str(nodei)]
    return H

class HGraph():
    
    def __init__(self, graph, probs, p, q, max_walks, walk_len, workers):

        self.graph = graph
        self.probs = probs
        self.p = p
        self.q = q
        self.max_walks = max_walks
        self.walk_len = walk_len
        self.workers = workers 
        return
    
    @timer('Computing probabilities')   
    def compute_probabilities(self):
        
        G = self.graph
        for source_node in G.nodes():
            for current_node in G.neighbors(source_node):
                probs_ = list()
                for destination in G.neighbors(current_node):

                    if source_node == destination:
                        prob_ = G[current_node][destination].get('weight',1) * (1/self.p)
                    elif destination in G.neighbors(source_node):
                        prob_ = G[current_node][destination].get('weight',1)
                    else:
                        prob_ = G[current_node][destination].get('weight',1) * (1/self.q)

                    probs_.append(prob_)

                self.probs[source_node]['probabilities'][current_node] = probs_/np.sum(probs_)
        
        return
    
    @timer('Generating Biased Random Walks')
    def generate_random_walks(self):
        
        G = self.graph
        walks = list()
        for start_node in G.nodes():
            for i in range(self.max_walks):
                
                walk = [start_node]
                walk_options = list(G[start_node])
                if len(walk_options)==0:
                    break
                first_step = np.random.choice(walk_options)
                walk.append(first_step)
                
                for k in range(self.walk_len-2):
                    walk_options = list(G[walk[-1]])
                    if len(walk_options)==0:
                        break
                    probabilities = self.probs[walk[-2]]['probabilities'][walk[-1]]
                    next_step = np.random.choice(walk_options, p=probabilities)
                    walk.append(next_step)
                
                walks.append(walk)
        np.random.shuffle(walks)
        walks = [list(map(str,walk)) for walk in walks]
        
        return walks


class node2vec(Models):

    def __init__(self, datasets,evlation,**kwargs):
        super(node2vec, self).__init__(datasets=datasets, evlation=evlation,**kwargs)
    @classmethod
    def is_preprocessing(cls):
        return False

    @classmethod
    def is_epoch(cls):
        return False

    def train_model(self, mat_content, **kwargs):

        embbeding = newprocess(input=mat_content, directed=False, p=1.0, q=1.0, d=128, walks=4, length=10,
                               workers=12,
                               window=5, output=None, content='network')
        scipy.io.savemat('node2vec_Embedding.mat', {"node2vec": embbeding})

        return 'node2vec_Embedding.mat', "node2vec"
