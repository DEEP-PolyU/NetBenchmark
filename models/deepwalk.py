from models.deepwalk_package import graph, walks as serialized_walks
from models.deepwalk_package.skipgram import Skipgram
from gensim.models import Word2Vec
import random
import numpy as np
import time
import scipy.io as sio



def deepwalk_fun(CombG, representation_size, number_walks):

    walk_length = 40
    max_memory_data_size = 1000000000
    window_size = 5
    seed = 0
    vertex_freq_degree = False

    G = graph.from_numpy(CombG)
    print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * walk_length

    print("Data size (walks*length): {}".format(data_size))

    if data_size < max_memory_data_size:
        print("Walking...")
        start_time = time.time()
        walks = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                            path_length=walk_length, alpha=0, rand=random.Random(seed))
        print("time elapsed: {:.2f}s".format(time.time() - start_time))
        print("Training...")
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, workers=1)
        print("time elapsed: {:.2f}s".format(time.time() - start_time))
    else:
        print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, max_memory_data_size))
        print("Walking...")

        walks_filebase = "deepwalk_cache" + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=number_walks,
                                                          path_length=walk_length, alpha=0,
                                                          rand=random.Random(seed),
                                                          num_workers=1)

        print("Counting vertex frequency...")
        if not vertex_freq_degree:
            vertex_counts = serialized_walks.count_textfiles(walk_files, 1)
        else:
            # use degree distribution for frequency in tree
            vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")
        model = Skipgram(sentences=serialized_walks.combine_files_iter(walk_files), vocabulary_counts=vertex_counts,
                         size=representation_size,
                         window=window_size, min_count=0, workers=1)

    word_vectors = model.wv
    H = np.zeros((CombG.shape[0], representation_size))
    H[:,0] = 1
    for nodei in G.nodes():
        H[nodei] = word_vectors[str(nodei)]
    return H

## start





class deepwalk:
    def deepwalkgood(self,rootdir,variable_name,number_walks,representation_size):

        mat_contents = sio.loadmat(rootdir)

        ComG = mat_contents[variable_name]

        embeding = deepwalk_fun(ComG, representation_size, number_walks)

        sio.savemat('Deepwalk_Embedding.mat', {"Deepwalk": embeding})