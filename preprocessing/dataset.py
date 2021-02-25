import scipy
import logging
import loadCora

logger = logging.getLogger(__name__)
def load_adjacency_matrix(file, variable_name="network"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    if variable_name not in data:
        variable_name[0]=variable_name[0].upper()
    return data[variable_name]

class Datasets:
    def __init__(self):
        super(Datasets, self).__init__()

    def get_graph(self):
        graph = None
        return graph

    @classmethod
    def attributed(cls):
        raise NotImplementedError

class ACM(Datasets):
    def __init__(self):
        super(ACM, self).__init__()

    def get_graph(self):
        dir='data/ACM/ACM.mat'
        return load_adjacency_matrix(dir)

    @classmethod
    def attributed(cls):
        return True


class Flickr(Datasets):
    def __init__(self):
        super(Flickr, self).__init__()

    def get_graph(self):
        dir = 'data/Flickr/Flickr_SDM.mat'

        return load_adjacency_matrix(dir)

    @classmethod
    def attributed(cls):
        return True

class BlogCatalog(Datasets):
    def __init__(self):
        super(BlogCatalog, self).__init__()

    def get_graph(self):
        dir = 'data/BlogCatalog/BlogCatalog.mat'
        return load_adjacency_matrix(dir)

    @classmethod
    def attributed(cls):
        return True

class Cora(Datasets):
    def __init__(self):
        super(Cora, self).__init__()

    def get_graph(self):
        adj, features, labels, idx_train, idx_val, idx_test = loadCora.load_citation()
        return adj

    @classmethod
    def attributed(cls):
        return True
