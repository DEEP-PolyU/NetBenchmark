import scipy
import logging
from .loadCora import load_citation

logger = logging.getLogger(__name__)
def load_adjacency_matrix(file, variable_name="network"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    if variable_name not in data:
        variable_name=variable_name.capitalize()
    return data[variable_name]

class Datasets:
    def __init__(self):
        super(Datasets, self).__init__()

    def get_graph(self,variable_name):
        graph = None
        return graph

    @classmethod
    def attributed(cls):
        raise NotImplementedError

class ACM(Datasets):
    def __init__(self):
        super(ACM, self).__init__()

    def get_graph(self,variable_name):
        dir='data/ACM/ACM.mat'
        return load_adjacency_matrix(dir,variable_name)

    @classmethod
    def attributed(cls):
        return True


class Flickr(Datasets):
    def __init__(self):
        super(Flickr, self).__init__()

    def get_graph(self,variable_name):
        dir = 'data/Flickr/Flickr_SDM.mat'

        return load_adjacency_matrix(dir,variable_name)

    @classmethod
    def attributed(cls):
        return True

class BlogCatalog(Datasets):
    def __init__(self):
        super(BlogCatalog, self).__init__()

    def get_graph(self,variable_name):
        dir = 'data/BlogCatalog/BlogCatalog.mat'
        return load_adjacency_matrix(dir,variable_name)

    @classmethod
    def attributed(cls):
        return True

class Cora(Datasets):
    def __init__(self):
        super(Cora, self).__init__()

    def get_graph(self,variable_name):
        adj, features, labels, idx_train, idx_val, idx_test = load_citation()
        return adj

    @classmethod
    def attributed(cls):
        return True
