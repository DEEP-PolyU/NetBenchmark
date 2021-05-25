import scipy
import logging
from .loadCora import load_citation

logger = logging.getLogger(__name__)
def load_adjacency_matrix(file):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    if 'Features' in data:
        data['Attributes']=data['Features']
        del data['Features']
    return data

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
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str="cora")
        data={"Network":adj,"Label":labels,"Attributes":features}
        return data

    @classmethod
    def attributed(cls):
        return True

class Citeseer(Datasets):
    def __init__(self):
        super(Citeseer, self).__init__()

    def get_graph(self):
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str="citeseer")
        data={"Network":adj,"Label":labels,"Attributes":features}
        return data

    @classmethod
    def attributed(cls):
        return True


class Citeseer(Datasets):
    def __init__(self):
        super(Citeseer, self).__init__()

    def get_graph(self):
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str="citeseer")
        data={"Network":adj,"Label":labels,"Attributes":features}
        return data

    @classmethod
    def attributed(cls):
        return True


class neil001(Datasets):
    def __init__(self):
        super(neil001, self).__init__()

    def get_graph(self):
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str="nell.0.001")
        data={"Network":adj,"Label":labels,"Attributes":features}
        return data

    @classmethod
    def attributed(cls):
        return True