import torch
import scipy.io as sio
import time
import preprocessing.preprocessing as pre
from evaluation.node_classification import node_classifcation
from evaluation.link_prediction import link_prediction
import numpy as np


class Models(torch.nn.Module):
    def __init__(self, *, datasets, evaluation, **kwargs):
        super(Models, self).__init__()
        if (self.is_preprocessing == True):
            self.preprocessing(datasets)
        if (self.is_epoch == True):
            self.forward()
        # mat_contents = sio.loadmat(datasets)
        mat_contents = datasets
        start_time = time.time()
        self.save_emb_name, self.model_name = self.train_model(datasets, **kwargs)
        print("time elapsed: {:.2f}s".format(time.time() - start_time))
        if evaluation == "node_classification":
            Label = mat_contents["Label"]
            matr = sio.loadmat(self.save_emb_name)
            model = matr[self.model_name]
            node_classifcation(np.array(model), Label)
        if evaluation == "link_prediction":
            matr = sio.loadmat(datasets)
            adj = matr['Network']
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_test_edges(adj)
            link_prediction(emb_name=self.save_emb_name, variable_name=self.model_name, edges_pos=val_edges,
                            edges_neg=val_edges_false)

    # @classmethod
    # def check_train_parameters(cls, **kwargs):
    #     return kwargs

    @classmethod
    def is_preprocessing(cls):
        raise NotImplementedError

    @classmethod
    def is_epoch(cls):
        raise NotImplementedError

    def forward(self):
        return None

    def train_model(self, mat_content, **kwargs):
        filename = ""
        return filename

    def preprocessing(self, filename):
        return None
