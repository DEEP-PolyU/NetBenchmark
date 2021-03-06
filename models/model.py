import torch
import scipy.io as sio
import time
import preprocessing.preprocessing as pre
from evaluation.node_classification import node_classifcation
from evaluation.link_prediction import link_prediction
import numpy as np
from hyperopt import fmin, tpe, hp, space_eval,Trials, partial


class Models(torch.nn.Module):
    def __init__(self, *, datasets, evaluation, **kwargs):
        super(Models, self).__init__()
        if (self.is_preprocessing == True):
            self.preprocessing(datasets)
        if (self.is_epoch == True):
            self.forward()
        self.mat_content=datasets
        space_dtree=self.check_train_parameters()
        Label = self.mat_content["Label"]
        trials = Trials()
        algo = partial(tpe.suggest)
        best = fmin(
            fn=self.get_score, space=space_dtree, algo=algo, max_evals=150, trials=trials)
        print(best)
        if evaluation == "node_classification":
            start_time = time.time()
            self.save_emb_name, self.model_name = self.train_model(**best)
            print("time elapsed: {:.2f}s".format(time.time() - start_time))
            matr = sio.loadmat(self.save_emb_name)
            model = matr[self.model_name]
            node_classifcation(np.array(model), Label)
        if evaluation == "link_prediction":
            matr = sio.loadmat(datasets)
            adj = matr['Network']
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_test_edges(adj)
            link_prediction(emb_name=self.save_emb_name, variable_name=self.model_name, edges_pos=val_edges,
                            edges_neg=val_edges_false)

    def check_train_parameters(self):
        return None

    @classmethod
    def is_preprocessing(cls):
        raise NotImplementedError

    @classmethod
    def is_epoch(cls):
        raise NotImplementedError

    def forward(self):
        return None

    def train_model(self, **kwargs):
        filename = ""
        return filename

    def get_score(self,params):
        score = ""
        return score

    def preprocessing(self, filename):
        return None
