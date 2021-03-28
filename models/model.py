import torch
import scipy.io as sio
import time
import preprocessing.preprocessing as pre
from evaluation.node_classification import node_classifcation
from evaluation.link_prediction import link_prediction,link_prediction_Automatic_tuning
import numpy as np
from hyperopt import fmin, tpe, hp, space_eval,Trials, partial


class Models(torch.nn.Module):
    def __init__(self, *, datasets, Time, **kwargs):
        self.mat_content=datasets
        self.stop_time = Time
        super(Models, self).__init__()
        if self.is_preprocessing():
            self.preprocessing(datasets)
        start_time = time.time()
        if self.is_deep_model():
            emb = self.deep_algo(self.stop_time)
        else:
            emb = self.shallow_algo()
        self.emb = emb
        self.end_time = time.time()-start_time


    def check_train_parameters(self):
        return None

    @classmethod
    def is_preprocessing(cls):
        raise NotImplementedError

    @classmethod
    def is_deep_model(cls):
        raise NotImplementedError

    def forward(self):
        return None

    def train_model(self, **kwargs):
        filename = ""
        return filename

    def get_score(self,params):
        emb = self.train_model(**params)
        adj = self.mat_content['Network']
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_test_edges(adj)
        score=link_prediction_Automatic_tuning(emb,edges_pos=val_edges,edges_neg=val_edges_false)
        return -score

    def preprocessing(self, filename):
        return None

    def deep_algo(self,stop_time):
        return None

    def shallow_algo(self):
        trials = Trials()
        algo = partial(tpe.suggest)
        space_dtree = self.check_train_parameters()
        best = fmin(
            fn=self.get_score, space=space_dtree, algo=algo, max_evals=150, trials=trials, timeout=self.stop_time)
        print('end of training:{:.2f}s'.format(self.stop_time))
        emb = self.train_model(**best)
        return emb

    def get_emb(self):
        return self.emb

    def get_time(self):
        return self.end_time



