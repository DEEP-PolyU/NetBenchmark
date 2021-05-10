import torch
import scipy.io as sio
import time
import preprocessing.preprocessing as pre
from evaluation.node_classification import node_classifcation
from evaluation.link_prediction import link_prediction,link_prediction_Automatic_tuning
from evaluation.node_classification import node_classifcation_test
import numpy as np
from hyperopt import fmin, tpe, hp, space_eval,Trials, partial


class Models(torch.nn.Module):
    def __init__(self, *, datasets, Time, evaluation,**kwargs):
        self.mat_content=datasets
        self.best = {}
        self.stop_time = Time
        self.evaluation = evaluation
        super(Models, self).__init__()
        if self.is_preprocessing():
            self.preprocessing(datasets)
        start_time = time.time()
        if self.is_deep_model():
            emb,best = self.deep_algo()
            self.best = best
            self.emb = emb
            self.end_time = time.time() - start_time
        else:
            emb,best = self.shallow_algo()
            self.best = best
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
        if params['evaluation'] == 'link_prediction':
            score=link_prediction_Automatic_tuning(emb,edges_pos=val_edges,edges_neg=val_edges_false)
        else:
            score=node_classifcation_test(np.array(emb),self.mat_content['Label'])

        return -score

    def preprocessing(self, filename):
        return None


    #hyperparameter tuning
    def deep_algo(self):
        trials = Trials()
        algo = partial(tpe.suggest)
        space_dtree = self.check_train_parameters()
        best = fmin(fn=self.get_score, space=space_dtree, algo=algo, max_evals=150, trials=trials, timeout=self.stop_time)
        print(best)
        print('end of training:{:.2f}s'.format(self.stop_time))
        emb = self.train_model(**best)

        return emb,best

    def shallow_algo(self):
        trials = Trials()
        algo = partial(tpe.suggest)
        space_dtree = self.check_train_parameters()
        best = fmin(
            fn=self.get_score, space=space_dtree, algo=algo, max_evals=150, trials=trials, timeout=self.stop_time)
        print(best)
        print('end of training:{:.2f}s'.format(self.stop_time))
        emb = self.train_model(**best)
        return emb,best

    def get_emb(self):
        return self.emb
    def get_best(self):
        return self.best
    def get_time(self):
        return self.end_time



