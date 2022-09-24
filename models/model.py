import torch
import scipy.io as sio
import time
import preprocessing.preprocessing as pre
from evaluation.node_classification import node_classifcation_10_time
from evaluation.link_prediction import link_prediction,link_prediction_Automatic_tuning
from evaluation.node_classification import node_classifcation_end2end
import numpy as np
import hyperopt
from hyperopt import fmin, tpe, hp, space_eval,Trials, partial,atpe
import os

class Models(torch.nn.Module):

    def __init__(self, *, datasets, time_setting, task_method,tuning,cuda,**kwargs):
        # Train on CPU (hide GPU) due to memory constraints
        # os.environ['CUDA_VISIBLE_DEVICES'] = [0,1,2,3,4,5,6]
        self.cpu_number = 4
        self.use_gpu = torch.cuda.is_available()
        cuda_name = 'cuda:' + cuda
        self.device = torch.device(cuda_name if self.use_gpu else 'cpu')
        self.mat_content=datasets
        self.best = {}
        self.stop_time = time_setting
        self.task_method = task_method
        self.tuning = tuning
        if task_method != 'task1':
            adj_train, train_edges, val_edges, val_edges_false = pre.mask_test_edges(self.mat_content['Network'])
            self.adj_train = adj_train
            self.val_edges = val_edges
            self.val_edges_false = val_edges_false
        super(Models, self).__init__()
        if self.is_preprocessing():
            self.preprocessing(datasets)

        if self.is_end2end():
            self.F1_mic, self.F1_mac, self.best = self.end2end()
        else:
            if self.is_preprocessing() == True:
                emb = self.train_model()
                self.emb = emb
                self.best = {}
                self.tuning_times = '0'
            else:
                emb, best, tuning_times = self.parameter_tuning()
                self.best = best
                self.emb = emb
                self.tuning_times=tuning_times
        start_time = time.time()
        self.end_time = time.time() - start_time



    def check_train_parameters(self):
        return None

    def replace_mat_content(self,adj):
        self.mat_content['Network']=adj

    @classmethod
    def is_preprocessing(cls):
        raise NotImplementedError

    @classmethod
    def is_deep_model(cls):
        raise NotImplementedError
    @classmethod
    def is_end2end(cls):
        raise NotImplementedError

    def forward(self):
        return None

    def train_model(self, **kwargs):
        embedding=''
        return embedding

    def get_score(self,params):
        try:

            if self.task_method == 'task2' or self.task_method == 'task3':

                self.replace_mat_content(self.adj_train)
                emb = self.train_model(**params)
                score = link_prediction_Automatic_tuning(emb, edges_pos=self.val_edges, edges_neg=self.val_edges_false)
            elif self.task_method == 'task1':
                emb = self.train_model(**params)
                score=node_classifcation_end2end(np.array(emb), self.mat_content['Label'])
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            score=0
        return -score

    def en2end_get_score(self,params):
        F1 = self.train_model(**params)

        return -F1

    def preprocessing(self, filename):
        return None


    #hyperparameter tuning
    def parameter_tuning(self):
        trials = Trials()
        if self.tuning == 'random':
            algo = partial(hyperopt.rand.suggest)
        elif self.tuning== 'tpe':
            algo = partial(tpe.suggest)
        else:
            algo = partial(atpe.suggest)

        space_dtree = self.check_train_parameters()
        best = fmin(fn=self.get_score, space=space_dtree, algo=algo, max_evals=10000, trials=trials, timeout=self.stop_time, rstate=np.random.default_rng(42))
        hyperparam = hyperopt.space_eval(space_dtree,best)
        tuning_time = len(trials)
        print(hyperparam)
        print('tuning_times:',tuning_time)
        print('end of training:{:.2f}s'.format(self.stop_time))
        emb = self.train_model(**hyperparam)

        return emb,hyperparam,tuning_time



    def end2end(self):
        trials = Trials()
        if self.tuning == 'random':
            algo = partial(hyperopt.rand.suggest)
        elif self.tuning == 'tpe':
            algo = partial(tpe.suggest)
        else:
            algo = partial(atpe.suggest)

        space_dtree = self.check_train_parameters()
        best = fmin(
            fn=self.en2end_get_score, space=space_dtree, algo=algo, max_evals=10000, trials=trials, timeout=self.stop_time)
        hyperparam = hyperopt.space_eval(space_dtree, best)
        print(hyperparam)
        tuning_time = len(trials)
        print('end of training:{:.2f}s'.format(self.stop_time))
        F1_mic, F1_mac = self.get_best_result(**hyperparam)
        self.tuning_times=tuning_time
        return F1_mic, F1_mac, hyperparam

    def end2endsocre(self):
        return self.F1_mic,self.F1_mac

    def get_time(self):
        return self.end_time



