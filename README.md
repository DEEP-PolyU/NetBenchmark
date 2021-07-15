# NetBenchmark
Node Representation Learning Benchmark


| Status        | Developing      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | QIAN Zhiqiang (zhiqiang.qian@connect.polyu.hk), QUO Zhihao (zhi-hao.guo@connect.polyu.hk), YU Rico (ricoyu.yu@connect.polyu.hk) , LIAN, Amber (amber.lian@connect.polyu.hk) |
| **Updated**   | 2021-05                                           |


## Objective

We aim at building a auto,fair and systematic evaluation platform to compare the results of different Network Embedding models. 
The implemented or modified models include [DeepWalk](https://github.com/phanein/deepwalk),  [node2vec](https://github.com/aditya-grover/node2vec), 
[GCN](https://github.com/tkipf/gcn), [NetMF](https://github.com/xptree/NetMF), GAE, [featWalk](https://github.com/xhuang31/FeatWalk_AAAI19), CAN.

Also, we imported several classic dataset, which includes Flickr, ACM, Cora, BlogCatalog.
We will implement more representative NE models continuously. 
Specifically, we welcome other researchers to contribute NE models into this platform.


## Design overview

### Parser system workflow

The system selects datasets and algorithms through the parameters provided by the command system,which can view details by
```bash
    python netBenchmark.py -h
```
optional arguments:

- -h, --help           
  show this help message and exit
-  --dataset {cora,flickr,blogcatalog,citeseer,pubmed,all}      
   select a available dataset (default: all)
-  --method {featwalk,netmf,deepwalk,prone,node2vec,dgi,gae,can_new,can_original,all}         
   The learning method
-  --task_method {task1,task2,task3}       
   The evaluation method
-  --variable_name VARIABLE_NAME        
   The name of features in dataset
-  --training_time TRAINING_TIME (default: 1.4)
   The total training time you want
-  --input_file INPUT_FILE      
   The input datasets you want
-  --tunning_method TUNNING_METHOD      
   The method of parameter tuning.(now includes Random search and tpe search)

An example
```bash
CUDA_VISIBLE_DEVICE="0,1,2,3,4,5" python netBenchmark.py --method=all --dataset=all --task_method=task1 --cuda_device=1 
```


## Design detail
### Dataset class: `Dataset`

All the input datasets inherit from a base class: Dataset.

Path:  ./preprocessing/dataset.py
```python
class Datasets:
    def __init__(self):
        super(Datasets, self).__init__()

    def get_graph(self,variable_name):
        graph = None
        return graph
    
    @classmethod
    def attributed(cls):
        raise NotImplementedError
```
This class aimed at dealing with different format of input source files and return a normalized format result. 
We now have 4 different methods to deal with different source files, which includes `mat`/`txt`/`(tx,ty,x,y)`/`npz`.After that, it all will return a DICT result.
```python
 data={"Network":adj,"Label":labels,"Attributes":feature}
```

### Base class: `Models`
All algorithms model inherit from a base class: ./models/model.py `Models`, which itself inherits from `torch.nn.Module`.
The main idea of this class is to tune parameters and obtain the best result, which will be evaluated by node classification or link prediction according to the different tasks.

#### How to import a new algorithm
we build a super class for algorithms to normalize the input and output.The dataset processed by Dataset.py will become a global variable that can be called as `self.mat_content`.

```python
def train_model(self, **kwargs):
   # need to add the algorithm
   return embedding
```
#### Embedding evaluation

In order to tune parameters according to different tasks,we need to calculate different score for different tasks. 
Meanwhile,train,val and test will be divided from all data before obtain score.
```python
def get_score(self,params):
     emb = self.train_model(**params)
     adj = self.mat_content['Network']
     adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_val_test_edges(adj)
     if self.task_method == 'task1':
         score=link_prediction_Automatic_tuning(emb,edges_pos=val_edges,edges_neg=val_edges_false)
     elif self.task_method == 'task2':
         score = link_prediction_Automatic_tuning(emb, edges_pos=val_edges, edges_neg=val_edges_false)
     else:
         score=node_classifcation_end2end(np.array(emb), self.mat_content['Label'])
    return -score
```
#### Automatic parameter tuning
After score can be obtained from get_score, it can be used to tune hyper-parameters.
for each algorithm's ,its parameters are different, and it will be recorded in `check_train_parameters`.
The third package hyperopt is a reliable package can help us tune parameters.
We choose two methods from it so far,which is random search and TPE respectively.
After tuning and find the best one, it will return the best embbeding and related hyper-parameters we set.
```python
trials = Trials()
if self.tuning == 'random':
    algo = partial(hyperopt.rand.suggest)
else:
    algo = partial(tpe.suggest)
space_dtree = self.check_train_parameters()
best = fmin(
    fn=self.get_score, space=space_dtree, algo=algo, max_evals=150, trials=trials, timeout=self.stop_time)
print(best)
print('end of training:{:.2f}s'.format(self.stop_time))
emb = self.train_model(**best)
return emb,best
```

### Evaluation class: `Evaluation layer`
Two purpose of evaluation layer is to tune the parameters as soon as possible and obtain the final accuracy fairly.
So, we build 2 function for both link prediction and node classification .

- `node_classifcation_10time(feature, labels)` 10-fold Node classification
- `node_classifcation_end2end(feature, labels)` 10-fold Node classification
