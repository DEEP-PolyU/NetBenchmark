# NetBenchmark
Node Representation Learning Benchmark


| Status        | Developing      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | QIAN Zhiqiang (zhiqiang.qian@connect.polyu.hk), QUO Zhihao (zhi-hao.guo@connect.polyu.hk), YU Rico (ricoyu.yu@connect.polyu.hk) , LIAN, Amber (amber.lian@connect.polyu.hk) |
| **Updated**   | 2021-03                                           |


## Objective

We aim at building a auto,fair and systematic evaluation platform to compare the results of different Network Embedding models. 
The implemented or modified models include [DeepWalk](https://github.com/phanein/deepwalk),  [node2vec](https://github.com/aditya-grover/node2vec), 
[GraRep](https://github.com/ShelsonCao/GraRep), 
[GCN](https://github.com/tkipf/gcn), [NetMF](https://github.com/xptree/NetMF), GAE, featWalk, CAN and DGI.

Also, we imported several classic dataset, which includes Flickr, ACM, Cora, BlogCatalog.
We will implement more representative NE models continuously. 
Specifically, we welcome other researchers to contribute NE models into this platform.


## Design overview

### Parser system workflow

The system selects data sets and algorithms through the parameters provided by the command system,which can view detailes by
```bash
    python netBenchmark.py -h
```

After choosing the dataset in `datasetdict` and methods in `modeldict`, the parser system will run the following code.

```python
Graph = datasetdict[args.dataset]
Graph=Graph.get_graph(Graph,variable_name= args.variable_name or 'network' )

model=modeldict[args.method]
model(Graph, args.evaluation)
```



## Design detail

### Dataset class: `Dataset`

All the input datasets inherit from a base class: Dataset.

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
Besides, the input file can be divided into 2 kinds of format,which is the mat file(can be parsed through `scipy.io.loadmat`) and others files(create the special code to parse).

### Base class: `Models`
All algorithms model inherit from a base class: `Models`, which itself inherits from `torch.nn.Module`.
The main idea of this class is to tune parameters and obtain the best result, which will be evaluated by node classification.

```python
class Models(torch.nn.Module):
    def __init__(self, *, output=None, save=True, **kwargs)
    
    @classmethod
    def check_train_parameters(cls, **kwargs):
        return kwargs
    
    @classmethod
    def is_preprocessing(cls):
        raise NotImplementedError
    
    @classmethod
    def is_epoch(cls):
        raise NotImplementedError
    
    def forward(self, graph, **kwargs)
    
    def train_model(self, rootdir, **kwargs):
        raise NotImplementedError
    
    def preprocessing(self,filename)
```
The following 3 methods should be overridden:

- `__init__` constructor of the layer, used to configure its behavior.

- `is_preprocessing(cls)` Determine whether the model needs to be preprocessed, if it needs to be preprocessed, jump to the preprocessing function for processing

- `is_epoch(cls)` Determine whether an epoch is involved

- `train_model(self, rootdir, **kwargs)`Training according to different models

#### Automatic parameter tuning
Each algorithm's parameters are different, and it will be recorded in `check_train_parameters`
```python
space_dtree=self.check_train_parameters()
trials = Trials()
algo = partial(tpe.suggest)
best = fmin(
    fn=self.get_score, space=space_dtree, algo=algo, max_evals=150, trials=trials)
```

### Evaluation class: `Evaluation layer`

- `node_classifcation(feature, labels)` 10-fold Node classification

- `link_prediction(emb_name,variable_name, edges_pos, edges_neg)`

Examples:
```python
def parse(**kwargs):
    return Evaluation, Graph_Filedir, Model

# parsing
args = {x: y for x, y in args.__dict__.items() if y is not None}
Evaluation, Graph_Filedir, Model = parse(**args)

# train
res = model.train_model(rootdir,**train_args)

# evaluation
if(Evaluation="node_classifcation"):
    node_classifcation()
if(Evaluation="link_prediction"):
    link_prediction()
```