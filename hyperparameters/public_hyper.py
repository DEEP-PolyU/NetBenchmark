from hyperopt import hp

SPACE_TREE = {
    'batch_size': hp.uniformint('batch_size', 1, 100),
    'nb_epochs': hp.uniformint('nb_epochs', 100, 5000),
    'lr': hp.choice('lr',[0.1, 0.01, 0.001, 0.0001, 0.005, 0.05, 0.00005]),
    'dropout': hp.uniform('dropout', 0, 0.75)
}