from hyperopt import hp

SPACE_TREE = {
    'batch_size': hp.uniformint('batch_size', 1, 100),
    'nb_epochs': hp.uniformint('nb_epochs', 100, 1500),
    'lr': hp.choice('lr', [0.1, 0.01, 0.001, 0.0001, 0.005, 0.05, 0.00005]),
    'dropout': hp.choice('dropout', [0, 0.05, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    'beta': hp.choice('beta', [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.005, 0.05, 0.00005]),
    'alpha': hp.choice('alpha', [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.005, 0.05, 0.00005]),
}