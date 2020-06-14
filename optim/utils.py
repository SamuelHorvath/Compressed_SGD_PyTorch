import numpy as np
import os
import glob
import random
import torch
from pickle import load, dump

SAVED_RUNS_PATH = 'saved_data/'
EXP_PATH = 'exps_setup/'


def save_run(suffix, run):
    if not os.path.isdir(SAVED_RUNS_PATH):
        os.mkdir(SAVED_RUNS_PATH)

    file = SAVED_RUNS_PATH + suffix + '.pickle'
    with open(file, 'wb') as f:
        dump(run, f)


def read_all_runs(exp, suffix=None):
    if suffix is None:
        suffix = exp['name']

    runs = list()
    runs_files = glob.glob(SAVED_RUNS_PATH + suffix + '_' + '[1-9]*.pickle')  # reads at most first ten runs
    for run_file in runs_files:
        runs.append(read_run(run_file))
    return runs


def read_run(file):
    with open(file, 'rb') as f:
        run = load(f)
    return run


def create_run():
    run = {'train_loss': [],
           'test_loss': [],
           'test_acc': []
           }
    return run


def update_run(train_loss, test_loss, test_acc, run):
    run['train_loss'].append(train_loss)
    run['test_loss'].append(test_loss)
    run['test_acc'].append(test_acc)


def save_exp(exp):
    if not os.path.isdir(EXP_PATH):
        os.mkdir(EXP_PATH)

    file = EXP_PATH + exp['name'] + '.pickle'
    with open(file, 'wb') as f:
        dump(exp, f)


def load_exp(exp_name):
    file = EXP_PATH + exp_name + '.pickle'
    with open(file, 'rb') as f:
        exp = load(f)
    return exp


def create_exp(name, dataset, net, n_workers, epochs, seed, batch_size, lrs, compression, error_feedback, criterion,
               master_compression=None, momentum=0, weight_decay=0):
    exp = {
        'name': name,
        'dataset_name': dataset,
        'net': net,
        'n_workers': n_workers,
        'epochs': epochs,
        'seed': seed,
        'batch_size': batch_size,
        'lrs':  lrs,
        'lr': None,
        'compression': compression,
        'master_compression': master_compression,
        'error_feedback': error_feedback,
        'criterion': criterion,
        'momentum': momentum,
        'weight_decay': weight_decay
            }
    return exp


def seed_everything(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
