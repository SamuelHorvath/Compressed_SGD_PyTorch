import torch
import numpy as np

from .utils import create_run, update_run, save_run, seed_everything
from .prep_data import create_loaders
from .gen_sgd import SGDGen

RUNS = 5


def train_workers(suffix, model, optimizer, criterion, epochs, train_loader_workers,
                  val_loader, test_loader, n_workers, hpo=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    run = create_run()
    train_loss = np.inf

    best_val_loss = np.inf
    test_loss = np.inf
    test_acc = 0

    for e in range(epochs):
        model.train()
        running_loss = 0
        train_loader_iter = [iter(train_loader_workers[w]) for w in range(n_workers)]
        iter_steps = len(train_loader_workers[0])
        for _ in range(iter_steps):
            for w_id in range(n_workers):
                data, labels = next(train_loader_iter[w_id])
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                running_loss += loss.item()
                optimizer.step_local_global(w_id)
                optimizer.zero_grad()

        train_loss = running_loss/(iter_steps*n_workers)

        val_loss, _ = accuracy_and_loss(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            test_loss, test_acc = accuracy_and_loss(model, test_loader, criterion, device)
            best_val_loss = val_loss

        update_run(train_loss, test_loss, test_acc, run)

        print("Epoch: {}/{}.. Training Loss: {:.5f}, Test Loss: {:.5f}, Test accuracy: {:.2f} "
              .format(e + 1, epochs, train_loss, test_loss, test_acc), end='\r')

    print('')
    if not hpo:
        save_run(suffix, run)

    return best_val_loss


def accuracy_and_loss(model, loader, criterion, device):
    correct = 0
    total_loss = 0

    model.eval()
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        output = model(data)
        loss = criterion(output, labels)
        total_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(loader.dataset)
    total_loss = total_loss / len(loader)

    return total_loss, accuracy


def tune_step_size(exp):
    best_val_loss = np.inf
    best_lr = 0

    seed = exp['seed']
    seed_everything(seed)
    hpo = True

    for lr in exp['lrs']:
        print('Learning rate {:2.4f}:'.format(lr))
        val_loss = run_workers(lr, exp, hpo=hpo)

        if val_loss < best_val_loss:
            best_lr = lr
            best_val_loss = val_loss
    return best_lr


def run_workers(lr, exp, suffix=None, hpo=False):
    dataset_name = exp['dataset_name']
    n_workers = exp['n_workers']
    batch_size = exp['batch_size']
    epochs = exp['epochs']
    criterion = exp['criterion']
    error_feedback = exp['error_feedback']
    momentum = exp['momentum']
    weight_decay = exp['weight_decay']
    compression = get_compression(**exp['compression'])
    master_compression = exp['master_compression']

    net = exp['net']
    model = net()

    train_loader_workers, val_loader, test_loader = create_loaders(dataset_name, n_workers, batch_size)

    optimizer = SGDGen(model.parameters(), lr=lr, n_workers=n_workers, error_feedback=error_feedback,
                       comp=compression, momentum=momentum, weight_decay=weight_decay, master_comp=master_compression)

    val_loss = train_workers(suffix, model, optimizer, criterion, epochs, train_loader_workers,
                             val_loader, test_loader, n_workers, hpo=hpo)
    return val_loss


def run_tuned_exp(exp, runs=RUNS, suffix=None):
    if suffix is None:
        suffix = exp['name']

    lr = exp['lr']

    if lr is None:
        raise ValueError("Tune step size first")

    seed = exp['seed']
    seed_everything(seed)

    for i in range(runs):
        print('Run {:3d}/{:3d}, Name {}:'.format(i+1, runs, suffix))
        suffix_run = suffix + '_' + str(i+1)
        run_workers(lr, exp, suffix_run)


def get_single_compression(wrapper, compression, **kwargs):
    if wrapper:
        return compression(**kwargs)
    else:
        return compression


def get_compression(combine=None, **kwargs):
    if combine is None:
        return get_single_compression(**kwargs)
    else:
        compression_1 = get_single_compression(**combine['comp_1'])
        compression_2 = get_single_compression(**combine['comp_2'])
        return combine['func'](compression_1, compression_2)
