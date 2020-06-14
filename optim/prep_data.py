import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def create_loaders(dataset_name, n_workers, batch_size, seed=42):

    train_data, test_data = load_data(dataset_name)

    train_loader_workers = dict()
    n = len(train_data)

    # preparing iterators for workers and validation set
    np.random.seed(seed)
    indices = np.arange(n)
    np.random.shuffle(indices)

    n_val = np.int(np.floor(0.1 * n))
    val_data = Subset(train_data, indices=indices[:n_val])

    indices = indices[n_val:]
    n = len(indices)
    a = np.int(np.floor(n / n_workers))
    top_ind = a * n_workers
    seq = range(a, top_ind, a)
    split = np.split(indices[:top_ind], seq)

    b = 0
    for ind in split:
        train_loader_workers[b] = DataLoader(Subset(train_data, ind), batch_size=batch_size, shuffle=True)
        b = b + 1

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader_workers, val_loader, test_loader


def load_data(dataset_name):

    if dataset_name == 'mnist':

        transform = transforms.ToTensor()

        train_data = datasets.MNIST(root='data', train=True,
                                    download=True, transform=transform)

        test_data = datasets.MNIST(root='data', train=False,
                                   download=True, transform=transform)
    elif dataset_name == 'cifar10':

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_data = datasets.CIFAR10(root='data', train=True,
                                      download=True, transform=transform)

        test_data = datasets.CIFAR10(root='data', train=False,
                                     download=True, transform=transform)
    elif dataset_name == 'cifar100':
        transform = transforms.ToTensor()  # add extra transforms
        train_data = datasets.CIFAR100(root='data', train=True,
                                       download=True, transform=transform)

        test_data = datasets.CIFAR100(root='data', train=False,
                                      download=True, transform=transform)
    else:
        raise ValueError(dataset_name + ' is not known.')

    return train_data, test_data
