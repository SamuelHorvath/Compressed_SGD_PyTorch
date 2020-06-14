import torch
import numpy as np


def prep_grad(x):
    x_flat = torch.unsqueeze(x, 0).flatten()
    dim = x.shape
    d = x_flat.shape[0]
    return x_flat, dim, d


def c_nat(x):
    x, dim, d = prep_grad(x)
    # get 2^n out of input
    h1 = torch.floor(torch.where(x != 0, torch.log2(torch.abs(x)), x))
    h2 = torch.where(x != 0, torch.pow(2, h1), x)
    # extract probability
    p = torch.where(x != 0, torch.div(torch.abs(x) - h2, h2), x)
    # sample random uniform vector
    unif = torch.rand_like(x)
    # generate zero one with probability p
    zero_one = torch.floor(unif + p)
    # generate output
    nat = torch.sign(x) * h2 * (1 + zero_one)
    return nat.reshape(dim)


def random_dithering_opt(x, p, s, natural):
    """
    :param x: vector to quantize
    :param p: norm parameter
    :param s: number of levels
    :param natural: if True, natural dithering is used
    :return: compressed vector
    """
    x, dim, d = prep_grad(x)
    # definition of random dithering
    norm = torch.norm(x, p=p)
    if norm == 0:
        return x.reshape(dim)

    if natural:
        s = int(2 ** (s - 1))
    f = torch.floor(s * torch.abs(x) / norm + torch.rand_like(x))/s

    if natural:
        f = c_nat(f)
    res = torch.sign(x) * f
    k = res * norm
    return k.reshape(dim)


def random_dithering_wrap(p=np.inf, s=2, natural=True):
    def random_dithering(x):
        return random_dithering_opt(x, p=p, s=s, natural=natural)
    return random_dithering


def rand_spars_opt(x, h):
    """
    :param x: vector to sparsify
    :param h: density
    :return: compressed vector
    """
    x, dim, d = prep_grad(x)
    # number of coordinates to keep
    r = int(np.maximum(1, np.floor(d * h)))
    # random vector of r ones and d-r zeros
    mask = torch.zeros_like(x)
    mask[torch.randperm(d)[:r]] = 1
    # just r random coordinates are kept
    t = mask * x * (d/r)
    t = t.reshape(dim)
    return t


def rand_spars_wrap(h=0.1):
    def rand_spars(x):
        return rand_spars_opt(x, h=h)
    return rand_spars


def top_k_opt(x, h):
    """
    :param x: vector to sparsify
    :param h: density
    :return: compressed vector
    """
    x, dim, d = prep_grad(x)
    # number of coordinates kept
    r = int(np.maximum(1, np.floor(d * h)))
    # positions of top_k coordinates
    _, ind = torch.topk(torch.abs(x), r)
    mask = torch.zeros_like(x)
    mask[ind] = 1
    t = mask * x
    t = t.reshape(dim)
    return t


def top_k_wrap(h=0.1):
    def top_k(x):
        return top_k_opt(x, h=h)
    return top_k


def grad_spars_opt(x, h, max_it):
    """
    :param x: vector to sparsify
    :param h: density
    :param max_it: maximum number of iterations of greedy algorithm
    :return: compressed vector
    """
    x, dim, d = prep_grad(x)
    # number of coordinates kept
    r = int(np.maximum(1, np.floor(d * h)))

    abs_x = torch.abs(x)
    abs_sum = torch.sum(abs_x)
    ones = torch.ones_like(x)
    p_0 = r * abs_x / abs_sum
    p = torch.min(p_0, ones)
    for _ in range(max_it):
        p_sub = p[(p != 1).nonzero(as_tuple=True)]
        p = torch.where(p >= ones, ones, p)
        if len(p_sub) == 0 or torch.sum(torch.abs(p_sub)) == 0:
            break
        else:
            c = (r - d + len(p_sub))/torch.sum(p_sub)
        p = torch.min(c * p, ones)
        if c <= 1:
            break
    prob = torch.rand_like(x)
    # avoid making very small gradient too big
    s = torch.where(p <= 10**(-6), x, x / p)
    # we keep just coordinates with high probability
    t = torch.where(prob <= p, s, torch.zeros_like(x))
    t = t.reshape(dim)
    return t


def grad_spars_wrap(h=0.1, max_it=4):
    def grad_spars(x):
        return grad_spars_opt(x, h=h, max_it=max_it)
    return grad_spars


def biased_unbiased_wrap(biased_comp, unbiased_comp):
    def error_quant(x):
        c_1 = biased_comp(x)
        error = x - c_1
        c_2 = unbiased_comp(error)
        return c_1 + c_2
    return error_quant


def combine_two_wrap(comp_1, comp_2):
    def combine(x):
        t_1 = comp_1(x)
        t_2 = comp_2(t_1)
        return t_2
    return combine
