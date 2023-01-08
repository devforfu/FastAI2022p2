# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_autograd.ipynb.

# %% auto 0
__all__ = ['MLP', 'softmax', 'ce', 'log_softmax', 'Dataset', 'Sampler', 'SequentialSampler', 'RandomSampler', 'DataLoader', 'acc',
           'Optimizer']

# %% ../nbs/03_autograd.ipynb 3
from dataclasses import dataclass, field
from functools import reduce
from typing import Protocol

# %% ../nbs/03_autograd.ipynb 4
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% ../nbs/03_autograd.ipynb 5
from tensorviewer import tv
from utils import get_mnist

# %% ../nbs/03_autograd.ipynb 12
class MLP(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n_i, n_h), nn.ReLU(), nn.Linear(n_h, n_o)])
    def forward(self, x):
        return reduce(lambda x, l: l(x), self.layers, x)

# %% ../nbs/03_autograd.ipynb 16
def softmax(t):
    p = torch.exp(t)
    return p / p.sum(1, keepdim=True)

# %% ../nbs/03_autograd.ipynb 19
def ce(logits, targets):
    probs = softmax(logits)
    probs = probs.gather(dim=1, index=targets.view(-1, 1)).squeeze(1)
    return -probs.log().mean()

# %% ../nbs/03_autograd.ipynb 22
def log_softmax(t):
    t.sub_(t.max(dim=1, keepdim=True)[0])
    t.sub_(t.exp().sum(dim=1, keepdim=True).log())
    return t

# %% ../nbs/03_autograd.ipynb 23
def ce(logits, targets):
    return -log_softmax(logits).gather(dim=1, index=targets.view(-1, 1)).squeeze(1).mean()

# %% ../nbs/03_autograd.ipynb 26
class Dataset:
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, index): return self.x[index], self.y[index]

# %% ../nbs/03_autograd.ipynb 30
class Sampler(Protocol):
    def get_idx(self, dataset: Dataset): pass

@dataclass
class SequentialSampler(Sampler):
    def get_idx(self, dataset: Dataset): return np.arange(len(dataset))

@dataclass
class RandomSampler(Sampler):
    def get_idx(self, dataset: Dataset): 
        idx = SequentialSampler().get_idx(dataset)
        np.random.shuffle(idx)
        return idx
    
@dataclass
class DataLoader:
    dataset: Dataset
    bs: int = 1
    shuffle: bool = False
    sampler: Sampler = None
    
    def __post_init__(self):
        if self.sampler is None:
            self.sampler = (RandomSampler if self.shuffle else SequentialSampler)()
    
    def __iter__(self):
        idx = self.sampler.get_idx(self.dataset)
        for i in range(0, len(idx), self.bs):
            yield self.dataset[idx[i:i+self.bs]]
            
    def __len__(self): return int(np.ceil(len(self.dataset) // self.bs))

# %% ../nbs/03_autograd.ipynb 34
def acc(x, y): return (x == y).float().mean()

# %% ../nbs/03_autograd.ipynb 35
class Optimizer:
    def __init__(self, params, lr): 
        self.params, self.lr = list(params), lr
    def step(self): 
        for param in self.params:
            param.data -= self.lr * param.grad
    def zero_grad(self):
        for param in self.params:
            param.grad.zero_()
