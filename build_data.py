# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 xx:xx:xx 2022

@author: ngnawejonas

datasets preprocessing
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose

from typing import Tuple, Any


SHOW_LOG = False

def log(msg, *args):
    if SHOW_LOG:
        print(msg, *args)

class MyDataSet(datasets.VisionDataset):
    def __init__(self, data, targets, root='data') -> None:
        super(MyDataSet, self).__init__(root)
        self.data = data
        self.targets= targets
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

    def __len__(self) -> int:
        return len(self.data) 

    def add_data(self, X, y):
        assert len(X) == len(y)
        X , y = X.to(self.data.device), y.to(self.targets.device)
        self.data = torch.cat([self.data, X])
        self.targets = torch.cat([self.targets, y])

    def cat(self, mydataset):
        self.add_data(mydataset.data, mydataset.targets)


def build_(num_sample=-1, datafn=None, seed=False):
    
    training_data = datafn(root="data",
                          train=True,
                          download=True, 
                          transform=ToTensor(),
                          )

    # Download test data from open datasets.
    test_data = datafn(root="data",
                      train=False,
                      download=True,
                      transform=ToTensor(),
                      )

    N = training_data.data.shape[0]
    if seed:
        X_L, unlabelled_data = random_split(training_data, [num_sample, N - num_sample],
                                generator=torch.Generator().manual_seed(42))
    else:
        X_L, unlabelled_data = random_split(training_data, [num_sample, N - num_sample])

    dtl = DataLoader(X_L, batch_size=len(X_L))
    for X,y in dtl:
        labelled_data = MyDataSet(data=X, targets=y)

    return labelled_data, unlabelled_data, test_data, training_data
    

def build_fashion_mnist(num_sample):
    raise NotImplementedError()
    
    
def build_svhn(num_sample):
    raise NotImplementedError()


def build_cifar(num_sample):
    raise NotImplementedError()

def build_data_func(dataset_name, num_sample, seed):
    dataset_name = dataset_name.lower()
    
    labelled = None; unlabelled=None; test=None;
    if dataset_name=='mnist':
        labelled, unlabelled, test, full_train = build_(num_sample, datasets.MNIST, seed=seed)
    
    elif dataset_name=='fashion_mnist':
        labelled, unlabelled, test, full_train = build_(num_sample, datasets.FashionMNIST, seed=seed)

    elif dataset_name=='svhn':
        # TO DO
        labelled, unlabelled, test, full_train = build_svhn(num_sample, seed=seed)
    
    elif dataset_name=='cifar10':
        # TO DO
        labelled, unlabelled, test, full_train = build_(num_sample, datasets.CIFAR10, seed=seed)
    else:
        raise NotImplementedError()
        
    return labelled, unlabelled, test, full_train
    
def getSize(dataset_name):
    dataset_name = dataset_name.lower()
    
    if dataset_name=='mnist':
        return (1,28,28, 10)

    elif dataset_name=='fashion_mnist':
        return (1,28,28,10)
    
    elif dataset_name=='svhn':
        return (3,32,32,10)
    
    elif dataset_name=='cifar10':
        return (3,32,32,10)
        
    elif dataset_name=='bag_shoes':
        return (3,64,64,2)
    
    elif dataset_name=='quick_draw':
        return (1,28,28, 4)
    else:
        raise NotImplementedError('Dataset not handled!')    
