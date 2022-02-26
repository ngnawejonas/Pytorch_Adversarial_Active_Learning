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

import scipy.misc as misc
from contextlib import closing
import h5py
import pickle as pkl
import cv2
from typing import Tuple, Any


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
        self.data = torch.cat([self.data, X])
        self.targets = torch.cat([self.targets, y])

    def cat(self, mydataset):
        self.add_data(mydataset.data, mydataset.targets)


def build_mnist(num_sample):
    
    training_data = datasets.MNIST(root="data",
                                          train=True,
                                          download=True, 
                                          transform=ToTensor(),
                                          )

    # Download test data from open datasets.
    test_data = datasets.MNIST(root="data",
                                      train=False,
                                      download=True,
                                      transform=ToTensor(),
                                      )
 
    N = training_data.data.shape[0]
    X_L, unlabelled_data = random_split(training_data, [num_sample, N - num_sample],
                            generator=torch.Generator().manual_seed(42))

    dtl = DataLoader(X_L, batch_size=len(X_L))
    for X,y in dtl:
        labelled_data = MyDataSet(data=X, targets=y)

    return labelled_data, unlabelled_data, test_data
    
##### I use the Fashion_MNIST dataset
def build_fashion_mnist(num_sample):
    raise NotImplementedError()
    
    
def build_svhn(num_sample):
    raise NotImplementedError()

##### I use the CIFAR-10 dataset
def build_cifar(num_sample):
    raise NotImplementedError()

def build_data_func(dataset_name, num_sample):
    dataset_name = dataset_name.lower()
    
    ##### I add 'fashion_mnist' to the list
    assert (dataset_name in ['mnist', 'svhn', 'cifar', 'fashion_mnist']), 'unknown dataset {}'.format(dataset_name)
    labelled = None; unlabelled=None; test=None;
    if dataset_name=='mnist':
        labelled, unlabelled, test = build_mnist(num_sample)
    
    if dataset_name=='fashion_mnist':
        labelled, unlabelled, test = build_fashion_mnist(num_sample)

    if dataset_name=='svhn':
        # TO DO
        labelled, unlabelled, test = build_svhn(num_sample)
    
    if dataset_name=='cifar':
        # TO DO
        labelled, unlabelled, test = build_cifar(num_sample)  
        
    return labelled, unlabelled, test
    
def getSize(dataset_name):
    dataset_name = dataset_name.lower()
    assert (dataset_name in ['mnist', 'svhn', 'cifar', 'bag_shoes', 'quick_draw', 'fashion_mnist']), 'unknown dataset {}'.format(dataset_name)
    
    if dataset_name=='mnist':
        return (1,28,28, 10)

    if dataset_name=='fashion_mnist':
        return (1,28,28,10)
    
    if dataset_name=='svhn':
        return (3,32,32,10)
    
    if dataset_name=='cifar':
        return (3,32,32,10)
        
    if dataset_name=='bag_shoes':
        return (3,64,64,2)
    
    if dataset_name=='quick_draw':
        return (1,28,28, 4)
        
    return None
    