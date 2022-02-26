# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:17:56 2017

@author: mducoffe
"""

from torch import nn
import torchvision.models as models

     
def build_model_func(network_archi, img_size=(1,28,28, 10)):
    
    network_archi = network_archi.lower()
    num_classes = img_size[3]
    img_size = (img_size[0], img_size[1], img_size[2])
    model = None
    assert (network_archi in ['vgg8', 'resnet18', 'alexnet']), ('unknown architecture', network_archi)
    if network_archi == 'vgg8':
        model = None
    if network_archi == 'resnet18':
        model = models.resnet18(num_classes=num_classes)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if network_archi == 'alexnet':
        model = None

    return model

