from functools import partial
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision
from typing import Literal, Tuple, Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from neu_perm.data_loaders import datasets_metadata

def vgg(dataset='mnist', type:Literal['vgg11', 'vgg16']='vgg11') -> nn.Module:
    ds_metadata = datasets_metadata[dataset]
    n_classes = ds_metadata['n_classes']
    n_channels = ds_metadata['n_channels']

    if type == 'vgg11':
        model = models.vgg11(weights=None, num_classes=n_classes)
    elif type == 'vgg16':
        model = models.vgg16(weights=None, num_classes=n_classes)

    if n_channels == 1:
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

    return model

vgg11 = partial(vgg, type='vgg11')
vgg16 = partial(vgg, type='vgg16')

def densenet(dataset='mnist', type:Literal['densenet121', 'densenet169']='densenet121') -> nn.Module:
    ds_metadata = datasets_metadata[dataset]
    n_classes = ds_metadata['n_classes']
    n_channels = ds_metadata['n_channels']

    if type == 'densenet121':
        model = models.densenet121(weights=None, num_classes=n_classes)
    elif type == 'densenet169':
        model = models.densenet169(weights=None, num_classes=n_classes)

    if n_channels == 1:
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model

densenet121 = partial(densenet, type='densenet121')
densenet169 = partial(densenet, type='densenet169')

def resnet(dataset='mnist', type:Literal['resnet50', 'resnet101']='resnet50') -> nn.Module:
    ds_metadata = datasets_metadata[dataset]
    n_classes = ds_metadata['n_classes']
    n_channels = ds_metadata['n_channels']

    if type == 'resnet50':
        model = models.resnet50(weights=None, num_classes=n_classes)
    elif type == 'resnet101':
        model = models.resnet101(weights=None, num_classes=n_classes)
    
    if n_channels == 1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model

resnet50 = partial(resnet, type='resnet50')
resnet101 = partial(resnet, type='resnet101')

model_map = {
    'vgg11': vgg11,
    'vgg16': vgg16,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'densenet121': densenet121,
    'densenet169': densenet169,
}

def get_model(model_name:str, dataset:str='mnist') -> nn.Module:
    return model_map[model_name](dataset=dataset)