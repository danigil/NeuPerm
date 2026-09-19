import torchvision.models as models
import torch
import torch.nn as nn
import torchvision
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from neu_perm.config import *

datasets_metadata = {
    'mnist': {
        'n_classes': 10,
        'n_channels': 1,
        'imsize': 28
    },
    'fashion_mnist': {
        'n_classes': 10,
        'n_channels': 1,
        'imsize': 28
    },
    'cifar10': {
        'n_classes': 10,
        'n_channels': 3,
        'imsize': 32
    },
    'cifar100': {
        'n_classes': 100,
        'n_channels': 3,
        'imsize': 32
    },
    'imagenet12': {
        'n_classes': 1000,
        'n_channels': 3,
        'imsize': 224
    },
}

def ds(
    ds_func,
    imsize: int,
    batch_size: int,
    num_workers: int=4,

    n_channels: int=1,
    grayscale_to_rgb: bool=False,

    transform_l_train: Optional[List]=None,
    transform_l_test: Optional[List]=None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    if transform_l_train is None and transform_l_test is None:
        transforms_l = [
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*n_channels, (0.5,)*n_channels)
        ]
        
        if grayscale_to_rgb:
            transforms_l.insert(0, transforms.Grayscale(num_output_channels=3))

        data_transform_train = data_transform_test = transforms.Compose(transforms_l)
    else:
        data_transform_train = transforms.Compose(transform_l_train)
        data_transform_test = transforms.Compose(transform_l_test)

    # MNIST datasets
    train_dataset = ds_func(
        root=DATASETS_DIR,
        train=True,
        transform=data_transform_train,
        download=True
    )
    test_dataset = ds_func(
        root=DATASETS_DIR,
        train=False,
        transform=data_transform_test,
        download=True
    )

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers
    )

    return train_loader, test_loader

def mnist(imsize: int,
          batch_size: int,
          num_workers: int=4,
          grayscale_to_rgb: bool=False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    train_loader, test_loader = ds(
        torchvision.datasets.MNIST,
        imsize=imsize,
        batch_size=batch_size,
        num_workers=num_workers,
        grayscale_to_rgb=grayscale_to_rgb,
        n_channels=1,
    )
    return train_loader, test_loader

def fashion_mnist(imsize: int,
          batch_size: int,
          num_workers: int=4,
          grayscale_to_rgb: bool=False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    train_loader, test_loader = ds(
        torchvision.datasets.FashionMNIST,
        imsize=imsize,
        batch_size=batch_size,
        num_workers=num_workers,
        grayscale_to_rgb=grayscale_to_rgb,
        n_channels=1,
    )
    return train_loader, test_loader

def cifar10(
    imsize: int,
    batch_size: int,
    num_workers: int=4,

    grayscale_to_rgb: bool=False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    train_transform_l = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    test_transform_l = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    train_loader, test_loader = ds(
        torchvision.datasets.CIFAR10,
        imsize=imsize,
        batch_size=batch_size,
        num_workers=num_workers,
        n_channels=3,

        transform_l_train=train_transform_l,
        transform_l_test=test_transform_l,
    )
    return train_loader, test_loader

def cifar100(
    imsize: int,
    batch_size: int,
    num_workers: int=4,

    grayscale_to_rgb: bool=False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    train_transform_l = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=imsize, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    test_transform_l = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    train_loader, test_loader = ds(
        torchvision.datasets.CIFAR100,
        imsize=imsize,
        batch_size=batch_size,
        num_workers=num_workers,
        n_channels=3,

        transform_l_train=train_transform_l,
        transform_l_test=test_transform_l,
    )
    return train_loader, test_loader

ds_map = {
    'mnist': mnist,
    'fashion_mnist': fashion_mnist,

    'cifar10': cifar10,
    'cifar100': cifar100,
}

def get_ds(dataset:str, imsize:int, batch_size:int, num_workers:int=4, grayscale_to_rgb:bool=False) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    ds_func = ds_map.get(dataset, None)
    if ds_func is None:
        raise ValueError(f"Dataset {dataset} not supported")
    
    return ds_func(
        imsize=imsize,
        batch_size=batch_size,
        num_workers=num_workers,
        grayscale_to_rgb=grayscale_to_rgb
    )