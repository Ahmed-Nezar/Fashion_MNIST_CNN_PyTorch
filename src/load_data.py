import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets


# load data
train_data = datasets.FashionMNIST(root='../input', train=True, download=True, transform=transforms.ToTensor())
validation_data = datasets.FashionMNIST(root='../input', train=False, download=True, transform=transforms.ToTensor())



