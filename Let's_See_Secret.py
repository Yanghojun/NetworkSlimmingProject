import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torchvision
import torchvision.transforms as transforms
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
import os
import json
import argparse
from models import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='resnet18', help='VGG-16, ResNet-18, LeNet')
    parser.add_argument('--arch', default='vgg', type=str,  help='architecture to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--depth', default=19, type=int,
                        help='depth of the neural network')
    args = parser.parse_args()

    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    load_path = './logs/checkpoint.pth.tar'
    saved_name = os.path.join(os.getcwd(), 'logs','secret')
    load_torch = torch.load(load_path)
    #model.load_state_dict(load_torch)
    #print(load_torch['mask'])
    f = open(saved_name, mode='w')
    f.write("\nstate_dict : \n")
    f.write(str(load_torch['state_dict']))
    f.write("\nepoch : \n")
    f.write(str(load_torch['epoch']))
    f.write("\nbest_prec1 : \n")
    f.write(str(load_torch['best_prec1']))
    f.write("\noptimizer : \n")
    f.write(str(load_torch['optimizer']))