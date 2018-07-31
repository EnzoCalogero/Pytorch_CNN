import argparse
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, transforms
import numpy as np
#import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import PIL
from PIL import Image

import training_functions

import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                        type=str,
                        default='flowers',
                        help='Root folder were all the training images are located.'
                        )
    parser.add_argument('--save_dir',
                        type=str,
                        default='ckPoint',
                        help='Directory where the checkpoints are saved'
                        )
    parser.add_argument('--arch',
                        dest='architecture',
                        default='vgg13',
                        action='store',
                        choices=['vgg13', 'densenet121'],
                        help='CNN Architecture type: vgg13, densenet121'
                        )
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='Value of the Learning Rate (default 0.01)'
                        )
    parser.add_argument('--hidden_units',
                        type=int,
                        default=512,
                        help='Number of hidden units'
                        )
    parser.add_argument('--epochs',
                        type=int,
                        default=6,
                        help='Number of Epoch for the training'
                        )
    parser.add_argument('--gpu', dest='gpu',
                        default=False,
                        action='store_true',
                        help='Please Use GPU for training'
                        )
    args=parser.parse_args()
    print("\n\n##########################################################################################")
    print("The Neural Network has been created with the following Parametres:")
    print("The data for the training is in the folder: {}".format(args.data_dir))
    print("The checkpoint with the save details of the network will be saved in the folder: {}".format(args.save_dir))
    print("The CCN Architecture of the is: {}".format(args.architecture))
    print("The Learning Rate is: {}".format(args.learning_rate))
    print("The number of Hidden Units is: {}".format(args.hidden_units))
    print("The model will run for  {} epochs ".format(args.epochs))
    if args.gpu:
        print("The model will run use the GPU ".format(args.gpu))
    else:
        print("....Please add the GPU......")
        exit()
    print("##########################################################################################\n\n")
    print(args.data_dir)
    artitecture = args.architecture
    hidden = args.hidden_units
    learn_rate = args.learning_rate
    epochs = args.epochs
    filename = args.save_dir
    gpu = True
    data_dir = args.data_dir

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    batch_size = 32

    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    print("start")
    optimizer, model = training_functions.define_model(artitecture=artitecture, hidden=hidden, learn_rate=learn_rate, gpu=True)
    print("phase1")
    training_functions.train_model(model=model, learn_rate=learn_rate, epochs=epochs, gpu=True)
    print("phase2")
    model.class_to_idx = train_data.class_to_idx
    training_functions.save_status(model=model, filename=filename, artitecture=artitecture, hidden=hidden)
    print("phase3")
    
    
if __name__ == "__main__":
     main()
    
