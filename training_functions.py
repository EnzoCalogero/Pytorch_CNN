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
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import PIL
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


def define_model(artitecture, hidden, learn_rate, gpu):
    from collections import OrderedDict

    if artitecture == 'vgg13':
        model = models.vgg16(pretrained=True)
        classifier_input_size = model.classifier[0].in_features
    elif artitecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        classifier_input_size = 1024
    else:
        print("No Valid Architecture!!!")
        exit()

    hidden = hidden
    classifier_output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden)),
        ('relu1', nn.ReLU()),
        ('drop', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(hidden, classifier_output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    if gpu == True:
        model.cuda()

    return optimizer, model


def train_model(model, learn_rate, epochs, gpu):
    # data needs for training the model

    data_dir = 'flowers'
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

    # Configurations for the Traing

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    epochs = epochs
    steps = 0
    training_loss = 0
    print_every = 50

    # Training Procedure

    for e in range(epochs):
        model.train()
        for images, labels in iter(trainloader):
            steps += 1
            if gpu == True:
                inputs = Variable(images.cuda())
                targets = Variable(labels.cuda())
            else:
                inputs = Variable(images)
                targets = Variable(labels)

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.data[0]

            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                validn_loss = 0
                for ii, (images, labels) in enumerate(validloader):

                    if gpu == True:
                        inputs = Variable(images.cuda(), volatile=True)
                        labels = Variable(labels.cuda(), volatile=True)
                    else:
                        inputs = Variable(images)
                        labels = Variable(labels)

                    output = model.forward(inputs)
                    validn_loss += criterion(output, labels).data[0]
                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(training_loss / print_every),
                      "Validation Loss: {:.3f}.. ".format(validn_loss / len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(validloader)))

                training_loss = 0

                model.train()
    print("Training complete successfully")

def save_status(model, filename, artitecture, hidden):

        checkpoint = {
            'state_dict': model.state_dict(),
            'image_datasets': model.class_to_idx,
            'artitecture': artitecture,
            'hidden': hidden,
            'model': model,
        }

        torch.save(checkpoint, filename)
        print("Saved")
