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


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = get_model(architecture=checkpoint['artitecture'], hidden=checkpoint['hidden'])

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['image_datasets']

    model.class_to_idx = {model.class_to_idx[k]: k for k in model.class_to_idx}

    return model


def process_image(image):
    img_loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()

    np_image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean) / std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image


def get_model(architecture, hidden):
    from collections import OrderedDict

    if architecture == 'vgg13':
        model = models.vgg16(pretrained=True)
        classifier_input_size = model.classifier[0].in_features
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        classifier_input_size = 1024
    else:
        print("error")
        stop()
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

    model.cuda()
    return model


def predict(image_path, model, topk=5):
    gpu_check = torch.cuda.is_available()
    model.eval()
    np_array = process_image(image_path)
    tensor = torch.from_numpy(np_array)
    var_inputs = Variable(tensor.float().cuda(), volatile=True)
    output = model.forward(var_inputs.unsqueeze(0))

    ps = torch.exp(output).data.topk(topk)
    probs = ps[0].cpu() if gpu_check else ps[0]
    classes = ps[1].cpu() if gpu_check else ps[1]

    classes_ = list()
    for item in classes.numpy()[0]:
        classes_.append(model.class_to_idx[item])

    return probs.numpy()[0], classes_

