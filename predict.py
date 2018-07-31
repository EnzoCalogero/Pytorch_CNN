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
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import PIL
from PIL import Image

import predict_functions

import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='flowers/test/54/image_05413.jpg',
                        help='filename of the Image to be classified'
                        )
    parser.add_argument('checkpoint',
                        type=str,
                        help='Checkpoint file to be used'
                        )
    parser.add_argument('--top_k',
                        type=int,
                        default=5,
                        help='Number of the top most likely classes to be shown'
                        )
    parser.add_argument('--category_names',
                        type=str,
                        default='cat_to_name.json',
                        help='Dictionary Json file number vs names'
                        )
    parser.add_argument('--gpu',
                        action='store_true',
                        default=False,
                        help='Please Use GPU for Prediction'
                        )

    args=parser.parse_args()
    print("\n\n##########################################################################################")
    print("The Neural Network for the Prediction has been created with the following Parametres:")
    print("The Image for the Prediction is: {}".format(args.input))
    print("The checkpoint with all the details is: {}".format(args.checkpoint))
    print("The Number of the most lively classes to be show is : {}".format(args.top_k))
    print("The Dictionary Json file number vs names is: {}".format(args.category_names))

    if args.gpu:
        print("The model will run use the GPU ".format(args.gpu))
    else:
        print("....Please add the GPU......")
        exit()
    print("##########################################################################################\n\n")
    checkpoint = args.checkpoint
    image_path = args.input
    map_file = args.category_names
    topk = args.top_k

    print("....Running The Prediction... ")
    model = predict_functions.load_checkpoint(filepath=checkpoint)
    print("phase1")
    probs, classes = predict_functions.predict(image_path, model, topk=topk)
    print("phase2")

    with open(map_file, 'r') as f:
        cat_to_name = json.load(f)
    print("Flower Class Name: Associated Probability: ")
    for i in range(0, len(classes)):
        print("Class: {} (Probability: {:.3f})".format(cat_to_name[classes[i]], probs[i]))

    return probs, classes



if __name__ == "__main__":
    main()