import torch 
from torch import nn
import numpy as np
import json
import network
from PIL import Image
import argparse
from torchvision import transforms


def load_model(filepath):
    checkpoint = torch.load(filepath)
    model = network.Model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def prepare_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    # https://discuss.pytorch.org/t/applying-transforms-to-a-single-image/56254
    image = transform(image)
    return image

def predict(image, model, topk, use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
        model.to_device(device)
        image = image.to(device)
    else:
        device = torch.device('cpu')
    # Turn off dropout   
    model.model.eval()
    
    with torch.no_grad():
        # Match the tensor shape to the tensor used for training model
        image = image.view(-1,image.shape[0],image.shape[1],image.shape[2])

        # Make prediction
        logps = model.forward(image)

        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(logps)
        # Class with highest probability is our predicted class, compare with true label
        probs, indices = ps.topk(topk, dim=1)

        # convert indices to actual category names
        # reverse the key: value pair
        idx_to_class = {v: k for k, v in model.class_to_idx.items()} 
        classes = [idx_to_class[i.item()] for i in indices[0]]

        # Get probabilities for each class
        probs = [i.item() for i in probs[0]]
    
    return probs,classes

def class_to_name(json_path, classes):
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    
    names = [cat_to_name[c] for c in classes]
    
    return names
    
parser = argparse.ArgumentParser(description='Image Classifier.')

parser.add_argument('imagepath', type=str, help='the path for import image')
parser.add_argument('filepath', type=str, help='filepath for checkpoint')
parser.add_argument('--top_k', type=int,help='the number of highly possile classes')
parser.add_argument('--gpu', action='store_true', help='use gpu or not')
parser.add_argument('--category_names', type=str,
              help='the path of mapping categoty name with predicted class')

args = parser.parse_args()

file_path = args.filepath
image_path = args.imagepath
top_k = args.top_k if args.top_k else 1
use_gpu = True if args.gpu else False
cat_to_name_path = args.category_names

model = load_model(file_path)
image = prepare_image(image_path)
probs, classes = predict(image, model, top_k, use_gpu)

print(f'predicted probability: {probs}')
if cat_to_name_path:
    names = class_to_name(cat_to_name_path, classes)
    print(f'predicted category: {names}')
else:
    print(f'predicted class: {classes}')
 
