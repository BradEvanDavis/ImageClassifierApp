# Imports here
import matplotlib.pyplot as plt
import torch as torch
from torchvision import datasets, transforms, models
import numpy as np
from glob import glob
import torch.nn as nn
import argparse
from collections import OrderedDict
import json
from PIL import Image
import pandas as pd
from IPython.display import display

# ---------------------------
# Initiate variables with default values
arch='resnet152'
checkpoint = 'checkpoint_{}.pth'.format(arch)  
img_path = 'flowers/test/100/image_07896.jpg'
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up parameters for entry in command line
topk=5
gpu=True
save_path = checkpoint
load_path_model = torch.load(checkpoint)['state_dict']
hidden_size =  torch.load(checkpoint)['hidden_size']
class_to_idx = torch.load(checkpoint)['class_to_idx']
output_features = torch.load(checkpoint)['output_size']
path = 'cat_to_name.json'
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
#-------------------------------------------
# Set up parameters for entry in command line
parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir',type=str, help='Location of directory with data for image classifier to train and test')
parser.add_argument('-a','--arch',action='store',type=str, help='Choose among 3 pretrained networks - vgg16, alexnet, and densenet121')
parser.add_argument('-l','--load_path_model',action='store',type=int, help='Select the name of the load_path')
parser.add_argument('-s','--save_path',action='store', type=str, help='Select name of file to save the trained model')
parser.add_argument('-c','--path',action='store', type=str, help='Choose an alternative cat_to_name JSON file if needed')
parser.add_argument('-i', '--img_path',type=str, help='Location of img file you are testing')

args = parser.parse_args()
#-------------------------------------------
# Select parameters entered in command line
if args.data_dir: data_dir = args.data_dir
else: data_dir = data_dir

#------------------------------------------------------------
if args.arch: arch = args.arch
else: arch = arch

if args.img_path: img_path = args.img_path
else: img_path = img_path
    
if args.save_path: save_path = args.save_path
else: save_path = save_path
        
if args.load_path_model: load_path = args.load_path_model
else: load_path_model = load_path_model

if args.path: path = args.path
else: path = path
# In[7]:
# TODO: Define your transforms for the training, validation, and testing sets
#Define normalization for images
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.255)

#Define data transforms
data_transforms = {'train':(transforms.Compose([transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)])),
                  'valid': (transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])),
                  'test': (transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)]))}

#Import Index
# In[6]:

def create_pretrained_model(arch=arch):

    if arch=='resnet152': 
        model_full = models.resnet152(pretrained=True)
        for param in model_full.parameters(): param.requires_grad=False
        for param in model_full.fc.parameters(): param.requires_grad=True
    elif arch=='vgg19': 
        model_full = models.vgg19(pretrained=True)
        for param in model_full.parameters(): param.requires_grad=False
        for param in model_full.classifier.parameters(): param.requires_grad=True
    elif arch=='vgg16':
        model_full = models.vgg16(pretrained=True)
        for param in model_full.parameters(): param.requires_grad=False
        for param in model_full.classifier.parameters(): param.requires_grad=True
    else: print('Please pick from resnet152, vgg19, or vgg16 architectures')

    if gpu: model_full.cuda()
    
    return model_full

#------------------------------------
def create_model(arch=arch, hidden_size=hidden_size, output_features=output_features, save_path=save_path):
    
    pretrained_model = create_pretrained_model(arch)
    
    if arch=='resnet152': 
        input_features = pretrained_model.fc.in_features
    else: 
        input_features = pretrained_model.classifier[0].in_features
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_size)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_size, output_features))
        ]))
    
    if arch=='resnet152': 
        pretrained_model.fc = classifier
    else: 
        pretrained_model.classifier = classifier
    if gpu: pretrained_model.cuda()
    return pretrained_model
#-------------------------------------

# In[17]:
#---------------------------------------
def load_cat_to_name(path=path):

    with open(path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

#---------------------------
def load_model(model, load_path_model=load_path_model):
    model.load_state_dict(load_path_model)
    return model

#----------------------------------------
#Image procesing for inference
def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(img_path)
    
    # Image resized and scaled
    w, h = image.size
    if w < h:
        new_w = 256
        new_h = int(h * float(new_w) / w)
    else:
        new_h = 256
        new_w = int(w * float(new_h) / h)
    
    image = image.resize((new_w, new_h))
    
    # Image cropped and scaled
    left = (new_w - 224) / 2
    right = new_w - (new_w - 224) / 2
    upper = (new_h - 224) / 2
    lower = new_h - (new_h - 224) / 2
    image = image.crop((left, upper, right, lower))
    
    # Image Normalized
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = np.array(image) / 255.0
    img_array = (img_array - mean) /  std
    
    # Transpose image to make it ready to be loaded as a tensor
    img_array = img_array.transpose((2, 1, 0))
    
    return img_array

#---------------------------------------
# Creates labels and probs
def predict(img_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    
    model.eval()
    model.cpu()
    
    # Predict the class from an image file
    processed_image = process_image(img_path)
    image_tensor = torch.from_numpy(np.expand_dims(processed_image, axis=0)).float()
    image_tensor = image_tensor
    
    with torch.no_grad():
        output = model(image_tensor)
    
    ps = torch.nn.functional.softmax(output, dim=1)
    ps = ps.cpu()
    probs, labels = ps.topk(topk)
    
    probs = np.array(probs)[0]
    labels = np.array(labels)[0]
 
    return probs, labels
    

#------------------------------------
# Displays Predictions

def predict_and_display(img_path, model, topk, class_to_idx=class_to_idx, device=device):
    # Display an image along with the top 5 classes
    processed_image = process_image(img_path)
    image_tensor = torch.from_numpy(np.expand_dims(processed_image, axis=0)).float()
    image_tensor = image_tensor.cpu()
    
    model.class_to_idx = class_to_idx

    probs, labels = predict(img_path, model, topk)
    probs = probs * 100
    label_map = {v: k for k, v in model.class_to_idx.items()}
    classes = [cat_to_name[label_map[l]] for l in labels]
    
    df = pd.DataFrame(data=list(zip(classes, probs)), columns=['Flower_Species','Probs'])
    df = df.sort_values(by='Probs', ascending=False)
    return df.head(topk)

#----------------------------------
cat_to_name = load_cat_to_name(path)
model = create_model()
# ---------------------------------
#Display a table along with the top k classes and their respective probs
model = load_model(model, load_path_model)
display(predict_and_display(img_path, model, topk))