# In[1]:
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
import numpy as np
import argparse
from PIL import Image
import json
import pandas as pd
from IPython.display import display

# Initiate variables with default values
output_features = 102
topk = 5
checkpoint = 'checkpoint.pth'   
arch='resnet152'
img_path = 'flowers/test/100/image_07896.jpg'

# Set up parameters for entry in command line
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_path = 'checkpoint.pth'

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Set up parameters for entry in command line
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_dir', type=str, help='Location of directory with data for image classifier to train and test')
parser.add_argument('-a','--arch',action='store',type=str, help='Choose pre-trained torch model to use for transfer learning')
parser.add_argument('-l','--load_path',action='store',type=str, help='Select the name of the load_path')
parser.add_argument('-img','--img_path',action='store', type=str,help='Choose the path of the image you wish to run inference on')
parser.add_argument('-topk','--topk',action='store', type=int, help='Select How Many Results you want printed')
parser.add_argument('-o','--output_features',action='store',type=int, help='Select number of output features for the last layer')

args = parser.parse_args()

# Select parameters entered in command line
if args.arch: arch = args.arch
else: arch = arch

if args.topk: topk = args.topk
else: topk = topk
        
if args.load_path: load_path = args.load_path
else: load_path = checkpoint

if args.data_dir: data_dir = args.data_dir
else: data_dir = data_dir
    
if args.img_path: img_path = args.img_path
else: img_path = img_path
    
if args.output_features: output_features = args.output_features
else: output_features = output_features

# In[7]:
#---------------------------------------
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir)
valid_dataset = datasets.ImageFolder(valid_dir)
test_dataset= datasets.ImageFolder(test_dir)

for i in [train_dir, test_dir, valid_dir]:
    for key in datasets.ImageFolder(i).class_to_idx.keys():
        datasets.ImageFolder(i).class_to_idx[str(key)] = cat_to_name[str(key)]

#---------------------------
def load_model(arch, load_path=None, output_features=output_features):

    model_full = models.resnet152(pretrained=True)
    #model_full = model_full.cpu()
    model_full.fc.out_features = output_features
    model_full.load_state_dict(torch.load(load_path))
    
    class_keys=[datasets.ImageFolder(train_dir).class_to_idx.keys(),
              datasets.ImageFolder(test_dir).class_to_idx.keys(), 
              datasets.ImageFolder(valid_dir).class_to_idx.keys()]

    for i in [train_dir, test_dir, valid_dir]:
        for key in datasets.ImageFolder(i).class_to_idx.keys():
            datasets.ImageFolder(i).class_to_idx[str(key)] = cat_to_name[str(key)]
        model_full.class_to_idx = datasets.ImageFolder(i).class_to_idx
    return model_full


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
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.cuda()
    
    # Predict the class from an image file
    processed_image = process_image(img_path)
    image_tensor = torch.from_numpy(np.expand_dims(processed_image, axis=0)).float()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
    
    probs, labels = output.topk(topk)
    probs = np.array(probs.exp().data)[0]
    labels = np.array(labels)[0]
 
    return probs, labels


#------------------------------------
# Displays Predictions

def predict_and_display(img_path, model, topk):
    # Display an image along with the top 5 classes
    
    processed_image = process_image(img_path)
    image_tensor = torch.from_numpy(np.expand_dims(processed_image, axis=0)).float()
    image_tensor = image_tensor.to(device)
    
    probs, labels = predict(img_path, model, topk)
    label_map = {v: k for k, v in model.class_to_idx.items()}
    classes = [cat_to_name[label_map[l]] for l in labels]
    class_ticks = np.arange(len(classes))
    
    df = pd.DataFrame(data=list(zip(classes, probs)), columns=['Flower_Species','Probs'])
    df = df.sort_values(by='Probs', ascending=False)
    return df.head(topk)

# ---------------------------------
#Display a table along with the top k classes and their respective probs
model = load_model(arch, load_path, output_features)
display(predict_and_display(img_path, model, topk))