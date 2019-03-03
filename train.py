# Imports here
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
import numpy as np
from glob import glob
import torch.nn as nn
import argparse
from collections import OrderedDict
import json

# ---------------------------
# Initiate variables with default values
arch='vgg19'
checkpoint = 'checkpoint_{}.pth'.format(arch)  
image_path = 'flowers/test/100/image_07896.jpg'

# Set up parameters for entry in command line
hidden_size = 1024
output_features = 102
learning_rate = 0.001
epochs = 10
gpu=True
save_path = checkpoint
load_path = None
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Set up parameters for entry in command line
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_dir', type=str, help='Location of directory with data for image classifier to train and test')
parser.add_argument('-a','--arch',action='store',type=str, help='Choose either resnet152, vgg19, or vgg16 pre-trained torch models to use for transfer learning')
parser.add_argument('-l','--load_path',action='store',type=str, help='Select the name of the load_path')
parser.add_argument('-o','--output_features',action='store',type=int, help='Select number of output features for the last layer')
parser.add_argument('-hs','--hidden_size',action='store',type=int, help='Select classifier hidden size of classifier layers via a list')
parser.add_argument('-lr','--learning_rate',action='store',type=float, help='Choose a float number as the learning rate for the model')
parser.add_argument('-e','--epochs',action='store',type=int, help='Choose the number of epochs you want to perform gradient descent')
parser.add_argument('-s','--save_path',action='store', type=str, help='Select name of file to save the trained model')
parser.add_argument('-g','--gpu',action='store_true',help='Use GPU if available')

args = parser.parse_args()

# Select parameters entered in command line
if args.arch: arch = args.arch
else: arch = arch

if args.output_features: output_features = args.output_features
else: output_features = output_features

if args.learning_rate: learning_rate = args.learning_rate
else: learning_rate = learning_rate

if args.epochs: epochs = args.epochs
else: epochs=epochs
    
if args.gpu: device=torch.cuda.is_available()

if args.save_path: save_path = args.save_path
else: save_path = save_path
        
if args.load_path: load_path = args.load_path
else: load_path = load_path

if args.data_dir: data_dir = args.data_dir
else: data_dir = data_dir

if args.hidden_size: hidden_size = args.hidden_size
else: hidden_size = hidden_size
 
# ------------------------------
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
#-------------------------------------------------

with open('cat_to_name.json', 'r') as cats:
    cat_to_name = json.load(cats)

batch_size = 64
num_workers = 0

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
test_dataset= datasets.ImageFolder(test_dir, transform=data_transforms['test'])

for i in [train_dir, test_dir, valid_dir]:
    for key in datasets.ImageFolder(i).class_to_idx.keys():
        datasets.ImageFolder(i).class_to_idx[str(key)] = cat_to_name[str(key)]

#define dataloaders
dataloaders = {'train': (torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)),
              'valid': (torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)),
              'test': (torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True))}


# ---------------------------------
# Creates model for transfer learning
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
def create_model(arch=arch, hidden_size=hidden_size, output_features=output_features):
    
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

#Save Checkpoint
def save_checkpoint(model, epoch, save_path=save_path, arch=arch):
    
    model.class_to_idx = train_dataset.class_to_idx
    if arch=='resnet152':
        checkpointRes = {'input_size': model.fc[0].in_features,
            'output_size': model.fc[-1].out_features,
            'hidden_size': hidden_size,
            'batch_size': batch_size,
            'learning_rate': optimizer.param_groups[-1]['lr'],
            'model_name': 'ImageClassifier',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'class_to_idx': model.class_to_idx}
        torch.save(checkpointRes, save_path)
    
    else:
        checkpointVGG = {'input_size': model.classifier[0].in_features,
            'output_size': model.classifier[-1].out_features,
            'hidden_size': hidden_size,
            'batch_size': batch_size,
            'learning_rate': optimizer.param_groups[-1]['lr'],
            'model_name': 'ImageClassifier',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'class_to_idx': model.class_to_idx}
        torch.save(checkpointVGG, save_path)

#------------------------------------
#Creates Training Loop

def train(n_epochs, loaders, model, optimizer, criterion, save_path, load_path, gpu):
    '''
    Returns trained model and saves weights
    '''
    use_cuda = torch.cuda.is_available()
    if gpu or use_cuda:
        gpu=True
    
    valid_loss_min = np.inf
    if load_path != None:
         checkpoint=torch.load(load_path)
        
    for epoch in range(1, n_epochs+1):
        #Initializing variables to monitor training and validation
        train_loss=0.0
        valid_loss=0.0
        accuracy=0.0
        
        #Load model with latest decrease in validation loss or for 1st epoch load specified data in function
        if load_path != None:
            epoch = checkpoint['epoch']+epoch
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
              
        #Model Training
        model.train()
        print('Starting Training')
        for batch_idx, (data, target) in enumerate(loaders['train']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output=model.forward(data)
            loss=criterion(output, target)
            loss.backward()
            optimizer.step()
            
            #Records average loss across batches
            train_loss = train_loss + ((1/ (batch_idx+1)) * (loss.data - train_loss))
           
            #accuracy
            ps = torch.nn.functional.softmax(output, dim=1)
            top_p, top_class = ps.topk(1,dim=1)
            equals = top_class == target.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.cuda.FloatTensor)).item()
            #prints stats every n batches within epoch
            if (batch_idx) % 100 == 0:
                print('Epoch: %d, Batch: %d, Loss: %.6f, Accuracy: %.1f' % (epoch, batch_idx+1, train_loss, accuracy))
        
        #check loss of test set vs loss of validation set    
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            
            #validation loss
            valid_loss = valid_loss + ((1/(batch_idx+1)) * (loss.data - valid_loss))
                
        # print training/validation statistics.  Model only saves when validation loss decreases vs prev run 
        if epoch == 1:
            valid_loss_min = valid_loss
            save_checkpoint(model, save_path=save_path, epoch=epoch)
                
        elif valid_loss < valid_loss_min and epoch > 1:
            save_checkpoint(model, save_path='checkpoint.pth',epoch=epoch)
            print('Validation Loss Decreased ({:.6f} --> {:.6f}).  Model Saved...'.format(
                valid_loss_min,
                valid_loss))
            valid_loss_min = valid_loss

                
        elif valid_loss > valid_loss_min and epoch > 1:
            print('Validation Loss Increased ({:.6f} --> {:.6f}).  Model Not Saved...'.format(
                valid_loss_min,
                valid_loss))
                
        #To seperate batch printouts 
        print('----------------------------------------------------------------------------------------')
    
    print('Epoch: %d, Batch: %d, Loss: %.6f, Accuracy: %.1f' % (epoch, batch_idx+1, train_loss, accuracy))
    # return trained model
    return model

# --------------------------------------------
# validation on the test set
def test_validation(model, dataloaders, checkpoint=checkpoint):
    
    checkpoint=torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
   
    test_loss = 0
    test_accuracy = 0
    total = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(dataloaders['test']):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            
            #test loss
            test_loss = test_loss + ((1/(batch_idx+1)) * (loss.data - test_loss))
            
        #accuracy
        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        test_accuracy = (100. * correct / total)
            
    return print('Loss: %.2f, Accuracy: %.1f' % (test_loss, test_accuracy))
#------------------------------------
# Runs Appropriate Functions
pretrained_model = create_pretrained_model(arch)
model = create_model()

if arch == 'resnet152': 
    fc_params = model.fc.parameters()
else: 
    fc_params = model.classifier.parameters()

criterion = nn.CrossEntropyLoss()
optimizer = (torch.optim.Adam(fc_params, lr=learning_rate, amsgrad=True))
train(epochs, dataloaders, model, optimizer, criterion, save_path, load_path, gpu)
test_validation(model, dataloaders, checkpoint)

print('-' * 10)
print('Your model has been successfully trained and saved.')
print('-' * 10)