# Imports here
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
from glob import glob
import torch.nn as nn
import argparse

# ---------------------------
# Initiate variables with default values
checkpoint = 'checkpoint.pth'  
arch='resnet152'
image_path = 'flowers/test/100/image_07896.jpg'

# Set up parameters for entry in command line
output_features = 102
learning_rate = 0.001
epochs = 10
save_path = 'checkpoint.pth'
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
parser.add_argument('-o','--output_features',action='store',type=int, help='Select number of output features for the last layer')
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

# In[6]:
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

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
def create_model(arch, output_features=output_features):

    model_full = models.resnet152(pretrained=True)
    model_full = model_full.cuda()
    model_full.fc.out_features = output_features

    for param in model_full.parameters():
        param.requires_grad=False
    
    for param in model_full.fc.parameters():
        param.requires_grad=True
      
    return model_full


#-------------------------------------
#Creates Training Loop
def train(n_epochs, loaders, model, optimizer, criterion, save_path, load_path):
    '''
    Returns trained model and saves weights
    '''
    use_cuda = torch.cuda.is_available()
    valid_loss_min = np.inf
    for epoch in range(1, n_epochs+1):
        #Initializing variables to monitor training and validation
        train_loss=0.0
        valid_loss=0.0
        accuracy=0.0

        #Load model with latest decrease in validation loss or for 1st epoch load specified data in function
        if load_path != None and epoch == 1:
            model.load_state_dict(torch.load(load_path))
            model = model.cuda()
        #elif epoch > 1:
         #   model.load_state_dict(torch.load(save_path))
          #  model = model.cuda()
                
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
            ps = torch.exp(output)
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
            torch.save(model.state_dict(), save_path)
                
        elif valid_loss < valid_loss_min and epoch > 1:
            torch.save(model.state_dict(), save_path)
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
def test_validation(model, dataloaders, criterion, load_path):
    use_cuda = torch.cuda.is_available()
    model.load_state_dict(torch.load(load_path))
    model = model.cuda()
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
model = create_model(arch, output_features)
criterion = nn.CrossEntropyLoss()
optimizer = (torch.optim.Adam(model.fc.parameters(), lr=learning_rate, amsgrad=True))
train(epochs, dataloaders, model, optimizer, criterion, save_path, load_path)
test_validation(model, dataloaders, nn.CrossEntropyLoss(), load_path)

print('-' * 10)
print('Your model has been successfully trained and saved.')
print('-' * 10)