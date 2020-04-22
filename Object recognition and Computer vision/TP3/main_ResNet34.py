import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm


# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='detected_bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--cgpu', type=str, default='gpu', metavar='F',
                    help='use CPU or GPU')
parser.add_argument('--augmented', type=bool, default=False, metavar='G',
                    help='use augmented data built from the initial one')
parser.add_argument('--train_all', type=bool, default=False, metavar='H',
                    help='use training + val dataset to train')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading os.path.dirname(os.path.abspath(__file__))

from data_ResNet34 import *


if args.augmented :
    # Modified version loading jointly the original dataset, and the dataset after some modifications on the picture 
    # to expand the database
    train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
        datasets.ImageFolder(args.data + '/train_images',
                             transform=data_transforms),
        datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms_2)]), batch_size=args.batch_size, shuffle=True, num_workers=1)
elif args.train_all :
    # Use all the database for training
    train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
        datasets.ImageFolder(args.data + '/train_images',
                             transform=data_transforms),
        datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms_2),
        datasets.ImageFolder(args.data + '/val_images',
                             transform=data_transforms),
        datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms_2)]), batch_size=args.batch_size, shuffle=True, num_workers=1)

else:
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train_images',
                             transform=data_transforms),
        batch_size=args.batch_size, shuffle=True, num_workers=1)




val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)


# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
#from model import *

           
#loading a pre-trained ResNet
model = torchvision.models.resnet34(pretrained=True)
#Changing the last layer so it matches our case
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 20)        
      
# We freeze all layers except the last one
layer = 0
for name, child in model.named_children():
    layer += 1
    if layer <= 9:
        for name2, params in child.named_parameters():
            params.requires_grad = False  
              
#Loading best model found in the previous step
model.load_state_dict(torch.load('experiment_ResNet34/best_model.pth'))
   
    
    
#modified so it can be chosen in the args
if use_cuda and (args.cgpu).lower()=='gpu':
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

    
optimizer = optim.Adam(model.parameters(), lr=args.lr)



def train(epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda and (args.cgpu).lower()=='gpu':
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    print('Training set: Accuracy: {}/{}'.format( correct, len(train_loader.dataset)))
    torch.save(model.state_dict(), 'state_after_train.pth')  
        
def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda and (args.cgpu).lower()=='gpu':
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
      
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    torch.save(model.state_dict(), 'state_after_val.pth')

