# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from workspace_utils import active_session
import numpy as np
import numpy
from PIL import Image
import seaborn as sns

arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}

def Load_Data(loc="./flowers"):
    data_dir = loc
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(260),transforms.CenterCrop(225),transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(260),transforms.CenterCrop(225),transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)

    valid_datasets=datasets.ImageFolder(valid_dir,transform=valid_transforms)

    test_datasets=datasets.ImageFolder(test_dir,transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loaders = torch.utils.data.DataLoader(train_datasets,batch_size=32,shuffle=True)

    valid_loaders = torch.utils.data.DataLoader(valid_datasets,batch_size=32)

    test_loaders = torch.utils.data.DataLoader(test_datasets,batch_size=32)
    
    return train_loaders,valid_loaders,test_loaders,train_datasets

def setup_para(architect='vgg16',hidden_layer1 = 5024, learningr = 0.001,dropout=0.2,power='gpu'):
    # TODO: Build and train your network
    device = torch.device("cuda" if torch.cuda.is_available() and power =='gpu' else "cpu")
    
    if architect == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architect == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif architect == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("sorry,but did you mean vgg16,densenet121,or alexnet?")
        
    for param in model.parameters():
        param.requires_grad = False
        
    
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(arch[architect], hidden_layer1)),
                            ('relu1', nn.ReLU()),
                            ('drop1',nn.Dropout(0.3)),
                            ('fc2', nn.Linear(hidden_layer1, 1024)),
                            ('relu2',nn.ReLU()),
                            ('drop2',nn.Dropout(0.3)),
                            ('fc3', nn.Linear(1024, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    model.to(device);
    
    return model,criterion,device,hidden_layer1

def train_network(model,criterion,device,train_loaders,valid_loaders,epochs = 14,learningr = 0.001):
    optimizer = optim.Adam(model.classifier.parameters(), lr=learningr)
    steps=0
    train_losses, valid_losses = [], []
    with active_session():
        for epoch in range(epochs):
            running_loss = 0
            for inputs, labels in train_loaders:
                steps+=1
                inputs, labels = inputs.to(device), labels.to(device)
        
                optimizer.zero_grad()
        
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
        
                    
                running_loss +=loss.item()
                if steps%60==0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
        
                    with torch.no_grad():
            
                        for images, labels in valid_loaders:
                            images, labels = images.to(device), labels.to(device)
                            logps= model(images)
                            batch_loss = criterion(logps, labels)
                
                            valid_loss+=batch_loss.item()
                    
                            ps = torch.exp(logps)
                            top_prob, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor))
                
                        train_losses.append(running_loss/len(train_loaders))
                        valid_losses.append(valid_loss/len(valid_loaders))
        
                        model.train()

                    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/len(train_loaders)),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(valid_loaders)),
                      "Valid Accuracy: {:.3f}".format(accuracy/len(valid_loaders)))
    return model,optimizer
                
def save_checkpoint(train_datasets,model,optimizer,path='MASTER_CHECKPOINT.pth',architect='vgg16',epochs=14,learningr = 0.001):
    
    checkpoint={
        'lr': learningr,
        'epochs': epochs,
        'optimizer': optimizer.state_dict,
        'arch': architect,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'class_to_idx': train_datasets.class_to_idx
    }
    torch.save(checkpoint, path)
    