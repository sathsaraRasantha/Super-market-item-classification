import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt 
import torch.nn.functional as F 
import torch 
import numpy as np 
from torchvision import transforms as T,datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch import nn
import torch.nn.functional as F
import timm # PyTorch Image Models
from torchsummary import  summary
import config


def imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

def copy_images(imagePaths, folder):

	if not os.path.exists(folder):
		os.makedirs(folder)
	
	for path in imagePaths:
		
		imageName = path.split(os.path.sep)[-1]
		label = path.split(os.path.sep)[-2]
		labelFolder = os.path.join(folder, label)
		
		if not os.path.exists(labelFolder):
			os.makedirs(labelFolder)
		
		destination = os.path.join(labelFolder, imageName)
		shutil.copy(path, destination)

def TrainValDataSplit():
    items =os.listdir(config.data_dir)
    for item in items:
        imagePaths = list(imlist(config.data_dir + '/'+item))
        np.random.shuffle(imagePaths)
        valPathsLen = int(len(imagePaths) * 0.2)
        trainPathsLen = len(imagePaths) - valPathsLen
        trainPaths = imagePaths[:trainPathsLen]
        valPaths = imagePaths[trainPathsLen:]
        copy_images(trainPaths, config.train_valid_splited_data_path+ '/train')
        copy_images(valPaths, config.train_valid_splited_data_path+'/validation')

def show_image(image,label,get_denormalize = True):
    
    image = image.permute(1,2,0)
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    
    if get_denormalize == True:
        image = image*std + mean
        image = np.clip(image,0,1)
        plt.imshow(image)
        plt.title(label)
        
    else: 
        plt.imshow(image)
        plt.title(label)

def show_grid(image,title = None):
    
    image = image.permute(1,2,0)
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    
    image = image*std + mean
    image = np.clip(image,0,1)
    
    plt.figure(figsize=[15, 15])
    plt.imshow(image)
    if title != None:
        plt.title(title)


def accuracy(y_pred,y_true):
    y_pred = F.softmax(y_pred,dim = 1)
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


def view_classify(image,ps,label):
    
    path = config.train_valid_splited_data_path+"/train"
    class_name = os.listdir(path)
    classes = np.array(class_name)

    ps = ps.cpu().data.numpy().squeeze()
    
    image = image.permute(1,2,0)
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    
    
    image = image*std + mean
    img = np.clip(image,0,1)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(8,12), ncols=2)
    ax1.imshow(img)
    ax1.set_title('Ground Truth : {}'.format(class_name[label]))
    ax1.axis('off')
    ax2.barh(classes, ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Predicted Class')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    return None



class CFG:
  epochs =config.epochs                         # No. of epochs for training the model
  lr = config.learning_rate                            # Learning rate
  batch_size = config.batch_size                        # Batch Size for Dataset

  model_name = 'efficientnet_b3'    # Model name (we are going to import model from timm)
  img_size = 300                          # Resize all the images to given

  # going to be used for loading dataset
  train_path= config.train_valid_splited_data_path+'/train'
  validate_path= config.train_valid_splited_data_path+'/validation'
#   test_path='./Product_Shapes/test'
def EfficientNetModeol():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("On which device we are on:{}".format(device))

    train_transform = T.Compose([
                             
                             T.Resize(size=(CFG.img_size,CFG.img_size)), # Resizing the image  
                             T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                             T.RandomAdjustSharpness(sharpness_factor=2),
                             T.RandomAutocontrast(),
                             T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels

    ])

    validate_transform = T.Compose([
                             
                             T.Resize(size=(CFG.img_size,CFG.img_size)), # Resizing the image 
                             T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels

    ])

    test_transform = T.Compose([
                             
                             T.Resize(size=(CFG.img_size,CFG.img_size)), # Resizing the image 
                             T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels

    ])

    # set paths
    trainset=datasets.ImageFolder(CFG.train_path,transform=train_transform)
    print("Trainset Size:  {}".format(len(trainset)))

    validateset=datasets.ImageFolder(CFG.validate_path,transform=validate_transform)
    print("validateset Size:  {}".format(len(validateset)))

# testset=datasets.ImageFolder(CFG.test_path,transform=test_transform)
# print("testset Size:  {}".format(len(testset)))

    trainloader = DataLoader(trainset,batch_size=CFG.batch_size,shuffle=True)
    print("No. of batches in trainloader:{}".format(len(trainloader)))
    print("No. of Total examples:{}".format(len(trainloader.dataset)))
# 
    validationloader = DataLoader(validateset,batch_size=CFG.batch_size,shuffle=True)
    print("No. of batches in validationloader:{}".format(len(validationloader)))
    print("No. of Total examples:{}".format(len(validationloader.dataset)))
# 
# testloader = DataLoader(testset,batch_size=CFG.batch_size,shuffle=True)
# print("No. of batches in testloader:{}".format(len(testloader)))
# print("No. of Total examples:{}".format(len(testloader.dataset)))

    model = timm.create_model(CFG.model_name,pretrained=True) 

    print(model)
    

    for param in model.parameters():
        param.requires_grad=False

    NUM_OF_CLASSES = len(os.listdir(CFG.train_path))
# 
    model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features=1536, out_features=1024,bias=True),
    nn.ReLU(), #ReLu to be the activation function
    nn.Dropout(p=0.3),
    nn.Linear(in_features=1024, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=NUM_OF_CLASSES,bias=True), 
    )

    model.classifier.requires_grad = True

    model.to(device) # move the model to GPU
    summary(model,input_size=(3,300,300))
    return device,model,trainloader,validationloader

class TrainingLoop():
    
    def __init__(self,criterion = None,optimizer = None,schedular = None):
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
    
    def train_batch_loop(self,model,trainloader):
        
        device,model,trainloader,validationloader = EfficientNetModeol()

        train_loss = 0.0
        train_acc = 0.0
        
        for images,labels in tqdm(trainloader): 
            
            # move the data to CPU
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = self.criterion(logits,labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_acc += accuracy(logits,labels)
            
        return train_loss / len(trainloader), train_acc / len(trainloader) 

    
    def valid_batch_loop(self,model,validloader):

        device,model,trainloader,validationloader = EfficientNetModeol()
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for images,labels in tqdm(validloader):
            
            # move the data to CPU
            images = images.to(device) 
            labels = labels.to(device)
            
            logits = model(images)
            loss = self.criterion(logits,labels)
            
            valid_loss += loss.item()
            valid_acc += accuracy(logits,labels)
            
        return valid_loss / len(validloader), valid_acc / len(validloader)
            
        
    def fit(self,model,trainloader,validloader,epochs):
        
        valid_min_loss = np.Inf 
        
        for i in range(epochs):
            
            model.train() # this turn on dropout
            avg_train_loss, avg_train_acc = self.train_batch_loop(model,trainloader) ###
            
            model.eval()  # this turns off the dropout lapyer and batch norm
            avg_valid_loss, avg_valid_acc = self.valid_batch_loop(model,validloader) ###
            
            if avg_valid_loss <= valid_min_loss :
                print("Valid_loss decreased {} --> {}".format(valid_min_loss,avg_valid_loss))
                torch.save(model.state_dict(),'ShapeLayer.pt')
                valid_min_loss = avg_valid_loss

                
            print("Epoch : {} Train Loss : {:.6f} Train Acc : {:.6f}".format(i+1, avg_train_loss, avg_train_acc))
            print("Epoch : {} Valid Loss : {:.6f} Valid Acc : {:.6f}".format(i+1, avg_valid_loss, avg_valid_acc))

def main():

    TrainValDataSplit()

    device,model,trainloader,validationloader = EfficientNetModeol()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = CFG.lr)

    trainer = TrainingLoop(criterion,optimizer)
    trainer.fit(model,trainloader,validationloader,epochs = CFG.epochs)
    torch.save(model.state_dict(), config.model_save_path)

if __name__ == "__main__":
    main()