#
# classes
import os
import timm
import torch
from torchvision import datasets, models, transforms
import numpy as np
from torchvision import transforms as T,datasets
import torch.nn.functional as F
from torch import nn

model_name = 'efficientnet_b3'  

model = timm.create_model(model_name) 


NUM_OF_CLASSES = 6

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features=1536, out_features=1024,bias=True),
    nn.ReLU(), #ReLu to be the activation function
    nn.Dropout(p=0.3),
    nn.Linear(in_features=1024, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=NUM_OF_CLASSES,bias=True), 
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

top_level_dir = [ name for name in os.listdir("/home/orelit/Projects -Sathsara/Planigo/data/Product Shapes") if os.path.isdir(os.path.join("/home/orelit/Projects -Sathsara/Planigo/data/Product Shapes", name)) ]
top_level_dir.sort()
top_level_dir
# load model
model.load_state_dict(torch.load('/home/orelit/Projects -Sathsara/Planigo/Models/shape_identifier.pt'), strict=False)
model.eval()
# load data
PATH = "/home/orelit/Projects -Sathsara/Planigo/data/test"

test_transform = T.Compose([
                             T.Resize(size=(300,300)), # Resizing the image 
                             T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels

])

testset_=datasets.ImageFolder(PATH,transform=test_transform)
print("test set Size:  {}".format(len(testset_)))

#  read images
#
def model_evaluation(ps,label,path):
    status = None
    #
    class_name = top_level_dir
    label = f"{class_name[3]}"
    classes = np.array(class_name)
    ps = ps.cpu().data.numpy().squeeze()
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    ps = np.array(ps)
    max_ps = ps.max()
    print(max_ps)
    itemindex = np.where(ps == max_ps)[0][0]
    class_pred = f"{class_name[itemindex]}"
    # eval
    if class_pred.lower() == label.lower():
        status = True
    else:
        status = False
    #
    print(f"{path}->{label} ------ {class_pred}")
    return status

positive = 0
negative = 0
#
print("file name -> real -------- pred")
for idx, (sample, target) in enumerate(testset_):
    # sample = image
    # target given class
    fname,_ = testset_.imgs[idx]
    fname = fname.split("/")[-1]
    image = sample
    label = target

    ps = model(image.unsqueeze(0))
    ps = F.softmax(ps,dim = 1)
    if(model_evaluation(ps,label,fname)==True):
        positive+=1
    else:
        negative+=1

#
print("*~~~~~~~~~~~~~~~~~~~~~*")
print("positive",positive)
print("negative",negative)
