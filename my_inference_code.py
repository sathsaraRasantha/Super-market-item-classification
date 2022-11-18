import math
import os
from torch.autograd import Variable
import timm
import torch
from torchvision import datasets, models, transforms
import numpy as np
from torchvision import transforms as T,datasets
import torch.nn.functional as F
from torch import nn
import pandas as pd
from PIL import Image

dir_to_dict = {}

def get_actual_class_by_file_name(file_name):
    for cls , files_ in dir_to_dict.items():
        for single_file in files_:
            if single_file == file_name:
                return cls

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



root_model_path = '/home/orelit/Projects -Sathsara/Planigo/Models/final_models'

class PlanigoProductClassificationInference:

    def __init__(self, device=None):

        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def inference(self, model, image,CLASSES):

        model = model.to(self.device).float()
        # data = data.to(self.device).float()
        model.eval()  # this turns off the dropout layer and batch norm

        class_names = CLASSES
        # 
        # image = image_loader(image_path)
        # array with probs
        ps = model(image)
        ps = F.softmax(ps,dim = 1)
        ps = ps.cpu().data.numpy().squeeze()
        # max prob
        max_ps = ps.max()
        itemindex = np.where(ps == max_ps)[0][0]
        class_pred = f"{class_names[itemindex]}"
        # 
        # image_probs.append(max_ps)
        # image_results.append(class_pred)

        return (class_pred ,max_ps)

model_paths = sorted(os.listdir(root_model_path ))
absolute_model_paths = []
for path in model_paths:
    absolute_path = root_model_path + '/'+path
    absolute_model_paths.append(absolute_path)

sections_data = pd.read_csv('/home/orelit/Projects -Sathsara/Planigo/Models/canned_food_sections.csv')

# sections = sections_data.columns
sections = [50,50,50,48,50,50,47,46]

model_details = []

for i in range(8):
    model = absolute_model_paths[i]
    class_df = pd.read_csv('/home/orelit/Projects -Sathsara/Planigo/Models/canned_food_sections.csv')
    class_name_column = class_df['Section_{}'.format(i+1)]
    class_names = [x for x in class_name_column if math.isnan(x) == False]
    num_classes = sections[i]
    model_details.append((model,num_classes,class_names))


test_path = "/home/orelit/Projects -Sathsara/Planigo/data/test"

test_data = []
for root, dirs, files in os.walk(test_path, topdown=False):
   for name in files:
      test_data.append(os.path.join(root, name))

loader = transforms.Compose([transforms.Resize(size=(300,300)), transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
# 
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    return image.to(device).unsqueeze(0)

max_prob_image = []
for image in test_data:
    image_probs = []
    image_results = []
    for model,num_of_classes ,class_names in model_details:
        print(model)
        model_name = 'efficientnet_b3' 

        Canned_food_model = timm.create_model(model_name) 

        NUM_OF_CLASSES = num_of_classes

        Canned_food_model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1536, out_features=1024,bias=True),
        nn.ReLU(), #ReLu to be the activation function
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1024, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=NUM_OF_CLASSES,bias=True), 
        )

        Canned_food_model.load_state_dict(torch.load(model), strict=False)
        Canned_food_model.eval()

    # test_transform = T.Compose([
    #                          T.Resize(size=(300,300)), # Resizing the image 
    #                          T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
    #                          T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels

    # ])
        input_image = image_loader(image)
        inference = PlanigoProductClassificationInference()
        max_ps,class_pred = inference.inference(Canned_food_model, input_image,class_names)
        image_probs.append(max_ps)
        image_results.append(class_pred)

        print('done')
    
    image_probs = np.array(image_probs)
    image_results = np.array(image_results)
    print(image_results)
    max_prob = image_results.max()
    best_index = np.where(image_results == max_prob)[0][0]
    best_model_path = models[best_index]["model_path"]
    # append to results
    row = [image.split("/")[-1],max_prob,best_model_path.split("/")[-1],image_results[best_index],get_actual_class_by_file_name(IMAGE_PATH.split("/")[-1])]
    max_prob_image.append(row)





