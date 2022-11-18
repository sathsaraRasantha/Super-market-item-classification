from PIL import *
import torch
import numpy as np
import os
from tqdm.notebook import tqdm
import torch 
import numpy as np 
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch import nn
import timm # PyTorch Image Models
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
# print result
from tabulate import tabulate
# data analysis
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

def get_classes(path):
    lines = []
    with open(path,'r',encoding = 'utf-8') as file:
        lines = file.read().strip().split("\n")
        strip_list = [item.strip() for item in lines]
        strip_list.sort()
    return strip_list

MODEL_FOLDER = "/home/orelit/Projects -Sathsara/Planigo/Models/Wine_models"
#[name for name in os.listdir(TEST_DATA_FOLDER) if os.path.isdir(os.path.join(TEST_DATA_FOLDER, name))]
class_names = {}
model_paths = []
model_paths_tmp = os.listdir(MODEL_FOLDER)
for path in model_paths_tmp:
    model_path = os.path.join(MODEL_FOLDER,path)
    if path.endswith(".pt"):
        model_paths.append(model_path)
        class_names[model_path] = get_classes(model_path.replace(".pt",".txt"))

#
model_paths.sort()
# 
IMAGE_PATH = ""
MODEL_NAME = "efficientnet_b3"
IMSIZE = 300
TEST_DATA_FOLDER = "/home/orelit/Projects -Sathsara/Planigo/data/test"
# pass out threshold
THRESHOLD = 0.90
# folderstructure
dir_to_dict = {}
classes_in_dir = os.listdir(TEST_DATA_FOLDER)
for cls in classes_in_dir:
    file_list = os.listdir(os.path.join(TEST_DATA_FOLDER,cls))
    dir_to_dict[cls] = file_list

def get_actual_class_by_file_name(file_name):
    for cls , files_ in dir_to_dict.items():
        for single_file in files_:
            if single_file == file_name:
                return cls

input_file_list = []
for root, dirs, files in os.walk(TEST_DATA_FOLDER, topdown=False):
   for name in files:
      input_file_list.append(os.path.join(root, name))

models = []
for model_path in model_paths:
  # check device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("On which device we are on:{}".format(device))
  # load model
  #load pretrained model
  model = timm.create_model(MODEL_NAME,pretrained=True) 
  classes = class_names[model_path]
  # 
  model.classifier = nn.Sequential(
      nn.Dropout(p=0.3),
      nn.Linear(in_features=1536, out_features=1024,bias=True),
      nn.ReLU(), #ReLu to be the activation function
      nn.Dropout(p=0.3),
      nn.Linear(in_features=1024, out_features=256),
      nn.ReLU(),
      nn.Linear(in_features=256, out_features=len(classes),bias=True), 
  )
  # freeze the pre-tarined model:
  for param in model.parameters():
    param.requires_grad=False
  # move the model to GPU
  model.to(device)
  # save and evaluation
  model.load_state_dict(torch.load(model_path))
  model.eval()
  m={}
  m["model"] = model
  m["classes"] = classes
  m["model_path"] = model_path
  models.append(m)

loader = transforms.Compose([transforms.Resize(size=(IMSIZE,IMSIZE)), transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
# 
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    return image.to(device).unsqueeze(0)

print(len(input_file_list))

results = []
#
for IMAGE_PATH in input_file_list:
    image_probs = []
    image_results = []
    for model_element in models:
        model = model_element["model"]
        class_names_ = model_element["classes"]
        # 
        image = image_loader(IMAGE_PATH)
        # array with probs
        ps = model(image)
        ps = F.softmax(ps,dim = 1)
        ps = ps.cpu().data.numpy().squeeze()
        # max prob
        max_ps = ps.max()
        itemindex = np.where(ps == max_ps)[0][0]
        class_pred = f"{class_names_[itemindex]}"
        # 
        image_probs.append(max_ps)
        image_results.append(class_pred)
    # check best result
    image_probs = np.array(image_probs)
    image_results = np.array(image_results)
    max_prob = image_probs.max()
    best_index = np.where(image_probs == max_prob)[0][0]
    best_model_path = models[best_index]["model_path"]
    # append to results
    row = [IMAGE_PATH.split("/")[-1],max_prob,best_model_path.split("/")[-1],image_results[best_index],get_actual_class_by_file_name(IMAGE_PATH.split("/")[-1])]
    results.append(row)


print(tabulate(results, headers=['File Name', 'Max probability','Model Name','Predicted Class','Actual Class']))

df = pd.DataFrame(results, columns=['File Name', 'Max probability','Model Name','Predicted Class','Actual Class'])
print(df.head())

y_true = df["Actual Class"]
y_pred = df["Predicted Class"]
def plot_cm(y_true, y_pred, figsize=(10,10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sn.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
plot_cm(y_true, y_pred)

print("Acc =>",((len(df.loc[df["Predicted Class"] == df["Actual Class"]])/len(input_file_list)))*100)

df_filtered = df
df_filtered.loc[df_filtered['Max probability'] < 0.85, ['Predicted Class']] = 'Not found'
print("Acc =>",((len(df_filtered.loc[df_filtered["Predicted Class"] == df_filtered["Actual Class"]])/len(df_filtered)))*100)