import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import torch
import glob
import pytorch_lightning as pl
from huggingface_hub import HfApi, Repository
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchmetrics import Accuracy
from transformers import ViTFeatureExtractor, ViTForImageClassification
from pytorch_lightning.callbacks import ModelCheckpoint

data_dir = Path("/home/orelit/Projects -Sathsara/Planigo/data/CVAT Data/Wine")

ds=ImageFolder(data_dir)
indices = torch.randperm(len(ds)).tolist()
n_val = math.floor(len(indices) * .15)
train_ds = torch.utils.data.Subset(ds, indices[:-n_val])
val_ds = torch.utils.data.Subset(ds, indices[-n_val:])


label2id = {}
id2label = {}
for i, class_name in enumerate(ds.classes):
  label2id[class_name] = str(i)
  id2label[str(i)] = class_name

class ImageClassificationCollator:
   def __init__(self, feature_extractor): 
      self.feature_extractor = feature_extractor
   def __call__(self, batch):  
      encodings = self.feature_extractor([x[0] for x in batch],
      return_tensors='pt')   
      encodings['labels'] = torch.tensor([x[1] for x in batch],    
      dtype=torch.long)
      return encodings

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
collator = ImageClassificationCollator(feature_extractor)
train_loader = DataLoader(train_ds, batch_size=32, 
   collate_fn=collator, num_workers=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collator, 
   num_workers=2)
model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
         num_labels=len(label2id),
         label2id=label2id,
         id2label=id2label)

class Classifier(pl.LightningModule):
   def __init__(self, model, lr: float = 2e-5, **kwargs): 
       super().__init__()
       self.save_hyperparameters('lr', *list(kwargs))
       self.model = model
       self.forward = self.model.forward 
       self.val_acc = Accuracy()
   def training_step(self, batch, batch_idx):
       outputs = self(**batch)
       self.log(f"train_loss", outputs.loss)
       return outputs.loss
   def validation_step(self, batch, batch_idx):
       outputs = self(**batch)
       self.log(f"val_loss", outputs.loss)
       acc = self.val_acc(outputs.logits.argmax(1), batch['labels'])
       self.log(f"val_acc", acc, prog_bar=True)
       return outputs.loss
   def configure_optimizers(self):
       return torch.optim.Adam(self.parameters(), 
                        lr=self.hparams.lr,weight_decay = 0.00025)


pl.seed_everything(42)
classifier = Classifier(model, lr=2e-5)
trainer = pl.Trainer(gpus=1, precision=16, max_epochs=3)
trainer.fit(classifier, train_loader, val_loader)