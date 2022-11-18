import os
from random import sample
import shutil

rootPath = "/home/orelit/Projects -Sathsara/Planigo/data/Canned_food"
allFolders = os.listdir(rootPath)

sectionPaths ='/home/orelit/Projects -Sathsara/Planigo/data/section4'
sectionFolders = os.listdir(sectionPaths)

irrelevantFolders = [x for x in allFolders  if x not in sectionFolders]

all_images = []
for folder in irrelevantFolders:
    path = rootPath+'/'+folder
    images = os.listdir(path)
    for image in images:
        image_path = path +'/'+image
        all_images.append(image_path)

print(len(all_images))
image_sample = sample(all_images,300)

for image in image_sample:
    shutil.copy(image, '/home/orelit/Projects -Sathsara/Planigo/data/irrelavnt_images_for_section_four')



