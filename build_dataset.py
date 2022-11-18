import numpy as np
import shutil
import os

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


path = "/home/orelit/Projects -Sathsara/Planigo/data/Canned food"

items =os.listdir(path)

for item in items:
	imagePaths = list(imlist('/home/orelit/Projects -Sathsara/Planigo/data/Canned food/{}'.format(item)))
	np.random.shuffle(imagePaths)
	valPathsLen = int(len(imagePaths) * 0.2)

	trainPathsLen = len(imagePaths) - valPathsLen

	trainPaths = imagePaths[:trainPathsLen]
	valPaths = imagePaths[trainPathsLen:]

	copy_images(trainPaths, '/home/orelit/Projects -Sathsara/Planigo/data/Canned_food_training/train')
	copy_images(valPaths, '/home/orelit/Projects -Sathsara/Planigo/data/Canned_food_training/validation')

	


