import pandas as pd
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage

from build_dataset import imlist
import random

# image_paths = list(imlist('/home/orelit/Projects -Sathsara/Planigo/data/Product Shapes/Bottles'))

# train_img = []
# for image_path in tqdm(image_paths):
#     img = imread(image_path)
#     img = img/255
#     train_img.append(img)

# train_x = np.array(train_img)
# print(train_x.shape)


# final_train_data = []
# final_target_train = []
# for i in tqdm(range(train_x.shape[0])):
#     final_train_data.append(train_x[i])
#     final_train_data.append(rotate(train_x[i], angle=45, mode = 'wrap'))
#     final_train_data.append(np.fliplr(train_x[i]))
#     final_train_data.append(np.flipud(train_x[i]))
#     final_train_data.append(random_noise(train_x[i],var=0.2**2))
 

# len(final_target_train), len(final_train_data)
# final_train = np.array(final_train_data)


# print(len(final_train))


# fig,ax = plt.subplots(nrows=1,ncols=5,figsize=(20,20))
# for i in range(5):
#     ax[i].imshow(final_train[i+30])
#     ax[i].axis('off')

# plt.show()
# import random
# for image in final_train:
#     image_name = random.randint(0,10000)
#     imsave("/home/orelit/Projects -Sathsara/Planigo/data/new/{}.png".format(image_name), image)

#  '7290015350150'  '7290015951227'

items =[ '7290015951227' ]

for item in items:
    imagePaths = list(imlist('/home/orelit/Projects -Sathsara/Planigo/data/VIT_train_wine_low_variance_test/{}'.format(item)))
    train_img = []
    for image_path in tqdm(imagePaths):
        img = imread(image_path)
        img = img/255
        train_img.append(img)
    
    train_x = np.array(train_img)
    final_train_data = []
    for i in tqdm(range(train_x.shape[0])):
        final_train_data.append(train_x[i])
        final_train_data.append(rotate(train_x[i], angle=45, mode = 'wrap'))
        final_train_data.append(np.fliplr(train_x[i]))
        final_train_data.append(np.flipud(train_x[i]))
        final_train_data.append(random_noise(train_x[i],var=0.2**2))

    final_train = np.array(final_train_data)

    os.mkdir('/home/orelit/Projects -Sathsara/Planigo/data/VIT_train_wine_low_variance_test_augmented/{}'.format(item))

    for image in final_train:

        image_name = random.randint(0,10000)
        imsave("/home/orelit/Projects -Sathsara/Planigo/data/VIT_train_wine_low_variance_test_augmented/{}/{}.png".format(item,image_name), image)

    



   
    
    
    