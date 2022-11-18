import os
import shutil

data_root = '/home/orelit/Projects -Sathsara/Planigo/data/Canned_food'

canned_food_products = os.listdir(data_root)
print(canned_food_products)

import random

def chunk(xs, n):
    ys = list(xs)
    random.shuffle(ys)
    ylen = len(ys)
    size = int(ylen / n)
    chunks = [ys[0+size*i : size*(i+1)] for i in range(n)]
    leftover = ylen - size*n
    edge = size*n
    for i in range(leftover):
            chunks[i%n].append(ys[edge+i])
    return chunks

folders_list = chunk(canned_food_products , 9)


for i,folder in enumerate(folders_list):
    
    for item in folder:
        
        folder_path = '/home/orelit/Projects -Sathsara/Planigo/data/Canned_food/{}'.format(item)
        destination = '/home/orelit/Projects -Sathsara/Planigo/data/Canned_food_Splitted/folder{}/{}'.format(i,item)

        try:
            shutil.copytree(folder_path, destination)
        except:
            pass

    print('folder {} done'.format(i))
        


