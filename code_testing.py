from collections import Counter
import re
import os
import shutil
import numpy as np

# with open("/home/orelit/Projects -Sathsara/Planigo/data/softdrinksfirst100_akalanka.txt") as file_in:
#     lines = []
#     for line in file_in:
#         line = re.sub('\n','',line)
#         lines.append(line)

# print(lines)

# rootPath = "/home/orelit/Projects -Sathsara/Planigo/data/CVAT Data/Soft drink"
# allFolders = os.listdir(rootPath)

# irrelevantFolders = [x for x in allFolders  if x not in lines]


# for folder in irrelevantFolders:
#     path = rootPath+'/'+folder
#     shutil.copytree(path, '/home/orelit/Projects -Sathsara/Planigo/data/soft_drinks_sath/{}'.format(folder))
    

  
# rootPath = "/home/orelit/Projects -Sathsara/Planigo/data/CVAT Data/Wine"
# allFolders = os.listdir(rootPath)
# print(len(allFolders ))

# wine_sub_path = '/home/orelit/Projects -Sathsara/Planigo/data/WINE'
# wine_sub_folders = os.listdir(wine_sub_path)


# folder_list = []
# for item in wine_sub_folders:
#     folders = os.listdir(wine_sub_path+'/'+item)
#     for folder in folders:
#         folder_list.append(folder)


# print(len(folder_list))

# WC = Counter(folder_list)

# for letter, count in WC.items():
#         if (count > 1):
#             print(letter)


# irrelevantFolders = [x for x in allFolders  if x not in folder_list]
# print(len(irrelevantFolders))

# for folder in irrelevantFolders:
#     path = rootPath+'/'+folder
#     shutil.copytree(path, '/home/orelit/Projects -Sathsara/Planigo/data/WINE/other/{}'.format(folder))

# new_data_root = '/home/orelit/Projects -Sathsara/Planigo/data/WINE_POLYGON_VIT'

# similar_products = os.listdir(new_data_root)
# print(similar_products)

# new_folder_list = []

# for item in similar_products:

#     folders = os.listdir(new_data_root+'/'+item)
#     print(folders)

#     for folder in folders:

#         new_folder_list.append(folder)


# old_data_root = '/home/orelit/Projects -Sathsara/Planigo/data/Wine (Polygon)'

# old_folder_list = os.listdir(old_data_root)

# MissingFolders = [x for x in old_folder_list  if x not in new_folder_list]
# print(len(MissingFolders))
# print(MissingFolders)

# for folder in MissingFolders:

#     folder_path = '/home/orelit/Projects -Sathsara/Planigo/data/Wine (Polygon)/{}'.format(folder)
#     destination = '/home/orelit/Projects -Sathsara/Planigo/data/WINE_POLYGON_VIT/Missing/{}'.format(folder)
#     try:
#         shutil.copytree(folder_path, destination)
#     except:
#         pass

new_folder_list = os.listdir('/home/orelit/Projects -Sathsara/Planigo/data/WINE_POLYGON_VIT/white_and_blue')

def imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


for folder in new_folder_list:

    try:
        imagePaths = list(imlist('/home/orelit/Projects -Sathsara/Planigo/data/Wine/{}'.format(folder)))
        np.random.shuffle(imagePaths)
        testPathsLen = int(len(imagePaths) * 0.05)
        testPaths = imagePaths[:testPathsLen]

        for path in testPaths:
            imageName = path.split(os.path.sep)[-1]
            shutil.copyfile(path, '/home/orelit/Projects -Sathsara/Planigo/data/intermediate_data/{}'.format(imageName))

    except:
        pass
   




