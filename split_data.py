import os
import shutil

old_data_root = '/home/orelit/Projects -Sathsara/Planigo/data/WINE_SIMILAR_PRODUCTS'

similar_products = os.listdir(old_data_root)
print(similar_products)

for item in similar_products:

    folders = os.listdir(old_data_root+'/'+item)
    print(folders)

    for folder in folders:

        folder_path = '/home/orelit/Projects -Sathsara/Planigo/data/Wine (Polygon)/{}'.format(folder)
        destination = '/home/orelit/Projects -Sathsara/Planigo/data/WINE_POLYGON_VIT/{}/{}'.format(item,folder)
        try:
            shutil.copytree(folder_path, destination)
        except:
            pass
