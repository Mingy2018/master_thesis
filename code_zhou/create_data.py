
import sys
sys.path.append('./lib')
sys.path.append( './utils')
import os

import dataset
import os
import voxel
import scipy.ndimage as nd
import numpy as np
import shutil, os
import glob
import random

def get_dataname(model_path):

    category_model_paths = dataset.get_all_subcategory_paths(model_path)

    # Model paths
    all_models = []
    all_ID = [f for f in os.listdir(model_path)]
    for subcategory_model_path in category_model_paths:
        subcategory_model = dataset.get_subcategory_model(subcategory_model_path)
        all_models.append(subcategory_model)
    #print 'The total number of data element is', len(all_models)

    return all_models, all_ID

def voxel_compress_padding(binvox_path, binvox_ID, aim_path, padding = True):
    gt = []
    for paths in binvox_path:
        model = voxel.read_voxel_data(paths[0])
        if padding:
            model = nd.zoom(model, (0.75, 0.75, 0.75), mode = 'constant', order = 0)
            model = np.pad(model, ((4,4),(4,4),(4,4)), 'constant')
        gt.append(model)
    gt = np.array(gt)

    for i in range(len(gt)):
        voxel.write_binvox_file(np.squeeze(gt[i]),aim_path + binvox_ID[i] + '.binvox')

def get_full_name(path):
    return glob.glob(path)

def main():

    # Part for compressing data
    # binvox, ID = get_dataname('/home/zmy/Datasets/03001627')
    # os.makedirs('/home/zmy/Datasets/03001627_ori')
    # voxel_compress_padding(binvox, ID,'/home/zmy/Datasets/03001627_ori/', False)

    # Part generate small dataset 800 ele
    binvox = os.listdir('/home/zmy/Datasets/03001627_test')
    id_ch = random.sample(range(0,999), 100)
    os.makedirs('/home/zmy/Datasets/03001627_test_1')
    for i in range(100):
        shutil.copy2('/home/zmy/Datasets/03001627_test/'+binvox[id_ch[i]], '/home/zmy/Datasets/03001627_test_1')
    # test_file = os.listdir('/home/zmy/Datasets/03001627_test')
    # for file in os.listdir('/home/zmy/Datasets/03001627_train'):
    #     if file in test_file:
    #         os.remove(os.path.join('/home/zmy/Datasets/03001627_train', file))

if __name__ == '__main__':
    main()





