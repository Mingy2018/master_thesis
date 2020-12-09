

import numpy as np
import scipy.ndimage as nd
from utils import binvox_rw
import glob

def read_voxel_data(model_path):
    with open(model_path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        return model.data

def voxelpath2matrix(voxel_path, padding = False):
    voxel_name = glob.glob(voxel_path+'/*')
    voxels = np.zeros((len(voxel_name),) + (1,32,32,32), dtype=np.float32)
    for i, name in enumerate(voxel_name):
        print('1',i)
        print('2',name)
        model = read_voxel_data(name)
        if padding:
            model = nd.zoom(model, (0.75, 0.75, 0.75), mode = 'constant', order = 0)
            model = np.pad(model, ((4,4),(4,4),(4,4)), 'constant')
        voxels[i] = model.astype(np.float32)
    return 3.0* voxels -1.0

def write_binvox_file(pred, filename):
    with open(filename, 'w') as f:
        voxel = binvox_rw.Voxels(pred, [32, 32, 32], [0, 0, 0], 1, 'xzy')
        binvox_rw.write(voxel, f)
        f.close()