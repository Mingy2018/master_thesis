import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import shutil

from VAE import *
from utils import npytar, save_volume
from utils import binvox_IO

ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)

def data_loader(fname):
    reader = npytar.NpyTarReader(fname)
    xc = np.zeros((reader.length(), ) + input_shape, dtype = np.float32)
    reader.reopen()
    for ix, (x, name) in enumerate(reader):
        xc[ix] = x.astype(np.float32)
    return 3.0 * xc - 1.0

if __name__ == '__main__':
    model = get_model()

    inputs = model['inputs']
    outputs = model['outputs']
    mu = model['mu']
    sigma = model['sigma']
    z = model['z']

    encoder = model['encoder']
    decoder = model['decoder']
    vae = model['vae']

    vae.load_weights('vae_binvox.h5')

    #data_test = data_loader('datasets/shapenet10_chairs_nr.tar')
    data_test, hash = binvox_IO.voxelpath2matrix('/home/zmy/Datasets/03001627_test')

    reconstructions = vae.predict(data_test)
    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists('reconstructions'):
        os.makedirs('reconstructions')

    if True:
        for i in range(reconstructions.shape[0]):
            shutil.copy2('/home/zmy/Datasets/03001627_test/'+hash[i]+'.binvox', './reconstructions')
            data_test[data_test > 0] = 1
            data_test[data_test < 0] = 0
            save_volume.save_binvox_output(data_test[i, 0, :], hash[i], 'reconstructions', '')

    for i in range(reconstructions.shape[0]):
        save_volume.save_binvox_output(reconstructions[i, 0, :], hash[i], 'reconstructions', '_gen', save_bin= True)
