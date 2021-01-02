import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import shutil
import sys

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
    test_result_path = sys.argv[1]
    save_the_img = bool(int(sys.argv[2]))

    model = get_model()

    inputs = model['inputs']
    outputs = model['outputs']
    mu = model['mu']
    sigma = model['sigma']
    z = model['z']

    encoder = model['encoder']
    decoder = model['decoder']
    vae = model['vae']

    # Set the weight files and test dataset path
    vae.load_weights('/home/zmy/TrainingData/tf1.x.keras/vae_binvox_train_kl.h5')
    data_test, hash = binvox_IO.voxelpath2matrix('./dataset/03001627_test_1')

    reconstructions = vae.predict(data_test)
    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    # save the original test dataset file
    for i in range(reconstructions.shape[0]):
        shutil.copy2('/home/zmy/Datasets/03001627_test_1/'+hash[i]+'.binvox', test_result_path)
        data_test[data_test > 0] = 1
        data_test[data_test < 0] = 0
        if save_the_img:
            save_volume.save_binvox_output(data_test[i, 0, :], hash[i], test_result_path, '', save_img= save_the_img)

    # save the generated objects files
    for i in range(reconstructions.shape[0]):
        save_volume.save_binvox_output(reconstructions[i, 0, :], hash[i], test_result_path, '_gen', save_bin= True, save_img= save_the_img)
