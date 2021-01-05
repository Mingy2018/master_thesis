import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K

from VAE import *
#from beta_VAE import *
from utils import npytar, binvox_IO
import glob

learning_rate_1 = 0.0001
learning_rate_2 = 0.005
momentum = 0.9
batch_size = 10
epoch_num = 10

ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)

def data_loader(fname):
    reader = npytar.NpyTarReader(fname)
    xc = np.zeros((reader.length(), ) + input_shape, dtype = np.float32)
    print('The shape of loaded data', xc.shape)
    reader.reopen()
    for ix, (x, name) in enumerate(reader):
        print('The shape of x', x.shape)
        xc[ix] = x.astype(np.float32)
    print('The shape of loaded data', xc.shape)
    return 3.0 * xc - 1.0

def weighted_binary_crossentropy(target, output):
    loss = -(98.0 * target * K.log(output) + 2.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
    return loss

def learning_rate_scheduler(epoch, lr):
    if epoch >= 1:
        lr = learning_rate_2
    return lr

if __name__ == '__main__':

    model = get_model()
    inputs = model['inputs']
    outputs = model['outputs']
    mu = model['mu']
    sigma = model['sigma']
    z = model['z']

    encoder = model['encoder']
    decoder = model['decoder']

    plot_model(encoder, to_file = 'vae_encoder.pdf', show_shapes = True)
    plot_model(decoder, to_file = 'vae_decoder.pdf', show_shapes = True)

    vae = model['vae']


    # Loss functions

    # kl-divergence
    kl_div = -0.5 * K.mean(1 + sigma - K.square(mu) - K.exp(sigma))

    # Loss function in Genrative ... paper: a specialized form of Binary Cross-Entropy (BCE)
    BCE_loss = K.cast(K.mean(weighted_binary_crossentropy(inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7))), 'float32')

    # loss in beta-vae paper

    # gamma = 1000
    # max_capacity = 50
    # latent_loss = K.mean(kl_div)
    # latent_loss = gamma * K.abs(latent_loss - max_capacity)
    # latent_loss = K.reshape(latent_loss, [1,1])

    # Total loss
    loss = kl_div + BCE_loss #+ latent_loss

    vae.add_loss(loss)
    sgd = SGD(lr = learning_rate_1, momentum = momentum, nesterov = True)
    vae.compile(optimizer = sgd, metrics = ['accuracy'])

    plot_model(vae, to_file = 'vae.pdf', show_shapes = True)

    data_train = binvox_IO.voxelpath2matrix('./dataset/03001627_train')

    vae.fit(
        data_train,
        epochs = epoch_num,
        batch_size = batch_size,
        validation_data = (data_train, None),
        callbacks = [LearningRateScheduler(learning_rate_scheduler)]
    )

    vae.save_weights('./weights.h5')
