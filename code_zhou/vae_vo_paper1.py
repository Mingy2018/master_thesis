#!/usr/bin/env python
import os
import sys

sys.path.append('./lib')
sys.path.append( './utils')
import visdom
import numpy as np
import tensorflow as tf
import dataIO as d
import config, dataset
import binvox_rw as bin
import voxel as V
import logdirs as lg
from datetime import datetime

from tqdm import *
from utils import *

'''
Global Parameters
'''
n_epochs = 20001
batch_size = 8
g_lr = 0.0025  # 0.0025
d_lr = 0.00001  # 0.00001
ae_lr = 0.001
beta = 0.5
d_thresh = 0.8
z_size = 200  # 200
leak_value = 0.2
cube_len = 32
obj_ratio = 0.7
obj = 'chair'
alpha_1 = 5 
alpha_2 = 0.0001
reg_l2 = 0.001

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
train_sample_directory = './train_data/'+dt_string+'/sample/'
model_directory = './train_data/'+dt_string+'/models/'
generated_model_directory = './train_data/'+dt_string+'/generated_objects/'
dirs=[train_sample_directory, model_directory, generated_model_directory]

is_local = False
weights = {}

# config for cudnn error
ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)

def voxel_var_encoder(inputs, keep_prob=0.5, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]
    with tf.variable_scope("V-encoder", reuse=reuse):
        e_1 = tf.nn.conv3d(inputs, weights['weV1'], strides=[1,1,1,1,1], padding="VALID")
        e_1 = tf.contrib.layers.batch_norm(e_1, is_training=phase_train)
        e_1 = tf.nn.elu(e_1)

        e_2 = tf.nn.conv3d(e_1, weights['weV2'], strides=strides, padding="SAME")
        e_2 = tf.contrib.layers.batch_norm(e_2, is_training=phase_train)
        e_2 = tf.nn.elu(e_2)

        e_3 = tf.nn.conv3d(e_2, weights['weV3'], strides=[1,1,1,1,1], padding="VALID")
        e_3 = tf.contrib.layers.batch_norm(e_3, is_training=phase_train)
        e_3 = tf.nn.elu(e_3)

        e_4 = tf.nn.conv3d(e_3, weights['weV4'], strides=strides, padding="SAME")
        e_4 = tf.contrib.layers.batch_norm(e_4, is_training=phase_train)
        e_4 = tf.nn.elu(e_4)

        def fc(name, x, num_outputs, batch_norm=True, relu=True):
            x = tf.layers.dense(tf.reshape(x, [-1, reduce(lambda a, b: a * b, x.shape.as_list()[1:])]), num_outputs,name=name)
            if batch_norm: x = tf.layers.batch_normalization(x, training=phase_train)
            if relu:
                x = tf.nn.relu(x)
            else:
                x = tf.nn.elu(x)
            return x

        e_5 = fc('fc1', e_4, 323, True, False)
        e_6 = fc('fc2', e_5, 100, False,False)

        z_mu = tf.layers.dense(e_6, units= z_size)
        z_sig = 0.5 * tf.layers.dense(e_6, units=z_size)
        epsilon = tf.random_normal(tf.stack([tf.shape(e_6)[0], z_size]))
        z = z_mu + tf.multiply(epsilon,tf.exp(z_sig))

    return z, z_mu, z_sig


def generator(z, batch_size=batch_size, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]
    with tf.variable_scope("gen", reuse=reuse):

        def fc(name, x, num_outputs, batch_norm=True, relu=True):
            x = tf.layers.dense(tf.reshape(x, [-1, reduce(lambda a, b: a * b, x.shape.as_list()[1:])]), num_outputs,name=name)
            if batch_norm: x = tf.layers.batch_normalization(x, training=phase_train)
            if relu:
                x = tf.nn.relu(x)
            else:
                x = tf.nn.elu(x)
            return x

        fc1 = fc('gfc1',z, 343, True, False)
        fc2 = tf.reshape(fc1, [-1,7,7,7,1] )
        g_1 = tf.nn.conv3d_transpose(fc2, weights['wg1'], (batch_size, 7, 7, 7, 64), strides=[1, 1, 1, 1, 1],padding="SAME",)
        g_1 = tf.contrib.layers.batch_norm(g_1, is_training=phase_train)
        g_1 = tf.nn.elu(g_1)

        g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size, 15, 15, 15, 32), strides=strides, padding="VALID")
        g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase_train)
        g_2 = tf.nn.elu(g_2)

        g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size, 15, 15, 15, 16), strides=[1,1,1,1,1],padding="SAME")
        g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase_train)
        g_3 = tf.nn.elu(g_3)
        
        g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size, 32, 32, 32, 8), strides=strides, padding="VALID")
        g_4 = tf.contrib.layers.batch_norm(g_4, is_training=phase_train)
        g_4 = tf.nn.elu(g_4)

        g_5 = tf.nn.conv3d_transpose(g_4, weights['wg5'], (batch_size, 32, 32, 32, 1), strides=[1,1,1,1,1], padding="SAME")

    return g_5

def initialiseWeights():
    global weights
    xavier_init = tf.contrib.layers.xavier_initializer()
    #l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    weights['weV1'] = tf.get_variable("weV1", shape=[3, 3, 3, 1, 8], initializer=xavier_init)
    weights['weV2'] = tf.get_variable("weV2", shape=[3, 3, 3, 8, 16], initializer=xavier_init)
    weights['weV3'] = tf.get_variable("weV3", shape=[3, 3, 3, 16, 32], initializer=xavier_init)
    weights['weV4'] = tf.get_variable("weV4", shape=[3, 3, 3, 32, 64], initializer=xavier_init)

    weights['wg1'] = tf.get_variable("wg1", shape=[3, 3, 3, 64, 1], initializer=xavier_init)
    weights['wg2'] = tf.get_variable("wg2", shape=[3, 3, 3, 32, 64], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("wg3", shape=[3, 3, 3, 16, 32], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 8, 16], initializer=xavier_init)
    weights['wg5'] = tf.get_variable("wg5", shape=[3, 3, 3, 1, 8], initializer=xavier_init)

    return weights


def trainGAN(is_dummy=False, checkpoint=None, subcate=config.subcate):
    weights = initialiseWeights()

    batch_vox = tf.placeholder(shape= [batch_size,cube_len,cube_len,cube_len],dtype= tf.float32)
    vol_tensor_1 = tf.expand_dims(batch_vox, -1)

    #vol_tensor_1 = tf.placeholder(shape=[batch_size, cube_len, cube_len, cube_len, 1], dtype=tf.float32)
    #change the range of target to [-1, 2]
    pre_vol_tensor_1 = np.multiply(3, vol_tensor_1) -1

    vol_tensor_2 = tf.placeholder(shape=[batch_size, cube_len, cube_len, cube_len, 1], dtype=tf.float32)
    # change the range of target [-1, 2]
    pre_vol_tensor_2 = np.multiply(3, vol_tensor_2) -1

    z_vector_v1, z_mu_v1, z_sig_v1 = voxel_var_encoder(pre_vol_tensor_1, phase_train=True, reuse=False)
    #z_vector_v1 = tf.maximum(tf.minimum(z_vector_v1, 0.99), -0.99)

    generated_vol = generator(z_vector_v1, phase_train=True, reuse=False)
    generated_vol = tf.nn.sigmoid(generated_vol)
    #generated_vol = tf.maximum(tf.minimum(generated_vol, 0.99), 0.01)

    para_gen = [var for var in tf.trainable_variables() if any(x in var.name for x in ['gen', 'wg'])]

    # Reconstruction loss
    mse_recon_loss = tf.reduce_mean(tf.pow(vol_tensor_1 - generated_vol, 2))

    # KL-divergence
    kl_divergence_vol = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + 2.0 * z_sig_v1 - z_mu_v1 ** 2 - tf.exp(2.0 * z_sig_v1),1))

    # Weighted binary cross-entropy
    def weighted_binary_crossentropy( output, target):
        return -0.98 * target * tf.log(output) + 0.02 *(1.0-target) * tf.log(1.0-output)
    cross_recon_loss = tf.reduce_mean(weighted_binary_crossentropy(tf.clip_by_value(generated_vol,1e-7, 1.0 -1e-7),pre_vol_tensor_1))

    # l2 loss
    l2_loss = tf.reduce_mean(tf.add_n([tf.nn.l2_loss(v) for v in para_gen]))

    # Toal loss
    loss = cross_recon_loss + kl_divergence_vol + l2_loss * reg_l2

    # graph record
    summary_r_loss=tf.summary.scalar("scalar", mse_recon_loss)
    summary_kl_div=tf.summary.scalar("scalar", kl_divergence_vol)
    summary_loss=tf.summary.scalar("scalar",loss)

    z_vector_v2, _, _ = voxel_var_encoder(pre_vol_tensor_2, phase_train=False, reuse=True)
    #z_vector_v2 = tf.maximum(tf.minimum(z_vector_v2, 0.99), -0.99)
    generated_vol_test = generator(z_vector_v2, phase_train=False, reuse=True)
    generated_vol_test = tf.nn.sigmoid(generated_vol_test)

    para_ae = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wg', 'gen', 'we', 'V-encoder'])]

    optimizer_op_ae = tf.train.AdamOptimizer(learning_rate=ae_lr, beta1=beta).minimize(loss, var_list=para_ae)
    saver = tf.train.Saver()

    # online viewer
    # vis = visdom.Visdom()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if checkpoint is not None:
            saver.restore(sess, checkpoint)

        if is_dummy:
            volumes = np.random.randint(0, 2, (batch_size, cube_len, cube_len, cube_len))
            print 'Using Dummy Data'
        else:
            category = config.dataset_categories[subcate]
            big_dataset = dataset.create_vox_dataset(config.dataset_model_path + category,
                                                     config.batch_size,
                                                     config.dataset_scale,
                                                     True)
            iterator = big_dataset.make_one_shot_iterator()
            batch_tensor = iterator.get_next()
            print("Create the training dataset successfully!")

        for epoch in range(n_epochs):

            batch = sess.run(batch_tensor)
            if batch[0].shape[0] is not batch_size:
                batch = sess.run(batch_tensor)

            #model_matrix = dataset.modelpath2matrix(batch[0])
            model_matrix = sess.run(vol_tensor_1, feed_dict={batch_vox:dataset.modelpath2matrix(batch[0])})

            #print "One example of model before change", model_matrix[0][2][2][2]
            # change the range of target
            #model_matrix = tf.expand_dims(model_matrix, -1)
            #print "One example of model after change", model_matrix[0][2][2][2].eval()

            d_summary_merge = tf.summary.merge([summary_r_loss,
                                                summary_kl_div,
                                                summary_loss])

            # print out the loss value-----------------------------------
            _, loss_1= sess.run([optimizer_op_ae, loss],feed_dict={vol_tensor_1: model_matrix})
            #print 'VAE Training ', "epoch: ", epoch, ', reconstruction_loss:', reconstruction_loss_1, "Type:", type(reconstruction_loss_1)
            #print 'VAE Training ', "epoch: ", epoch, ', kl_loss:', kl_divergence_1,"Type:", type(kl_divergence_1)
            #print 'VAE Training ', "epoch: ", epoch, ', l2_loss:', l2_loss_1,"Type:", type(l2_loss_1)
            print 'VAE Training ', "epoch: ", epoch, ', Total_loss:', loss_1,"Type:", type(loss_1)

            # crate logging directories
            lg.create_log_dict(dirs)
            # output generated chairs
            if ((epoch % 100 == 0) and (epoch <= 3200)) or ((epoch % 200 == 0) and (epoch > 3200)):
                if batch[0].shape[0] is not batch_size:
                    batch = sess.run(batch_tensor)
                model_matrix3 = dataset.modelpath2matrix(batch[0])
                model_matrix3 = tf.expand_dims(model_matrix3, -1)
                g_objects = sess.run(generated_vol_test, feed_dict={vol_tensor_2: model_matrix3.eval()}) # batch * 32 * 32 * 32 * 1
                g_objects.dump(train_sample_directory + '/biasfree_' + str(epoch))
                id_ch = np.random.randint(0, batch_size, 4)

                for i in range(4):
                    print 'The max value of generated object', g_objects[id_ch[i]].max()
                    print 'The min value of generated object', g_objects[id_ch[i]].min()
                    if g_objects[id_ch[i]].max() > 0.5:
                        V.write_binvox_file(np.squeeze(g_objects[id_ch[i]] > 0.5), generated_model_directory + '_'.join(map(str, [epoch, batch[1][i]])) + '.binvox')
                        #plot while training
                        #d.plotVoxelVisdom(np.squeeze(g_objects[id_ch[i]] > 0.1), vis, '_'.join(map(str, [epoch, i])))

            if ((epoch % 100 == 0) and (epoch <= 6000)) or ((epoch % 1000 == 0) and (epoch > 6000)):
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, save_path=model_directory + '/biasfree_' + str(epoch) + '.cptk')


def testGAN(trained_model_path=None, n_batches=40):
    weights = initialiseWeights()

    z_vector = tf.placeholder(shape=[batch_size, z_size], dtype=tf.float32)
    # net_g_test = generator(z_vector, phase_train=True, reuse=True)
    net_g_test = generator(z_vector, phase_train=False, reuse=tf.AUTO_REUSE)

    vis = visdom.Visdom()

    sess = tf.Session()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, trained_model_path)

        # output generated chairs
        for i in range(n_batches):
            next_sigma = float(raw_input())
            z_sample = np.random.normal(0, next_sigma, size=[batch_size, z_size]).astype(np.float32)
            g_objects = sess.run(net_g_test, feed_dict={z_vector: z_sample})
            id_ch = np.random.randint(0, batch_size, 4)
            for i in range(4):
                print g_objects[id_ch[i]].max(), g_objects[id_ch[i]].min(), g_objects[id_ch[i]].shape
                if g_objects[id_ch[i]].max() > 0.5:
                    d.plotVoxelVisdom(np.squeeze(g_objects[id_ch[i]] > 0.5), vis, '_'.join(map(str, [i])))

if __name__ == '__main__':
    test = bool(int(sys.argv[1]))
    if test:
        path = sys.argv[2]
        testGAN(trained_model_path=path)
    else:
        ckpt = sys.argv[2]
        if ckpt == '0':
            trainGAN(is_dummy=False, checkpoint=None)
        else:
            trainGAN(is_dummy=False, checkpoint=ckpt)
