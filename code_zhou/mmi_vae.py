#!/usr/bin/env python
import os
import sys

sys.path.append('./lib')
import visdom

import numpy as np
import tensorflow as tf
import dataIO as d
import config, dataset
import binvox_rw as bin
import voxel as V

from tqdm import *
from utils import *

'''
Global Parameters
'''
# n_epochs = 20001
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
n_views = 12


train_sample_directory = './train_sample/'
model_directory = './models/'
generated_model_directory = './generated_model/'

is_local = False
weights = {}

# config for cudnn error
ConFig=tf.ConfigProto()
ConFig.gpu_options.allow_growth=True
session=tf.Session(config=ConFig)


def voxel_var_encoder(inputs, keep_prob=0.5, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]
    with tf.variable_scope("V-encoder", reuse=reuse):
        e_1 = tf.nn.conv3d(inputs, weights['weV1'], strides=strides, padding="SAME")
        e_1 = tf.contrib.layers.batch_norm(e_1, is_training=phase_train)
        e_1 = lrelu(e_1, leak_value)

        e_2 = tf.nn.conv3d(e_1, weights['weV2'], strides=strides, padding="SAME")
        e_2 = tf.contrib.layers.batch_norm(e_2, is_training=phase_train)
        e_2 = lrelu(e_2, leak_value)

        e_3 = tf.nn.conv3d(e_2, weights['weV3'], strides=strides, padding="SAME")
        e_3 = tf.contrib.layers.batch_norm(e_3, is_training=phase_train)
        e_3 = lrelu(e_3, leak_value)

        e_4 = tf.nn.conv3d(e_3, weights['weV4'], strides=[1, 1, 1, 1, 1], padding="VALID")
        e_4 = tf.nn.sigmoid(e_4)

        # normal distribution and sampling
        x = tf.nn.dropout(e_4, keep_prob)
        x = tf.contrib.layers.flatten(x) # B * z_size
        z_mu = tf.layers.dense(x, units= z_size)
        z_sig = 0.5 * tf.layers.dense(x, units=z_size)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], z_size]))
        z = z_mu + tf.multiply(epsilon,tf.exp(z_sig))


    return z, z_mu, z_sig

def img_var_encoder(inputs, keep_prob=0.5, phase_train=True, reuse=False):
    """
    input: Batch x Viewns x Width x Height x Channels (tensor)
    """
    # transpose views : (BxVxWxHxC) -> (VxBxWxHxC)
    views = tf.transpose(inputs, perm=[1, 0, 2, 3, 4])

    view_pool = []

    for i in xrange(n_views):
        # set reuse True for i > 0, for weight-sharing
        reuse = reuse or (i != 0)
    with tf.variable_scope("I-encoder", reuse=reuse):
        view = tf.gather(views, i)  # BxWxHxC B*128*128*3

    conv1 = tf.nn.conv2d(view, weights['weI1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.contrib.layers.batch_norm(conv1, is_training=phase_train)
    conv1 = tf.nn.relu(conv1)  # B*128*128*64

    conv2_1 = tf.nn.conv2d(conv1, weights['weB2_1'], strides=[1, 2, 2, 1], padding='SAME')
    conv2_2 = tf.nn.conv2d(conv2_1, weights['weB2_2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_3 = tf.nn.conv2d(conv2_2, weights['weB2_3'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.concat([conv2_1, conv2_3], axis=-1)
    conv2 = tf.contrib.layers.batch_norm(conv2, is_training=phase_train)
    conv2 = tf.nn.relu(conv2)  # B*64*64*128

    conv3_1 = tf.nn.conv2d(conv2, weights['weB3_1'], strides=[1, 2, 2, 1], padding='SAME')
    conv3_2 = tf.nn.conv2d(conv3_1, weights['weB3_2'], strides=[1, 1, 1, 1], padding='SAME')
    conv3_3 = tf.nn.conv2d(conv3_2, weights['weB3_3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.concat([conv3_1, conv3_3], axis=-1)
    conv3 = tf.contrib.layers.batch_norm(conv3, is_training=phase_train)
    conv3 = tf.nn.relu(conv3)  # B*32*32*256

    conv4_1 = tf.nn.conv2d(conv3, weights['weB4_1'], strides=[1, 2, 2, 1], padding='SAME')
    conv4_2 = tf.nn.conv2d(conv4_1, weights['weB4_2'], strides=[1, 1, 1, 1], padding='SAME')
    conv4_3 = tf.nn.conv2d(conv4_2, weights['weB4_3'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.concat([conv4_1, conv4_3], axis=-1)
    conv4 = tf.contrib.layers.batch_norm(conv4, is_training=phase_train)
    conv4 = tf.nn.relu(conv4)  # B*16*16*512

    conv5_1 = tf.nn.conv2d(conv4, weights['weB5_1'], strides=[1, 2, 2, 1], padding='SAME')
    conv5_2 = tf.nn.conv2d(conv5_1, weights['weB5_2'], strides=[1, 1, 1, 1], padding='SAME')
    conv5_3 = tf.nn.conv2d(conv5_2, weights['weB5_3'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = tf.concat([conv5_1, conv5_3], axis=-1)
    conv5 = tf.contrib.layers.batch_norm(conv5, is_training=phase_train)
    conv5 = tf.nn.relu(conv5)  # B*8*8*1024

    pool6 = tf.nn.avg_pool(conv5, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')  # B*1*1*1024
    # dim = np.prod(pool5.get_shape().as_list()[1:])
    # reshape = tf.reshape(pool5, [-1, dim])
    view_pool.append(pool6)

    # view_pool = tf.reshape(view_pool, (batch_size, 1, 1, -1, n_views)) #B*1*1*1024*V
    view_pool = tf.stack(view_pool, 0)
    view_pool = tf.transpose(view_pool, perm=[1, 2, 3, 4, 0])  # B*1*1*1024*V
    print view_pool.shape
    pool6_vp = tf.reduce_max(view_pool, axis=-1)  # B*1*1*1024

with tf.variable_scope("encoderB", reuse=reuse):
    fc7 = tf.nn.conv2d(pool6_vp, weights['weB7'], strides=[1, 1, 1, 1], padding='SAME')  # B*1*1*512
    fc8 = tf.nn.conv2d(fc7, weights['weB8'], strides=[1, 1, 1, 1], padding='SAME')  # B*1*1*z_size
    fc8 = tf.nn.tanh(fc8)

return fc8


def generator(z, batch_size=batch_size, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]

    with tf.variable_scope("gen", reuse=reuse):
        z = tf.reshape(z, (batch_size, 1, 1, 1, z_size))
        g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size, 4, 4, 4, 512), strides=[1, 1, 1, 1, 1],
                                     padding="VALID")
        g_1 = tf.contrib.layers.batch_norm(g_1, is_training=phase_train)
        g_1 = tf.nn.relu(g_1)

        g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size, 8, 8, 8, 256), strides=strides, padding="SAME")
        g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase_train)
        g_2 = tf.nn.relu(g_2)

        g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size, 16, 16, 16, 128), strides=strides,
                                     padding="SAME")
        g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase_train)
        g_3 = tf.nn.relu(g_3)

        g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size, 32, 32, 32, 1), strides=strides, padding="SAME")
        g_4 = tf.nn.tanh(g_4)

    return g_4


def initialiseWeights():
    global weights
    xavier_init = tf.contrib.layers.xavier_initializer()

    weights['weV1'] = tf.get_variable("weV1", shape=[4, 4, 4, 1, 128], initializer=xavier_init)
    weights['weV2'] = tf.get_variable("weV2", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['weV3'] = tf.get_variable("weV3", shape=[4, 4, 4, 256, 512], initializer=xavier_init)
    weights['weV4'] = tf.get_variable("weV4", shape=[4, 4, 4, 512, z_size], initializer=xavier_init)

    weights['wg1'] = tf.get_variable("wg1", shape=[4, 4, 4, 512, z_size], initializer=xavier_init)
    weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 256, 512], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 1, 128], initializer=xavier_init)

    return weights


def trainGAN(is_dummy=False, checkpoint=None, subcate=config.subcate):
    weights = initialiseWeights()

    vol_tensor_1 = tf.placeholder(shape=[batch_size, cube_len, cube_len, cube_len, 1], dtype=tf.float32)
    vol_tensor_2 = tf.placeholder(shape=[batch_size, cube_len, cube_len, cube_len, 1], dtype=tf.float32)

    z_vector_v1, z_mu_v1, z_sig_v1 = voxel_var_encoder(vol_tensor_1, phase_train=True, reuse=False)
    z_vector_v1 = tf.maximum(tf.minimum(z_vector_v1, 0.99), -0.99)

    generated_vol = generator(z_vector_v1, phase_train=True, reuse=False)
    generated_vol = tf.maximum(tf.minimum(generated_vol, 0.99), 0.01)

    # Reconstruction loss
    # recon_loss = -tf.reduce_mean(vol_tensor_1 * tf.log(generated_vol) + (1 - vol_tensor_1) * tf.log(1 - generated_vol))
    recon_loss = tf.reduce_mean(tf.pow(vol_tensor_1 - generated_vol, 2))

    # KL-divergence
    kl_divergence_vol = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + 2.0 * z_sig_v1 - z_mu_v1 ** 2 - tf.exp(2.0 * z_sig_v1),1))

    # Toal loss
    loss =  alpha_2 * recon_loss + alpha_1 * kl_divergence_vol

    #
    summary_r_loss=tf.summary.scalar("scalar", recon_loss)
    summary_kl_div=tf.summary.scalar("scalar", kl_divergence_vol)
    summary_loss=tf.summary.scalar("scalar",loss)


    z_vector_v2, _, _ = voxel_var_encoder(vol_tensor_2, phase_train=False, reuse=True)
    z_vector_v2 = tf.maximum(tf.minimum(z_vector_v2, 0.99), -0.99)
    generated_vol_test = generator(z_vector_v2, phase_train=False, reuse=True)

    para_ae = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wg', 'gen', 'we', 'encoder'])]

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

        for epoch in range(config.epoch):
            batch = sess.run(batch_tensor)
            if batch[0].shape[0] is not batch_size:
                batch = sess.run(batch_tensor)
            model_matrix = dataset.modelpath2matrix(batch[0])
            model_matrix = tf.expand_dims(model_matrix, -1)

            d_summary_merge = tf.summary.merge([summary_r_loss,
                                                summary_kl_div,
                                                summary_loss])
            # print out the loss value-----------------------------------

            _, loss_1, reconstruction_loss_1, kl_divergence_1 = sess.run([optimizer_op_ae, loss, recon_loss, kl_divergence_vol],feed_dict={vol_tensor_1: model_matrix.eval()})
            print 'VAE Training ', "epoch: ", epoch, ', reconstruction_loss:', reconstruction_loss_1
            print 'VAE Training ', "epoch: ", epoch, ', kl_loss:', kl_divergence_1
            print 'VAE Training ', "epoch: ", epoch, ', Total_loss:', loss_1
            # ------------------------------------------------------------

            # output generated chairs
            if ((epoch % 100 == 0) and (epoch <= 3200)) or ((epoch % 200 == 0) and (epoch > 3200)):
                if batch[0].shape[0] is not batch_size:
                    batch = sess.run(batch_tensor)
                model_matrix3 = dataset.modelpath2matrix(batch[0])
                model_matrix3 = tf.expand_dims(model_matrix3, -1)
                g_objects = sess.run(generated_vol_test, feed_dict={vol_tensor_2: model_matrix3.eval()}) # batch * 32 * 32 * 32 * 1

                if not os.path.exists(train_sample_directory):
                    os.makedirs(train_sample_directory)
                if not os.path.exists(generated_model_directory):
                    os.makedirs(generated_model_directory)
                g_objects.dump(train_sample_directory + '/biasfree_' + str(epoch))
                id_ch = np.random.randint(0, batch_size, 4)

                for i in range(4):
                    if g_objects[id_ch[i]].max() > 0.5:
                        V.write_binvox_file(np.squeeze(g_objects[id_ch[i]] > 0.5), generated_model_directory + '_'.join(map(str, [epoch, batch[1][i]])) + '.binvox')
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
