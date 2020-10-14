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
n_epochs   = 20001
batch_size = 8
g_lr       = 0.0025 #0.0025
d_lr       = 0.00001 #0.00001
ae_lr	   = 0.001
beta       = 0.5
d_thresh   = 0.8
z_size     = 200 #200
leak_value = 0.2
cube_len   = 32
obj_ratio  = 0.7
obj        = 'chair' 

train_sample_directory = './train_sample/'
model_directory = './models/'
generated_model_directory='./generated_model/'

is_local = False

weights = {}

def encoderA(inputs, phase_train=True, reuse=False):

	strides    = [1,2,2,2,1]
	with tf.variable_scope("encoderA", reuse=reuse):
		e_1 = tf.nn.conv3d(inputs, weights['weA1'], strides=strides, padding="SAME")
		e_1 = tf.contrib.layers.batch_norm(e_1, is_training=phase_train)
		e_1 = lrelu(e_1, leak_value)

		e_2 = tf.nn.conv3d(e_1, weights['weA2'], strides=strides, padding="SAME")
		e_2 = tf.contrib.layers.batch_norm(e_2, is_training=phase_train)
		e_2 = lrelu(e_2, leak_value)

		e_3 = tf.nn.conv3d(e_2, weights['weA3'], strides=strides, padding="SAME")
		e_3 = tf.contrib.layers.batch_norm(e_3, is_training=phase_train)
		e_3 = lrelu(e_3, leak_value)

		e_4 = tf.nn.conv3d(e_3, weights['weA4'], strides=[1,1,1,1,1], padding="VALID")
		e_4 = tf.nn.tanh(e_4)

	return e_4


def generator(z, batch_size=batch_size, phase_train=True, reuse=False):

	strides    = [1,2,2,2,1]

	with tf.variable_scope("gen", reuse=reuse):
		z = tf.reshape(z, (batch_size, 1, 1, 1, z_size))
		g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size,4,4,4,512), strides=[1,1,1,1,1], padding="VALID")
		g_1 = tf.contrib.layers.batch_norm(g_1, is_training=phase_train)
		g_1 = tf.nn.relu(g_1)

		g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size,8,8,8,256), strides=strides, padding="SAME")
		g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase_train)
		g_2 = tf.nn.relu(g_2)

		g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size,16,16,16,128), strides=strides, padding="SAME")
		g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase_train)
		g_3 = tf.nn.relu(g_3)

		g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size,32,32,32,1), strides=strides, padding="SAME")
		g_4 = tf.nn.tanh(g_4)

	return g_4

def initialiseWeights():

	global weights
	xavier_init = tf.contrib.layers.xavier_initializer()

	weights['weA1'] = tf.get_variable("weA1", shape=[4, 4, 4, 1, 128], initializer=xavier_init)
	weights['weA2'] = tf.get_variable("weA2", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
	weights['weA3'] = tf.get_variable("weA3", shape=[4, 4, 4, 256, 512], initializer=xavier_init)
	weights['weA4'] = tf.get_variable("weA4", shape=[4, 4, 4, 512, z_size], initializer=xavier_init)

	weights['wg1'] = tf.get_variable("wg1", shape=[4, 4, 4, 512, z_size], initializer=xavier_init)
	weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 256, 512], initializer=xavier_init)
	weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
	weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 1, 128], initializer=xavier_init)

	return weights


def trainGAN(is_dummy=False, checkpoint=None, subcate=config.subcate):

	weights =  initialiseWeights()

	#x_vector = tf.placeholder(shape=[batch_size,cube_len,cube_len,cube_len,1],dtype=tf.float32)
	x2_vector = tf.placeholder(shape=[batch_size,cube_len,cube_len,cube_len,1],dtype=tf.float32)
	x3_vector = tf.placeholder(shape=[batch_size,cube_len,cube_len,cube_len,1],dtype=tf.float32)

	x2z_vector = encoderA(x2_vector, phase_train=True, reuse=False)
	x2z_vector = tf.maximum(tf.minimum(x2z_vector, 0.99), -0.99)
	#x2z_vector = tf.maximum(tf.minimum(x2z_vector, 0.99), 0.01)
	# Guassian loss between xz_vector (e4) and Z_vector?
	net_g_train = generator(x2z_vector, phase_train=True, reuse=False)
	net_g_train = tf.maximum(tf.minimum(net_g_train, 0.99), 0.01)

	# Reconstruction loss
	#recon_loss = -tf.reduce_mean(x2_vector * tf.log(net_g_train) + (1 - x2_vector) * tf.log(1 - net_g_train))
	recon_loss = tf.reduce_mean(tf.pow(x2_vector - net_g_train, 2))

	x3z_vector = encoderA(x3_vector, phase_train=False, reuse=True)
	x3z_vector = tf.maximum(tf.minimum(x3z_vector, 0.99), -0.99)
	net_g_test = generator(x3z_vector, phase_train=False, reuse=True)

	para_ae = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wg', 'gen', 'we', 'encoder'])]

	optimizer_op_ae = tf.train.AdamOptimizer(learning_rate=ae_lr,beta1=beta).minimize(recon_loss,var_list=para_ae)
	saver = tf.train.Saver()
	vis = visdom.Visdom()


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if checkpoint is not None:
			saver.restore(sess, checkpoint)

		if is_dummy:
			volumes = np.random.randint(0,2,(batch_size,cube_len,cube_len,cube_len))
			print 'Using Dummy Data'
		else:
			# volumes = d.getAll(obj=obj, train=True, is_local=is_local, obj_ratio=obj_ratio, cube_len=32)
			# print 'Using ' + obj + ' Data'
		# volumes = volumes[...,np.newaxis].astype(np.float)
			category = config.dataset_categories[subcate]
			big_dataset = dataset.create_subcate_dataset(config.dataset_img_path + category,
														 config.dataset_model_path + category,
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
				# img_matrix = dataset.img2matrix(batch[0], config.sequence_length)
				model_matrix = dataset.modelpath2matrix(batch[1])
				model_matrix = tf.expand_dims(model_matrix, -1)
				print 'Dimension of model_matrix:', model_matrix.shape

				_, reconstruction_loss = sess.run([optimizer_op_ae, recon_loss],feed_dict={x2_vector:model_matrix.eval()})
				print 'AE Training ', "epoch: ", epoch,', reconstruction_loss:', reconstruction_loss

				# output generated chairs
				if ((epoch % 100 == 0)and(epoch<=3200)) or ((epoch % 200 == 0)and(epoch>3200)):
					#idx3 = np.random.randint(len(volumes), size=batch_size)
					#x3 = volumes[idx3]
					if batch[0].shape[0] is not batch_size:
						batch = sess.run(batch_tensor)
					model_matrix3 = dataset.modelpath2matrix(batch[1])
					model_matrix3 = tf.expand_dims(model_matrix3, -1)
					g_objects=sess.run(net_g_test, feed_dict={x3_vector: model_matrix3.eval()}) # batch * 32 * 32 * 32 * 1

					if not os.path.exists(train_sample_directory):
						os.makedirs(train_sample_directory)
					if not os.path.exists(generated_model_directory):
						os.makedirs(generated_model_directory)
					g_objects.dump(train_sample_directory+'/biasfree_'+str(epoch))
					id_ch = np.random.randint(0, batch_size, 4) # numpy.random.randint(low,high=None,size=None,dtype) -> (4,) dimension
					# print 'The shape of generated objects',g_objects.shape
					# print 'id_ch is:' , id_ch
					# print 'id_ch shape is:', id_ch.shape

					for i in range(4):
						if g_objects[id_ch[i]].max() > 0.5:
							# save generated models into binvox files
							# print 'test model batch' , batch[2]
							# print 'one element', batch[2][i]
							V.write_binvox_file(np.squeeze(g_objects[id_ch[i]]>0.5),generated_model_directory+ '_'.join(map(str,[epoch,batch[2][i]]))+'.binvox')
							d.plotVoxelVisdom(np.squeeze(g_objects[id_ch[i]]>0.1), vis, '_'.join(map(str,[epoch,i])))

				if ((epoch % 100 == 0)and(epoch<=6000)) or ((epoch % 1000 == 0)and(epoch>6000)):
					if not os.path.exists(model_directory):
						os.makedirs(model_directory)
					saver.save(sess, save_path = model_directory + '/biasfree_' + str(epoch) + '.cptk')


def testGAN(trained_model_path=None, n_batches=40):

	weights = initialiseWeights()

	z_vector = tf.placeholder(shape=[batch_size,z_size],dtype=tf.float32)
	#net_g_test = generator(z_vector, phase_train=True, reuse=True)
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
			g_objects = sess.run(net_g_test,feed_dict={z_vector:z_sample})
			id_ch = np.random.randint(0, batch_size, 4)
			for i in range(4):
				print g_objects[id_ch[i]].max(), g_objects[id_ch[i]].min(), g_objects[id_ch[i]].shape
				if g_objects[id_ch[i]].max() > 0.5:
					d.plotVoxelVisdom(np.squeeze(g_objects[id_ch[i]]>0.5), vis, '_'.join(map(str,[i])))

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

