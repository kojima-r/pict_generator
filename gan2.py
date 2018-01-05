#!/usr/bin/env python
"""Wasserstein generative adversarial network for MNIST (Arjovsky et
al., 2017). It modifies GANs (Goodfellow et al., 2014) to optimize
under the Wasserstein distance.
References
----------
http://edwardlib.org/tutorials/gan
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import tensorflow as tf

from edward.models import Uniform,Normal
from observations import mnist
#from tensorflow.contrib import slim
from keras import backend as K
from keras.layers import *
from edward.util import Progbar

nch=3
M =50	# batch size during training
D=200
nepoch=500
K.set_learning_phase(0)  

#K.in_training_phase()
def generator(array, batch_size):
	"""Generate batch with respect to array's first axis."""
	start = 0	# pointer to where we are in iteration
	while True:
		stop = start + batch_size
		diff = stop - array.shape[0]
		if diff <= 0:
			batch = array[start:stop]
			start += batch_size
		else:
			batch = np.concatenate((array[start:], array[:diff]))
			start = diff
		batch = batch.astype(np.float32) / 255.0	# normalize pixel intensities
		batch = np.random.binomial(1, batch)	# binarize images
		yield batch.reshape((batch_size,-1))


def generative_network(eps):
	hidden = Dense(4*4*256, activation=None)(eps)
	hidden=Reshape([4,4,256])(hidden)
	act=None
	seq=[
		normalization.BatchNormalization(),
		convolutional.Conv2DTranspose(128,(4,4),strides=(2, 2), padding='same',activation=act),
		normalization.BatchNormalization(),
		advanced_activations.LeakyReLU(alpha=0.3),
		convolutional.Conv2DTranspose(64,(4,4),strides=(2, 2), padding='same',activation=act),
		normalization.BatchNormalization(),
		advanced_activations.LeakyReLU(alpha=0.3),
		convolutional.Conv2DTranspose(32,(4,4),strides=(2, 2), padding='same',activation=act),
		normalization.BatchNormalization(),
		advanced_activations.LeakyReLU(alpha=0.3),
		convolutional.Conv2DTranspose(16,(4,4),strides=(2, 2), padding='same',activation=act),
		normalization.BatchNormalization(),
		advanced_activations.LeakyReLU(alpha=0.3),
		convolutional.Conv2DTranspose(8,(4,4),strides=(2, 2), padding='same',activation=act),
		normalization.BatchNormalization(),
		advanced_activations.LeakyReLU(alpha=0.3),
		convolutional.Conv2DTranspose(8,(4,4),strides=(1, 1), padding='same',activation=act),
		normalization.BatchNormalization(),
		advanced_activations.LeakyReLU(alpha=0.3),
		convolutional.Conv2DTranspose(8,(4,4),strides=(1, 1), padding='same',activation=act),
		normalization.BatchNormalization(),
		advanced_activations.LeakyReLU(alpha=0.3),
		convolutional.Conv2DTranspose(nch,(1,1),strides=(1, 1), padding='same'),
		#convolutional.UpSampling2D(size=(1, 1)),
		Reshape([128*128*nch])
		]

	#seq=[
	#	Dense(128,activation="relu"),
	#	Dense(784,activation="sigmoid")
	#]
	for layer in seq:
		hidden=layer(hidden)
	return hidden


def discriminative_network(x):
	#seq=[
	#	Dense(128,activation="relu"),
	#	Dense(1,activation=None)
	#]
	hidden = tf.reshape((tf.cast(x, tf.float32)),[M,128,128,nch])
	act=None
	seq=[
		convolutional.Conv2D(32,(4,4),strides=(2, 2), padding='same',activation=act),
		#normalization.BatchNormalization(),
		advanced_activations.LeakyReLU(alpha=0.3),
		convolutional.Conv2D(64,(4,4),strides=(2, 2), padding='same',activation=act),
		#64x64
		#normalization.BatchNormalization(),
		advanced_activations.LeakyReLU(alpha=0.3),
		convolutional.Conv2D(128,(4,4),strides=(2, 2), padding='same',activation=act),
		#normalization.BatchNormalization(),
		advanced_activations.LeakyReLU(alpha=0.3),
		#32x32
		Flatten(),
		#convolutional.Conv2D(256,(4,4),strides=(2, 2), padding='same',activation=act),
		#normalization.BatchNormalization(),
		Dense(4*4*256,activation=act),
		advanced_activations.LeakyReLU(alpha=0.3),
		Dense(2,activation=None),
		]

	for layer in seq:
		hidden=layer(hidden)
		print(hidden.shape)
	return hidden


def plot(samples):
	fig = plt.figure(figsize=(4, 4))
	gs = gridspec.GridSpec(4, 4)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(128, 128,nch))
		#plt.imshow(sample.reshape(128, 128), cmap='Greys_r')

	return fig


ed.set_seed(42)

data_dir = "./data"
out_dir = "./out"
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

# DATA. MNIST batches are fed at training time.

#(x_train, _), (x_test, _) = mnist(data_dir)
x_data=np.load("data.npy")
n=int(x_data.shape[0]*0.9)
x_train=x_data[:1000]

x_train_generator = generator(x_train, M)
x_ph = tf.placeholder(tf.float32, [M, 128*128*nch])

# MODEL
with tf.variable_scope("Gen"):
	#eps = Uniform(low=tf.zeros([M, D]) - 1.0, high=tf.ones([M, D]))
	eps = Normal(loc=tf.zeros([M, D]), scale=tf.ones([M, D]))
	x = generative_network(eps)

# INFERENCE
#optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)
#optimizer_d = tf.train.RMSPropOptimizer(learning_rate=5e-6)
optimizer = tf.train.AdamOptimizer(learning_rate=5e-5)
optimizer_d = tf.train.AdamOptimizer(learning_rate=5e-6)

inference = ed.GANInference(
		data={x: x_ph}, discriminator=discriminative_network)
inference.initialize(
		optimizer=optimizer, optimizer_d=optimizer_d)
		#n_iter=15000, n_print=1000)
#		n_iter=15000, n_print=1000, clip=0.01, penalty=None)

sess = ed.get_session()
tf.global_variables_initializer().run()

i = 0
n_iter_per_epoch=x_train.shape[0] // M
for epoch in range(nepoch):
	pbar = Progbar(n_iter_per_epoch)
	loss_g=0
	loss_d=0
	for t in range(n_iter_per_epoch):
		x_batch = next(x_train_generator)
		info_d=inference.update(feed_dict={x_ph: x_batch}, variables="Disc")
		loss_d+=info_d["loss_d"]
		for _ in range(5):
			info_g= inference.update(feed_dict={x_ph: x_batch}, variables="Gen")
		loss_g+=info_g["loss"]
		pbar.update(t)
	print("loss gen: %3.6f loss disc: %3.6f"%(loss_g,loss_d))
		# note: not printing discriminative objective; ``info_dict`` above
		# does not store it since updating only "Gen"
		#info_dict['t'] = info_dict['t'] // 2	# say set of 6 updates is 1 iteration
		#inference.print_progress(info_dict)


	idx = np.random.randint(M, size=16)
	samples = sess.run(x)
	samples = samples[idx, ]
	
	fig = plot(samples)
	plt.savefig(os.path.join(out_dir, '{}.png').format(
	str(i).zfill(3)), bbox_inches='tight')
	plt.close(fig)
	i+=1


