#!/usr/bin/env python
"""Dirichlet-Categorical model.
Posterior inference with Edward's BBVI.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Categorical, Dirichlet, Normal, PointMass,Empirical

N = 1000
K = 4

# DATA
pi_true = np.random.dirichlet(np.array([20.0, 30.0, 10.0, 10.0]))
z_data = np.array([np.random.choice(K, 1, p=pi_true)[0] for n in range(N)])
x_data=[0]*N
for n in range(N):
	k=z_data[n]
	print(k)
	x_data[n] = np.random.normal(loc=(k-1)*10.0,scale=0.1)
print('pi={}'.format(pi_true))

# MODEL
pi = Dirichlet(tf.ones(4))
z = Categorical(probs=tf.ones([N, 1]) * pi)
m=[0.0 for k in range(K)]
#m=tf.Variable(tf.zeros((K,)))
s=[1.0 for k in range(K)]
phi = Normal(loc=m,scale=s)
mu=tf.gather(phi,z)
x = Normal(loc=mu,scale=[1.0]*N)

print(x)
# INFERENCE
qpi = Dirichlet(tf.nn.softplus(tf.Variable(tf.random_normal([K]))))
qz_var = tf.Variable(tf.zeros((N,K))+0.1)
qz = Categorical(logits=qz_var)
qphi = Normal(loc=tf.Variable(tf.zeros((K,))),scale=[1.0]*K)
#

qpi = Empirical(tf.Variable(tf.ones([K]) / K))
qphi = Empirical(tf.Variable(tf.zeros([K])))
qz = Empirical(tf.Variable(tf.zeros([N,K])))

#qz = Empirical(tf.Variable(tf.zeros([N, K], dtype=tf.int32)))
#qz = PointMass(params=tf.Variable(tf.zeros([N], dtype=tf.int32)))

learning_rate=0.0001
#inference = ed.KLqp({pi: qpi,phi:qphi,z:qz}, data={x: x_data})
#inference = ed.MAP({pi: qpi,phi:qphi}, data={x: x_data})
inference = ed.Gibbs({pi: qpi,z:qz,phi:qphi}, data={x: x_data})
#inference.run(n_iter=1500, n_samples=100,optimizer = tf.train.AdamOptimizer(learning_rate))

inference.run(n_iter=15000)

sess = ed.get_session()
print('Inferred pi={}'.format(sess.run(qphi.mean())))

