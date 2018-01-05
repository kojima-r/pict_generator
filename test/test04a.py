from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Categorical, Dirichlet, Normal


D = 4  # number of documents
#N = [20, 30, 30, 10]  # words per doc
N = [int(np.random.uniform(10,30)) for d in range(D)]
print(N)
K = 3  # number of topics
###
theta_true=np.zeros([D, K])
for d in range(D):
	theta_true[d,:] = np.random.dirichlet(np.array([1.0]*K))
phi_true=np.zeros([K])
for k in range(K):
	phi_true[k] = np.random.normal(loc=k*10.0,scale=0.1)
z_true=[ np.zeros([N[d]]) for d in range(D)]
w_data=[ np.zeros([N[d]]) for d in range(D)]
print(phi_true)
for d in range(D):
	for n in range(N[d]):
		z_true[d][n]= np.random.choice(K, p=theta_true[d,:])
		w_data[d][n]= np.random.normal(loc=phi_true[int(z_true[d][n])],scale=1)
###
theta = Dirichlet(tf.zeros([D, K]) + 0.1)
phi = Normal(loc=tf.zeros([K]), scale=tf.ones([K]))
z = [[0] * N[d] for d in  range(D)]
w = [[0] * N[d] for d in  range(D)]
for d in range(D):
	for n in range(N[d]):
		z[d][n] = Categorical(theta[d, :])
		w[d][n] = Normal(loc=phi[z[d][n]],scale=1.0)

###
# INFERENCE
qtheta = Dirichlet(tf.nn.softplus(tf.Variable(tf.random_normal([D,K]))))
qphi = Normal(loc=tf.Variable(tf.random_normal([K])),scale=tf.ones([K]))
qz = [[None] * N[d] for d in  range(D)]
for d in range(D):
	for n in range(N[d]):
		qz_var = tf.Variable(tf.zeros(K)+0.1)
		qz[d][n] = Categorical(logits=qz_var)

data={}
for d in range(D):
	for n in range(N[d]):
		data[w[d][n]]=w_data[d][n]
hvars={theta: qtheta, phi:qphi}
for d in range(D):
	for n in range(N[d]):
		hvars[z[d][n]]=qz[d][n]

inference = ed.KLqp(hvars, data=data)
#inference = ed.Gibbs(hvars, data=data)
inference.run(n_iter=1000, n_samples=50)

sess = ed.get_session()
print('Inferred theta={}'.format(sess.run(qphi.mean())))

