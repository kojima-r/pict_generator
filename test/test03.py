
import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import six
import tensorflow as tf
from edward.models import Categorical, Dirichlet, Empirical, InverseGamma, \
    MultivariateNormalDiag, Normal, ParamMixture




D=3
K=10
x = Normal(tf.zeros(D),
	tf.ones(D),
	sample_shape=K)
print(x)
#s=ed.Gibbs({x:x}) 
#inference.initialize()
sess = ed.get_session()
#tf.global_variables_initializer().run()
print(sess.run(x._sample_n(10)))

