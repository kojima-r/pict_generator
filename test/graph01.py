import tensorflow as tf
import numpy as np


with tf.Session() as sess:
	a=tf.Variable(1.0,tf.float32,name="a")
	b=tf.Variable(2.0,tf.float32,name="b")
	c=tf.Variable(3.0,tf.float32,name="c")
	o=a*b+c
	init = tf.global_variables_initializer()
	sess.run(init)
	print(o)
	summary_writer = tf.summary.FileWriter('./log',graph = sess.graph)
	print(sess.run(o))
	
