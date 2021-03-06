import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')

x_data1 = xy[0:3]
y_data1 = xy[3:]
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

X = tf.placeholder('float32', [None, 3])
Y = tf.placeholder('float32', [None, 3])

#W =  tf.Variable(tf.zeros([3, 3]))
W = tf.Variable([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]])
#W = tf.Variable([[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]])
#W = tf.Variable([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])

hypothesis = tf.nn.softmax(tf.matmul(X, W))

learning_rate = 0.01

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	print('W ------------------------------------------')
	print(sess.run(W))
	for step in range(2000001):
		sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
		if step % 5000 == 0:
			print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))
	print('------------------------------------------')
	a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
	print(a, sess.run(tf.argmax(a,1)))

	b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
	print(b, sess.run(tf.argmax(b,1)))

	c = sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
	print(c, sess.run(tf.argmax(c,1)))

	all = sess.run(hypothesis, feed_dict={X:[[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
	print(all, sess.run(tf.argmax(all, 1)))

	print('W ------------------------------------------')
	print(sess.run(W))

	print('x_data ------------------------------------------')
	print(x_data)

	print('x_data1 ------------------------------------------')
	print(x_data1)

	print('xy ------------------------------------------')
	print(xy)
#	print(tf.matmul(X,W))
