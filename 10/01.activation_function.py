# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True)

x_data = np.transpose(xy[0:-1])
#y_data = xy[-1]
y_data = np.transpose([xy[-1]])
#y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2,4], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([4,4], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([4,1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([4]), name="Bias1")
b2 = tf.Variable(tf.zeros([4]), name="Bias2")
b3 = tf.Variable(tf.zeros([1]), name="Bias3")

#L2 = tf.nn.relu(tf.matmul(X, W1) + b1)
#L3 = tf.nn.relu(tf.matmul(L2, W2) + b2)
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

learning_rate = tf.Variable(0.2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

#sess = tf.Session()
with tf.Session() as sess:
	sess.run(init)

	print('------------------------------------------')
	for step in range(10000):
		sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
		if step % 200 == 0:
			print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
			print("   W1:", sess.run(W1))
			print("   W2:", sess.run(W2))


	correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print('------------------------------------------')
	print(sess.run(tf.floor(hypothesis+0.5), feed_dict={X:x_data,Y:y_data}))
	print('------------------------------------------')
	print(sess.run(Y, feed_dict={X:x_data,Y:y_data}))
	print('------------------------------------------')
	print(sess.run(correct_prediction, feed_dict={X:x_data,Y:y_data}))
	print('------------------------------------------')
	print(sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data,Y:y_data}))
	print('------------------------------------------')
	#print("Accuracy:", sess.run( accuracy, feed_dict={X:x_data, Y:y_data}))
	print("Accuracy:", accuracy.eval({X:x_data, Y:y_data}))	#with tf.Session() as sess: 로 sess를 생성해야 동작함
	print('------------------------------------------')
	print(x_data)
	print('------------------------------------------')
	print(y_data)

