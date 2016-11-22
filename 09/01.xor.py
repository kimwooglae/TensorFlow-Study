import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')

x_data = np.transpose(xy[0:-1])
y_data = np.transpose(xy[-1])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))

h = tf.matmul(X, W)
hypothesis = tf.div(1., 1.+tf.exp(-h))

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print '------------------------------------------'
for step in xrange(2001):
	sess.run(train, feed_dict={X:x_data, Y:y_data})
	if step % 200 == 0:
		print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)


correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data,Y:y_data})
print "Accuracy:", sess.run( accuracy, feed_dict={X:x_data, Y:y_data})
