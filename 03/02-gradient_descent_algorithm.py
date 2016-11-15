import tensorflow as tf

#x_data = [1.,2.,10.]
#y_data = [10.,20.,100.]
x_data = [2.,3.,9.,10.]
y_data = [60.,50.,80.,90.]

W = tf.Variable(tf.random_uniform([1], -100.0, 100.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

descent = W - tf.mul(0.01, tf.reduce_mean(tf.mul((tf.mul(W,X)-Y),X)))
W1 = W.assign(descent)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(30):
	sess.run(W1, feed_dict={X:x_data, Y:y_data})
	print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)