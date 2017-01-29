import tensorflow as tf

#x_data = [1.,2.,10.]
#y_data = [1.,2.,10.]
x_data = [2.,3.,9.,10.]
y_data = [20.,30.,90.,100.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

descent = W - tf.mul(0.01, tf.reduce_mean(tf.mul((tf.mul(W,X)-Y),X)))
update = W.assign(descent)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(100):
	sess.run(update, feed_dict={X:x_data, Y:y_data})
	print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))
# 	print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))
