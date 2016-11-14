import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.mul(a,b)

sess = tf.Session()

print sess.run(add, feed_dict={a:2, b:33})
print sess.run(mul, feed_dict={a:24, b:22})