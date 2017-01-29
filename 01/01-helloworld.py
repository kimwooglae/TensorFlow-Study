import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!!!')

print(hello)

session = tf.Session()

print(session.run(hello))

a = tf.constant(2)
b = tf.constant(3)

c =  a + b

print(a)
print(b)
print(c)

print(session.run(c))


