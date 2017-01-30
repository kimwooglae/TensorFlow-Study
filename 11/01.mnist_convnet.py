import tensorflow as tf
import input_data as input_data
import matplotlib.pyplot as plt
import random
import numpy as np

def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")




w = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))
w2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
w3 = tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01))
w4 = tf.Variable(tf.random_normal([128 * 4 * 4, 625],stddev=0.01))
w_o = tf.Variable(tf.random_normal([625, 10],stddev=0.01))



l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1,1,1,1], padding="SAME"))
l1 = tf.nn.max_pool(l1a, ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")
l1 = tf.nn.dropout(l1, p_keep_conv)

l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1,1,1,1], padding="SAME"))
l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")
l2 = tf.nn.dropout(l2, p_keep_conv)

l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1,1,1,1], padding="SAME"))
l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")
l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
l3 = tf.nn.dropout(l3, p_keep_conv)

l4 = tf.nn.relu(tf.matmul(l3, w4))
l4 = tf.nn.dropout(l4, p_keep_conv)

py_x = tf.matmul(l4, w_o)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv:0.8, p_keep_hidden:0.5})

        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:256]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                                                                 Y: teY[test_indices],
                                                                                                 p_keep_conv : 1.0,
                                                                                                 p_keep_hidden :1.0})))

