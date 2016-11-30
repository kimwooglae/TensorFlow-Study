import tensorflow as tf
import input_data as input_data
import matplotlib.pyplot as plt
import random
import datetime

def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

learning_rate = 0.0001
training_epochs = 50
#batch_size = [1, 2, 4, 5, 10, 11, 20, 22, 25, 44, 50, 55, 100, 110, 125, 220, 250, 275, 500, 550]
#batch_size = [20, 22, 25, 44, 50, 55, 100, 110, 125, 220, 250, 275, 500, 550]
batch_size = [20, 44, 55, 100, 125, 220]
#batch_size = [25, 44, 50, 55, 100, 110, 125, 220, 250, 275, 500, 550]
display_step = 1
dropout_rateArr = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
#dropout_rateArr = [0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]


#tf graph input
x = tf.placeholder("float", [None, 784])  # 28 * 28 = 784
y = tf.placeholder("float", [None, 10])  # 0-9 ==> 10

#create model


#set model weights
W1 = tf.get_variable("W1", shape=[784,1031], initializer=xaver_init(784,1031))
W2 = tf.get_variable("W2", shape=[1031,1993], initializer=xaver_init(1031,1993))
W3 = tf.get_variable("W3", shape=[1993,2017], initializer=xaver_init(1993,2017))
W4 = tf.get_variable("W4", shape=[2017,1031], initializer=xaver_init(2017,1031))
W5 = tf.get_variable("W5", shape=[1031,1993], initializer=xaver_init(1031,1993))
W6 = tf.get_variable("W6", shape=[1993,2017], initializer=xaver_init(1993,2017))
W7 = tf.get_variable("W7", shape=[2017,1031], initializer=xaver_init(2017,1031))
W8 = tf.get_variable("W8", shape=[1031,1993], initializer=xaver_init(1031,1993))
W9 = tf.get_variable("W9", shape=[1993,2017], initializer=xaver_init(1993,2017))
W10 = tf.get_variable("W10", shape=[2017,1031], initializer=xaver_init(2017,1031))
W11 = tf.get_variable("W11", shape=[1031,1993], initializer=xaver_init(1031,1993))
W12 = tf.get_variable("W12", shape=[1993,2017], initializer=xaver_init(1993,2017))
W13 = tf.get_variable("W13", shape=[2017,1031], initializer=xaver_init(2017,1031))
W14 = tf.get_variable("W14", shape=[1031,1993], initializer=xaver_init(1031,1993))
W15 = tf.get_variable("W15", shape=[1993,2017], initializer=xaver_init(1993,2017))
W16 = tf.get_variable("W16", shape=[2017,1031], initializer=xaver_init(2017,1031))
W17 = tf.get_variable("W17", shape=[1031,1993], initializer=xaver_init(1031,1993))
W18 = tf.get_variable("W18", shape=[1993,2017], initializer=xaver_init(1993,2017))
W19 = tf.get_variable("W19", shape=[2017,1031], initializer=xaver_init(2017,1031))
W20 = tf.get_variable("W20", shape=[1031,10], initializer=xaver_init(1031,10))		#0.9672	#0.9696

#W1 = tf.Variable(tf.random_normal([784, 1031]))
#W2 = tf.Variable(tf.random_normal([1031, 1031]))
#W3 = tf.Variable(tf.random_normal([1031, 10]))

B1 = tf.Variable(tf.random_normal([1031]))
B2 = tf.Variable(tf.random_normal([1993]))
B3 = tf.Variable(tf.random_normal([2017]))
B4 = tf.Variable(tf.random_normal([1031]))
B5 = tf.Variable(tf.random_normal([1993]))
B6 = tf.Variable(tf.random_normal([2017]))
B7 = tf.Variable(tf.random_normal([1031]))
B8 = tf.Variable(tf.random_normal([1993]))
B9 = tf.Variable(tf.random_normal([2017]))
B10 = tf.Variable(tf.random_normal([1031]))
B11 = tf.Variable(tf.random_normal([1993]))
B12 = tf.Variable(tf.random_normal([2017]))
B13 = tf.Variable(tf.random_normal([1031]))
B14 = tf.Variable(tf.random_normal([1993]))
B15 = tf.Variable(tf.random_normal([2017]))
B16 = tf.Variable(tf.random_normal([1031]))
B17 = tf.Variable(tf.random_normal([1993]))
B18 = tf.Variable(tf.random_normal([2017]))
B19 = tf.Variable(tf.random_normal([1031]))
B20 = tf.Variable(tf.random_normal([10]))

dropout_rate = tf.placeholder("float")
#construct model
_L1 = tf.nn.relu(tf.add(tf.matmul(x,W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2), B2))
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2,W3), B3))
L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3,W4), B4))
L4 = tf.nn.dropout(_L4, dropout_rate)

#activation = tf.add(tf.matmul(L4,W20), B20)

_L5 = tf.nn.relu(tf.add(tf.matmul(L4,W5), B5))
L5 = tf.nn.dropout(_L5, dropout_rate)
_L6 = tf.nn.relu(tf.add(tf.matmul(L5,W6), B6))
L6 = tf.nn.dropout(_L6, dropout_rate)
_L7 = tf.nn.relu(tf.add(tf.matmul(L6,W7), B7))
L7 = tf.nn.dropout(_L7, dropout_rate)

activation = tf.add(tf.matmul(L7,W20), B20)

#_L8 = tf.nn.relu(tf.add(tf.matmul(L7,W8), B8))
#L8 = tf.nn.dropout(_L8, dropout_rate)
#_L9 = tf.nn.relu(tf.add(tf.matmul(L8,W9), B9))
#L9 = tf.nn.dropout(_L9, dropout_rate)
#_L10 = tf.nn.relu(tf.add(tf.matmul(L9,W10), B10))
#L10 = tf.nn.dropout(_L10, dropout_rate)

#activation = tf.add(tf.matmul(L10,W20), B20)

#_L11 = tf.nn.relu(tf.add(tf.matmul(L10,W11), B11))
#L11 = tf.nn.dropout(_L11, dropout_rate)
#_L12 = tf.nn.relu(tf.add(tf.matmul(L11,W12), B12))
#L12 = tf.nn.dropout(_L12, dropout_rate)
#_L13 = tf.nn.relu(tf.add(tf.matmul(L12,W13), B13))
#L13 = tf.nn.dropout(_L13, dropout_rate)

#_L14 = tf.nn.relu(tf.add(tf.matmul(L13,W14), B14))
#L14 = tf.nn.dropout(_L14, dropout_rate)
#_L15 = tf.nn.relu(tf.add(tf.matmul(L14,W15), B15))
#L15 = tf.nn.dropout(_L15, dropout_rate)
#_L16 = tf.nn.relu(tf.add(tf.matmul(L15,W16), B16))
#L16 = tf.nn.dropout(_L16, dropout_rate)

#_L17 = tf.nn.relu(tf.add(tf.matmul(L16,W17), B17))
#L17 = tf.nn.dropout(_L17, dropout_rate)
#_L18 = tf.nn.relu(tf.add(tf.matmul(L17,W18), B18))
#L18 = tf.nn.dropout(_L18, dropout_rate)
#_L19 = tf.nn.relu(tf.add(tf.matmul(L18,W19), B19))
#L19 = tf.nn.dropout(_L19, dropout_rate)

#activation = tf.add(tf.matmul(L19,W20), B20)
#activation = tf.nn.softmax(tf.matmul(x,W) + b)

#minimize error using cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(activation, y))
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation),reduction_indices=1))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) #O.9582
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #0.9254

init = tf.initialize_all_variables()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
checkpoint_dir = "cps/"

print "mnist.train.num_examples: ",  mnist.train.num_examples

NUM_THREADS = 50
#sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
	print ('load learning')
	saver.restore(sess, ckpt.model_checkpoint_path)

for epoch in range(training_epochs):
	avg_cost = 0.
	total_batch = int(mnist.train.num_examples/batch_size[epoch%len(batch_size)])
	# Loop over all batches
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size[epoch%len(batch_size)])
		#Fit training using batch data
		sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys, dropout_rate:dropout_rateArr[epoch%len(dropout_rateArr)]})
		#compute average loss
		avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys, dropout_rate:dropout_rateArr[epoch%len(dropout_rateArr)]})/total_batch
	# Display logs per epoch step
	if epoch % display_step == 0:
		print datetime.datetime.now(), " Batch: ", batch_size[epoch%len(batch_size)] ," dropout_rate: ", dropout_rateArr[epoch%len(dropout_rateArr)] , " Epoch:", "%04d" % (epoch+1), "cost=", "{:9f}".format(avg_cost)
print "Optimization Finished!"

# Test model
correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print "Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
print "Accuracy: ", sess.run(accuracy, {x:mnist.test.images, y:mnist.test.labels, dropout_rate:1})

#Get one and predict

r = random.randint(0, mnist.test.num_examples -1)
print "Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))
print "Prediction: ", sess.run(tf.argmax(activation,1), {x:mnist.test.images[r:r+1], dropout_rate:1.0})

#Show the image
#plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap="Greys", interpolation="nearest")
#plt.show()




 