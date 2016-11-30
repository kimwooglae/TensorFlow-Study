import tensorflow as tf
import input_data as input_data
import matplotlib.pyplot as plt
import random

def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1



#tf graph input
x = tf.placeholder("float", [None, 784])  # 28 * 28 = 784
y = tf.placeholder("float", [None, 10])  # 0-9 ==> 10

#create model


#set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#construct model
activation = tf.nn.softmax(tf.matmul(x,W) + b)

#minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation),reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
checkpoint_dir = "cps/"

NUM_THREADS = 20
sess = tf.Session(config=tf.ConfigProto(
    intra_op_parallelism_threads=NUM_THREADS))
#sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
	print ('load learning')
	saver.restore(sess, ckpt.model_checkpoint_path)

for epoch in range(training_epochs):
	avg_cost = 0.
	total_batch = int(mnist.train.num_examples/batch_size)
	# Loop over all batches
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		#Fit training using batch data
		sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})
		#compute average loss
		avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys})/total_batch
	# Display logs per epoch step
	if epoch % display_step == 0:
		print "Epoch:", "%04d" % (epoch+1), "cost=", "{:9f}".format(avg_cost)
print "Optimization Finished!"

# Test model
correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print "Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
print "Accuracy: ", sess.run(accuracy, {x:mnist.test.images, y:mnist.test.labels})

#Get one and predict

r = random.randint(0, mnist.test.num_examples -1)
print "Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))
print "Prediction: ", sess.run(tf.argmax(activation,1), {x:mnist.test.images[r:r+1]})

#Show the image
plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap="Greys", interpolation="nearest")
plt.show()




 