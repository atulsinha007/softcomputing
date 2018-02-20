import tensorflow as tf
import numpy
inputdim = 1
outputdim = 1
x=tf.placeholder('float')
tar=tf.placeholder('float')
traindata = [2014,1600, 2400, 1416, 3000]
trainout = [400, 330, 369, 232, 540]
traindata = numpy.array( traindata )
trainout = numpy.array( trainout )
training_steps = 1000

# w1=tf.Variable(tf.random_normal([inputdim, outputdim]))
w1 = tf.Variable([0.2])
# b1=tf.Variable(tf.random_normal([outputdim]))
b1 = tf.Variable([-.0002])
#prediction=tf.nn.relu(tf.add(tf.matmul(x,w1),b1))
prediction=x*w1 + b1

cost=tf.losses.mean_squared_error(labels = tar, predictions= prediction)
learing_rate = 0.000000005
optmzr = tf.train.GradientDescentOptimizer(learing_rate).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_steps):
	           p, c, pred = sess.run([optmzr, cost, prediction], feed_dict={x:traindata, tar:trainout})
	           # print("epoch no = ", epoch, pred)
	           # print("cost = ", c)
	#print("cost value = ", c)
	print("weight = ", w1.eval())
	print("bias = ", b1.eval())
	print("prediction = ", pred)
	print("Final cost = ", c)
	print("--------------------------------------------\n\t\tPredicting y for x\n--------------------------------------------")
	for i in range(1200, 4500, 200):
		pre = i*w1.eval() + b1.eval()
		print("x = ", i, "prediction = ",pre)

	

