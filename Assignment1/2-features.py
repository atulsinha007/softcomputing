import tensorflow as tf
import numpy
inputdim = 2
outputdim = 1
def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

x=tf.placeholder('float',[None,inputdim])
tar=tf.placeholder('float')
#par=tf.placeholder('float')
#traindata, trainout
traindata = [[3, 2014],[3, 1600], [ 3, 2400], [2, 1416], [4, 3000]]
trainout = [400, 330, 369, 232, 540]
traindata = numpy.array( traindata )
trainout = numpy.array( trainout )
training_steps = 100

# w1=tf.Variable(tf.random_normal([inputdim, outputdim]))
# b1=tf.Variable(tf.random_normal([outputdim]))
w1 = tf.Variable([[0.2], [0.4]])
b1 = tf.Variable([-0.2])
prediction=tf.nn.relu(tf.add(tf.matmul(x,w1),b1))

cost=tf.losses.mean_squared_error(labels = tar, predictions= prediction)
optmzr = tf.train.GradientDescentOptimizer(0.000000005).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#print(sess.run(cost,feed_dict={x:traindata,tar:trainout}))
	for epoch in range(training_steps):
	           p, c, pred = sess.run([optmzr, cost, prediction], feed_dict={x:traindata, tar:trainout})
	           # print("epoch no = ", epoch, pred)
	           # print("cost = ", c)
	#correct = tf.equal(tf.cast(tf.argmax(prediction, axis=1),'int8'), par)
	print("cost value = ", c)
	#cost_val = sess.run(loss, )
	weight = w1.eval()
	b = b1.eval()
	print("weights = ", w1.eval())
	print("bias = ", b1.eval())
	print("pred = ", pred)
	arr = numpy.array([[i,j] for i in range(1200, 4000, 600) for j in range(2, 6) ])
	print("Final cost = ", c)
	print("--------------------------------------------\n\t\tPredicting y for x\n--------------------------------------------")		
	x1 = 1200
	length = len(arr)
	i1 = 0
	i2 = 2
	x2 = 2
	for x in arr:
		print("x1 = ", x1, "x2 = ", x2, "prediction = ", ReLU(numpy.dot(x,weight)+b))
		i1 += 1
		i2 = (i2+1)%4
		x2 = 2 + i2
		if(i1%4 == 0):
			x1 += 500
