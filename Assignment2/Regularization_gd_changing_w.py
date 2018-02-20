import numpy as np 
import tensorflow as tf
import load_data
import test

lambda_val = 100
epochs = 4000
learning_rate = 0.0001
# ********************hypothesis w0 + w1x0 + w2x1 + w3x0^2 + w4x1^2 + w5x0x1 + w6x0^2x1^2*********************
def normalization_x(x):
	maxx = np.max(x, axis = 0)
	minn = np.min(x, axis = 0)
	for row in x:
		for i in range(row.size):
			row[i] = (row[i] - minn[i])
			row[i] = row[i]/(maxx[i] - minn[i])
			
	return x

def normalization_y(x):
	maxx = np.max(x, axis = 0)
	minn = np.min(x, axis = 0)
	print(maxx, minn)
	for i in range(len(x)):
		x[i] = x[i] - minn
		x[i] = x[i]/(maxx-minn)
	return x


def convert(x):
	aux_lis = []
	for z in x:
		temp2 = (z[0], z[1], z[0]*z[0], z[1]*z[1], z[0]*z[1], z[0]*z[0]*z[1]*z[1] )
		aux_lis.append(temp2)
	x = np.array(aux_lis)
	return x

def print_y(expected, y):
	# print(np.shape(expected))
	# print(np.shape(y))
	rows = len(y)
	for i in range(rows):
		print(expected[i], "\t", y[i])

def output(weight, bias, testx, testy):
	y = []
	maxx = np.max(testy, axis = 0)
	minn = np.min(testy, axis = 0)
	for i in range(len(testx)):
		temp = np.add(np.matmul(testx[i], weight), bias)
		# temp = temp*(maxx - minn) + minn
		y.append(temp)
	# print_y(y, testy)
	average_error = test.accuracy(y, testy)
	print(average_error)

def main(parameter):
	trainx, trainy, testx, testy = load_data.load_data()
	trainx = convert(trainx)
	testx = convert(testx)
	trainx = trainx.astype(float)
	trainy = trainy.astype(float)
	trainx = normalization_x(trainx)
	testx = normalization_x(testx)
	# trainy = normalization_y(trainy)
	inputdim = trainx[0].size
	outputdim = trainy[0].size
	x = tf.placeholder('float',[None,inputdim])
	tar = tf.placeholder('float')
	weights = tf.Variable(tf.random_uniform([inputdim, outputdim], -1.0, 1.0))
	ses = tf.InteractiveSession()
	ses.run(weights.initializer)
	reg = weights.eval()
	# print(reg.shape)
	reg = reg[parameter:, :]
	# print(reg.shape)
	# print(type(reg))
	# print("--",reg)
	ses.close()
	reg = tf.Variable(reg)
	biases = tf.Variable(tf.random_uniform([outputdim], -1.0, 1.0))
	prediction = tf.add(tf.matmul(x, weights), biases)
	cost = tf.losses.mean_squared_error(labels = tar, predictions = prediction)
	regularization_parameter = tf.nn.l2_loss(reg)
	cost = cost + (lambda_val * regularization_parameter)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
		   p, c, pred = sess.run([optimizer, cost, prediction], feed_dict = {x:trainx, tar:trainy})
		weight = weights.eval()
		bias = biases.eval()
	# print(np.shape(weight))
	# print(np.shape(testx))
	output(weight, bias, testx, testy)



if __name__ == '__main__':
	for i in range(6):
		main(i)