import numpy as np 
import tensorflow as tf
import river_load_data
import cv2
import test

epochs = 1000
learning_rate = 0.0001

def sigmoid_aux(x):
	z = 1/ (1+ np.exp(-x))
	if z < 0.5:
		return 0
	else:
		return 1

def print_y(expected, y):
	rows = len(y)
	for i in range(rows):
		print(expected[i], "\t", y[i])

def denormalize(y, maxx, minn):
	y = y*(maxx[len(maxx)-1] - minn[len(minn)-1]) + minn[len(minn)-1]
	return y

def output(weight, bias, testx, testy, maxx, minn):
	y = []
	count = 0
	for i in range(len(testx)):
		temp = np.add(np.matmul(testx[i], weight), bias)
		temp = sigmoid_aux(temp)
		if temp == 1:
			count += 1
		y.append(temp)
	# print_y(y, testy)
	testy = denormalize(testy, maxx, minn)
	y = np.asarray(y)
	y = y.astype(float)
	y = denormalize(y, maxx, minn)
	accuracy = test.accuracy(y, testy)
	cnt = 0
	
	for i in testy:
		if i == 1:
			cnt += 1
	# print("actual ones = ", cnt)
	# print("estimated ones = ",count, len(testx))
	print("\n******************Accuracy******************\n")
	print("Accuracy = ", accuracy, "%")

def main():

	img = cv2.imread("river.jpg")
	img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
	trainx, trainy, testx, testy, maxx, minn = river_load_data.create_dataset(img)
	
	inputdim = trainx[0].size
	outputdim = trainy[0].size
	x = tf.placeholder('float',[None,inputdim])
	tar = tf.placeholder('float')
	# weights = tf.Variable(tf.random_uniform([inputdim, outputdim], 0, 1))
	# biases = tf.Variable(tf.random_normal([outputdim], -1, 1))
	
	weights = tf.Variable([[ 0.84668195],[ 0.14493632]])
	biases = tf.Variable([-0.861283034])
	prediction = tf.nn.sigmoid(tf.add(tf.matmul(x, weights), biases)) 
	cost = tf.reduce_mean(-tf.reduce_sum(tar * tf.log(prediction), reduction_indices=1))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("\nInitial weights = ", weights.eval())
		print("\nInitial bias = ", biases.eval())
		for epoch in range(epochs):
		   _, c= sess.run([optimizer, cost], feed_dict = {x:trainx, tar:trainy})
		weight = weights.eval()
		bias = biases.eval()
	output(weight, bias, testx, testy, maxx, minn)
	print("\nFinal weights = ", weight)
	print("\nFinal bias = ", bias)
	

if __name__ == '__main__':
	main()
