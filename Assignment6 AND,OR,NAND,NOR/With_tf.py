import tensorflow as tf 
import numpy as np
import load_data

epochs = 10
alpha = 0.001
def sigmoid(x):
	return (1/(1+ np.exp(-x)))
	
def test(weight, bias, testx, testy):

	print(weight.shape)
	print(bias.shape)
	print(testx.shape)
	for i in range(len(testy)):
		t = testx[i]
		t = np.reshape(t, (t.shape[0], 1))
		y = sigmoid(np.add(np.matmul(weight.T, t), bias))
		print(y)
	# print(y)


def main():

	trainx, trainy, testx, testy = load_data.load_data()
	inputdim = trainx.shape[1]
	outputdim = trainy.shape[1]

	x = tf.placeholder('float',[None,inputdim])
	y = tf.placeholder('float')

	weights = tf.Variable(tf.random_normal([inputdim, outputdim]))
	biases = tf.Variable(tf.random_normal([outputdim]))
	prediction = tf.nn.sigmoid(tf.add(tf.matmul(x, weights), biases))
	cost = tf.losses.mean_squared_error(labels = y, predictions = prediction)
	optmzr = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# print("Initial weights:\n", weights.eval())
		# print("Initial biases:\n", biases.eval())
		for epoch in range(epochs):
		           p, c, pred = sess.run([optmzr, cost, prediction], feed_dict = {x:trainx, y:trainy})
		print("cost value = ", c)
		weight = weights.eval()
		bias = biases.eval()
	# print(weight)
	test(weight, bias, testx, testy)


if __name__ == '__main__':
	main()