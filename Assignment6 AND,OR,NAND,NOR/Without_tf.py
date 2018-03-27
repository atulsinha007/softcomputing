import numpy as np
import load_data

epochs = 500
alpha = 0.1

def activation(val):
	if val[0][0] <= 0:
		return 0
	else:
		return 1

def test(weights, biases, testx, testy):

	print("OUTPUT\tACTUAL")
	for i in range(len(testy)):
		t = testx[i].reshape((1,2))
		output = np.dot(t, weights) + biases
		output = activation(output)
		print(output,"\t",testy[i][0])



def main():

	print("Input filename")
	data = input()
	trainx, trainy, testx, testy = load_data.load_data(data)
	inputdim = trainx.shape[1]
	outputdim = trainy.shape[1]

	weights = np.random.normal(-0.5, 0.5, (inputdim, outputdim))
	biases = np.random.normal(-0.5, 0.5, (outputdim))
	print("INITIAL\n")
	print(weights.flatten())
	print(biases)
	print("\n")
	for epoch in range(epochs):
		for i in range(4):
			t = trainx[i].reshape((1,2))
			output = np.dot(t, weights) + biases
			output = activation(output)
			delta = (trainy[i] - output)
			weights = weights + delta*alpha*t.T
			biases = biases + delta*alpha*1
	print("AFTER TRAINING\n")
	print(weights.flatten())
	print(biases)
	print("\n")
	test(weights, biases, testx, testy)


if __name__ == '__main__':
	main()