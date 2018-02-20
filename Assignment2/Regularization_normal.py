import numpy as np 
import tensorflow as tf
import load_data
import test
lambda_val = 10

#	*************************hypothesis w0x0 + w1x1 + w2x2 + w3x1^2 + w4x2^2 + w5x1x2*************************

def calc_normal_eq(z):
	x = z
	xT = np.transpose(x)
	w = np.matmul(np.matmul(np.linalg.inv(np.matmul(xT, x)), xT),y)
	return w

def convert(x):
	rows = len(x)
	z = []
	for i in range(rows):
		z.append([1] + list(x[i]))
	z = np.array(z)
	matrix = np.zeros((6,6))
	for i in range(1,6):
		matrix[i][i] = 1

	aux_lis = []
	for i in range(len(z)):
		temp = z[i]
		temp2 = (temp[0], temp[1], temp[2], temp[1]*temp[1], temp[2]*temp[2], temp[1]*temp[2])
		aux_lis.append(temp2)
	x = np.array(aux_lis)
	return x, matrix

def calc_hypothesis(x, y):	
	x, matrix = convert(x)
	xT = np.transpose(x)
	xTx = np.matmul(xT,x)
	w = np.linalg.inv(np.add(xTx, np.multiply(lambda_val, matrix))) 
	w = np.matmul(w, np.matmul(xT, y))
	w = np.reshape(w, (np.shape(w)[0], 1))
	y = np.matmul(x, w)
	return w, y	


def main():
	trainx, trainy, testx, testy = load_data.load_data()
	w, y1 = calc_hypothesis(trainx, trainy)
	y1 = np.reshape(y1, (np.shape(y1)[0], ))

	# train_accuracy = test.accuracy(y1, trainy)
	# print(train_accuracy)

	testx, _ = convert(testx)
	n = len(testx)
	y_dash = []
	for i in range(n):
		y_dash.append(np.matmul(testx[i],w))

	test.compare(y_dash, testy)
	test_accuracy = test.accuracy(y_dash, testy)
	print(test_accuracy)
	
	
if __name__ == '__main__':
	main()