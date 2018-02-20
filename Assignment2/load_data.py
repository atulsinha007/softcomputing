import numpy as np 
import tensorflow as tf

def load_data():
	data = np.loadtxt("data.csv", delimiter=',')
	# np.random.shuffle(data)
	data = data.astype(float)
	n_rows = int(0.75*len(data))
	n_cols = len(data[0])
	trainx = data[:n_rows, :n_cols-1]
	trainy = data[:n_rows, -1]
	testx = data[n_rows:, :n_cols-1]
	testy = data[n_rows:, -1]
	return trainx, trainy, testx, testy