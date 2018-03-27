import numpy as np 

def load_data():
	data = np.loadtxt("dataset.csv", delimiter=',')
	data = data.astype(float)
	n_rows = 8
	n_cols = len(data[0])
	trainx = data[:n_rows, :n_cols-1]
	trainy = data[:n_rows, -1:]
	testx = data[n_rows:, :n_cols-1]
	testy = data[n_rows:, -1:]
	return trainx, trainy, testx, testy