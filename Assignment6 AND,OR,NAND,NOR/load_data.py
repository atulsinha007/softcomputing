import numpy as np 
import tensorflow as tf

def load_data(data):
	data = np.loadtxt(data, delimiter=',')
	n_rows = 4
	n_cols = len(data[0])
	trainx = data[:n_rows, :n_cols-1]
	trainy = data[:n_rows, -1:]
	testx = data[n_rows:, :n_cols-1]
	testy = data[n_rows:, -1:]
	return trainx, trainy, testx, testy