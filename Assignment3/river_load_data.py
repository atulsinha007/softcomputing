import numpy as np 

def normalization(x):
	maxx = np.max(x, axis = 0)
	minn = np.min(x, axis = 0)
	for row in x:
		for i in range(row.size):
			row[i] = (row[i] - minn[i])
			row[i] = row[i]/(maxx[i] - minn[i])
		
	return x, maxx, minn


def create_dataset(img):
	rows = len(img)
	cols = len(img[0])
	data = []
	for i in range(rows):	
		for j in range(cols):
			row = []
			# row.append(i)
			row.append(j)
			row.append(img[i][j])
			if(img[i][j] > 220 and j < cols/2 and j > 100):
				row.append(1)
			else:
				row.append(0)
			data.append(row)
	data = np.asarray(data)
	data = data.astype(float)
	np.random.shuffle(data)
	data, maxx, minn = normalization(data)
	n_rows = int(0.75*len(data))
	n_cols = len(data[0])
	trainx = data[:n_rows, :n_cols-1]
	trainy = data[:n_rows, -1:]
	testx = data[n_rows:, :n_cols-1]
	testy = data[n_rows:, -1:]	
	return trainx, trainy, testx, testy, maxx, minn