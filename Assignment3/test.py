import numpy as np 

def accuracy(output, testy):
	n = len(output)
	cnt = 0
	for i in range(n):
		# print(np.shape(output), np.shape(testy))
		if output[i] == testy[i]:
			cnt += 1
	accuracy_percentage = (float(cnt)/n * 100)
	return accuracy_percentage
	
def compare(output, testy):
	n = len(output)
	sum = 0
	for i in range(n):
		print(output[i], testy[i])