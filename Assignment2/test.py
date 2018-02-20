import numpy as np 

def accuracy(output, testy):
	n = len(output)
	sum = 0
	for i in range(n):
		# print(output[i], testy[i])
		sum += np.abs(output[i] - testy[i])

	accuracy_percentage = (float(sum)/n)
	return accuracy_percentage
	
def compare(output, testy):
	n = len(output)
	sum = 0
	for i in range(n):
		print(output[i], testy[i])