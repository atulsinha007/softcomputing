import numpy as np 
import pca
import load_data
import test
import sys

n_classes = 5
k = 20
m = 10

imposter = False

def scatter_within_class(dic_of_classes):
	SW = 0
	for i in dic_of_classes.keys():
		mean_temp = (np.mean(dic_of_classes[i], axis = 1)).reshape(dic_of_classes[i].shape[0],1)
		
		SW += np.matmul((dic_of_classes[i] - mean_temp),((dic_of_classes[i] - mean_temp).T))
	return SW


def scatter_between_classes(dic_of_classes, mean_signature):
	SB = 0
	for i in dic_of_classes.keys():
		mean_temp = (np.mean(dic_of_classes[i], axis = 1)).reshape(dic_of_classes[i].shape[0],1)
		sigma_i = np.matmul(mean_temp - mean_signature, mean_temp.T)
		SB += sigma_i
	return SB


def construct_feature(eigenvector, m):
	feature_matrix = eigenvector[:m,:]
	return feature_matrix


def fisher_faces(feature_matrix, signature_face):
	return np.matmul(np.transpose(feature_matrix), signature_face)


def min_distance(ff, Projected_Fisher_Test_Img):
	ffT = ff.T
	min_dist = np.linalg.norm(ffT[0].reshape(Projected_Fisher_Test_Img.shape) - Projected_Fisher_Test_Img)
	index = 0
	cnt = 0
	for row in ffT:
		dist = np.linalg.norm(row.reshape(Projected_Fisher_Test_Img.shape) - Projected_Fisher_Test_Img)
		if dist < min_dist:
			min_dist = dist
			index = cnt
		cnt += 1

	# if min_dist > 1000000000: #80
	# 	print("-------------Imposter-------------")
	# 	imposter = True
	# 	return "imposter"

	if min_dist > 340000000:
		print("-------------Imposter-------------")
		imposter = True
		return "imposter"
	print(min_dist)
	return (index//4 + 1)



def testing(testx, testy, mean_train, ff, feature_matrix, eigen_faces):
	a, b = testx.shape
	prediction = []

	for test_image in testx.T:
		test_image = test_image.reshape(a,1) - mean_train
		PEF = np.matmul(eigen_faces, test_image)

		Projected_Fisher_Test_Img = np.matmul(np.transpose(feature_matrix), PEF)
		predicted_class = min_distance(ff, Projected_Fisher_Test_Img)
		
		if predicted_class != "imposter":
			prediction.append(predicted_class)
		else:
			prediction.append(-1)
	test.accuracy(prediction, testy)


def main():

	trainx, trainy, testx, testy, signature_face, eigen_faces = pca.run()
	dic_of_classes = {}

	mean_signature = (signature_face.mean(axis = 1)).reshape(signature_face.shape[0],1)
	mean_train = (trainx.mean(axis = 1)).reshape(trainx.shape[0], 1)

	for i in range(n_classes):
		dic_of_classes[i+1] = signature_face[:, i:i+4] #4 pics of one person in trainx

	means = {}

	for x,y in dic_of_classes.items():
		mean_val = y.mean(axis = 1)
		mean_val = mean_val.reshape(mean_val.shape[0], 1)
		means[x] = mean_val

	SW = scatter_within_class(dic_of_classes)
	SB = scatter_between_classes(dic_of_classes, mean_signature)
	J = np.matmul(np.linalg.inv(SW), SB)

	eigenvalue, eigenvector = np.linalg.eig(J)
	idx = eigenvalue.argsort()[::-1]   
	eigenvalue = eigenvalue[idx]
	eigenvector = eigenvector[:,idx]

	feature_matrix = eigenvector[:m, :].T
	ff = fisher_faces(feature_matrix, signature_face)
	testing(testx, testy, mean_train, ff, feature_matrix, eigen_faces)


if __name__ == '__main__':
	main()