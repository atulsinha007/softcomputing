import cv2
import os
import numpy as np
import load_data
import test 

k = 20

def normalization(x, mean):

    x = x-mean
    return x

def run():

    trainx, trainy, testx, testy = load_data.load_data()
    rows, cols = trainx.shape
    mean = trainx.mean(axis = 1)
    mean = mean.reshape(rows, 1)
    new_trainx = normalization(trainx, mean)
    covariance_matrix = np.cov(new_trainx.T)
    # print("Covariance matrix shape =", covariance_matrix.shape)
    eigenvalue, eigenvector = np.linalg.eig(covariance_matrix)

    idx = eigenvalue.argsort()[::-1]   
    eigenvalue = eigenvalue[idx]
    eigenvector = eigenvector[:,idx]

    sigma = eigenvector[0:k, :]
    sigma = sigma.T
    eigen_faces = np.dot(sigma.T , new_trainx.T)

    signature_face = eigen_faces.dot(new_trainx)
    # print("Signature face = ", signature_face.shape)
    # print("Eigen faces = ", eigen_faces.shape)
    # test.test(mean, testx, testy, eigen_faces, signature_face)

    return trainx, trainy, testx, testy, signature_face, eigen_faces

# if __name__ == '__main__':
#     main()