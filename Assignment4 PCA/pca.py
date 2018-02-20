import cv2
import os
import numpy as np
import load_data
import test 

def normalization(x, mean):

    x = x-mean
    return x

def main():
    
    trainx, trainy, testx, testy = load_data.load_data()
    rows, cols = trainx.shape
    mean = trainx.mean(axis = 1)
    mean = mean.reshape(rows, 1)
    new_trainx = normalization(trainx, mean)
    k = 20
    cov_matrix = np.cov(new_trainx.T)
    eigenval, eigenvec = np.linalg.eig(cov_matrix)
    ind = eigenval.argsort()[::-1]   
    eigenval = eigenval[ind]
    eigenvec = eigenvec[:,ind]

    sigma = eigenvec[0:k, :]
    sigma = sigma.T
    eigen_faces = np.dot(sigma.T , new_trainx.T)
    signature_faces = eigen_faces.dot(new_trainx)
    test.test(mean, testx, testy, eigen_faces, signature_faces)


if __name__ == '__main__':
    main()