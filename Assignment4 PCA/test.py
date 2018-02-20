import cv2
import os
import numpy as np

imposter = False

def accuracy(prediction, testy):

    cnt = 0
    for x, y in zip(prediction, testy):
        print(x, "\t", y)
        if x == y or x == -1:
            cnt += 1
    print("Accuracy = ", (cnt/testy.shape[0])*100)

def test(mean, testx, testy, eigen_face, signature_face):
    a, b = testx.shape
    prediction = []

    for test_image in testx.T:

        test_image = test_image.reshape(a,1) - mean
        final_eigenface = eigen_face.dot(test_image)
        min_dist = np.linalg.norm(signature_face.T[0].reshape(final_eigenface.shape) - final_eigenface)
        index = 0
        count = 0

        for col in signature_face.T:
            v = col.reshape(final_eigenface.shape)
            dist = np.linalg.norm(v - final_eigenface)
            if dist < min_dist:
                min_dist = dist
                index = count
            count += 1

        if min_dist > 640000000:
            print("-------------Imposter-------------")
            imposter = True
            prediction.append(-1)
        else :
            print(min_dist)
            prediction.append(index//4 + 1)

    accuracy(prediction, testy)
