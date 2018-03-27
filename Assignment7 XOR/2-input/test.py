from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import load_data

def testing(model, testx, testy):
    prediction = model.predict(testx)
    result = [round(temp[0]) for temp in prediction]
    print("\nTESTING\n")
    print("A\t B\t A^B\t Expected")
    print("----------------------------------\n")
    acc = 0
    for i in range(len(testx)):
        print(testx[i][0],"\t", testx[i][1], "\t", result[i],"\t", testy[i][0])
        if result[i] == testy[i][0]:
            acc += 1
    acc = (acc / len(testx)) * 100
    print("Accuracy:",acc,"%")
