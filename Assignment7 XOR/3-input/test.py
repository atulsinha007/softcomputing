from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import load_data
import test

def testing(model, testx, testy):
    dataset = np.loadtxt("test.csv", delimiter=',')
    rows = len(dataset)
    cols = len(dataset[0])
    x = dataset[:rows, : cols - 1]
    y = dataset[:rows, cols - 1 : cols]
    x = np.array(x)
    y = np.array(y)

    prediction = model.predict(x)
    result = [round(temp[0]) for temp in prediction]
    print("\nTESTING:\n")
    print("A\t B\t C\t A^B^C\t Expected")
    acc = 0
    for i in range(rows):
        print(x[i][0],"\t",x[i][1],"\t",x[i][2],"\t",result[i],"\t\t",y[i][0])
        if(result[i] == y[i][0]):
            acc += 1
    acc = (acc / rows) * 100
    print("Accuracy:", acc,"%")