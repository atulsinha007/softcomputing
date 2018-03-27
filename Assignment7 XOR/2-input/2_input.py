from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import load_data
import test


def detail_about_model(model):
    print("\nSUMMARY OF THE MODEL:")
    summary = model.summary()

    weights_h = model.layers[0].get_weights()[0]
    weights_o = model.layers[1].get_weights()[0]
    bias_h = model.layers[0].get_weights()[1]
    bias_o = model.layers[1].get_weights()[1]
    print(summary)
    print("\n\nWeights of Hidden layer 1:")
    print(weights_h)
    print("\nbias:")
    print(bias_h)
    print("\nWeights of Output layer:")
    print(weights_o)
    print("\nbias:")
    print(bias_o)

def main():
    trainx, trainy, testx, testy = load_data.load_data()

    model = Sequential()
    model.add(Dense(8, input_dim=2, activation = 'tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.2))
    model.fit(trainx, trainy, batch_size=4, epochs=500)

    detail_about_model(model)

    test.testing(model, testx, testy)

if __name__ == '__main__':
    main()