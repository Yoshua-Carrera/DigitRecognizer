from numpy.core.fromnumeric import argmax
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

from dataImportCleaning import dataTool

def nn_prediction(showEval: bool, epochs: int, splt: bool, testSize: float=.25):
    data_Tool = dataTool('data/train.csv', 'data/test.csv')
    train, test =  data_Tool.retrieveData

    if splt:
        x_train, x_test, y_train, y_test = data_Tool.split_data(train, 'label', splt, testSize)
        x_train, x_test = data_Tool.normalize(train.drop('label', axis = 1), [x_train, x_test])
    else:
        x_train, y_train = data_Tool.split_data(train, 'label', splt, testSize)
        x_train = data_Tool.normalize(train.drop('label', axis = 1), [x_train])

    # add neural network model
    nn = Sequential()

    '''
    first layer is input layer and has 784 inputs and relu activation
    second layer has 128 neurons and also relu activation
    third layer has 128 neurons and also relu activation
    fourth layer has 10 output neurons, it's one for each digit and it's softmax activation

    '''
    nn.add(Dense(10, input_dim=784, activation='relu'))
    nn.add(Dense(128, activation='relu'))
    nn.add(Dense(128, activation='relu'))
    nn.add(Dense(10, activation='softmax'))

    nn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    nn.summary()

    # x_train_array = np.asarray(x_train)
    
    history = nn.fit(x_train, y_train, epochs=epochs, batch_size=10)

    prediction = nn.predict(test)
    actual_prediction = []

    for item in prediction:
        actual_prediction.append(np.argmax(item))
    
    if splt:
        print(f'Accuracy: {accuracy_score(actual_prediction, y_test)}')
    
    if showEval:
        plt.plot(history.epoch, history.history['loss'])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

        plt.plot(history.epoch, history.history['accuracy'])
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.show()

        y_pred = nn.predict(x_test)

        loss,accuracy = nn.evaluate(x_test,y_test)
        print(loss,accuracy)
        
        return actual_prediction
    else:
        print(data_Tool.brr)

        return actual_prediction
        