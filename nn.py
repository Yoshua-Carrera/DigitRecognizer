import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from dataImportCleaning import dataTool

dataTool = dataTool('data/train.csv', 'data/test.csv')

train, test =  dataTool.retrieveData

x_train, x_test, y_train, y_test = dataTool.split_data(train, 'label', True, 0.4)

x_train, x_test = dataTool.normalize(train.drop('label', axis = 1), [x_train, x_test])

nn = Sequential()

nn.add(Dense(10, input_dim=784, activation='relu'))
nn.add(Dense(128, activation='relu'))
nn.add(Dense(128, activation='relu'))
nn.add(Dense(10, activation='softmax'))

nn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn.summary()

x_train_array = np.asarray(x_train)

history = nn.fit(x_train_array, y_train, epochs=100, batch_size=10)

plt.plot(history.epoch, history.history['loss'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

plt.plot(history.epoch, history.history['accuracy'])
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()