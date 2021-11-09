import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
train

test = pd.read_csv('data/test.csv')
test

print(f'Train size: {train.shape}')
print(f'Test size: {test.shape}')

from sklearn.svm import SVC

x_train = train.drop('label', axis = 1)
y_train = train['label']

svclassifier = SVC(kernel = 'linear')
svclassifier.fit(x_train,y_train)

y_pred_SVM = svclassifier.predict(test)

