import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv').head(100)

test = pd.read_csv('data/test.csv').head(100)

print(f'Train size: {train.shape}')
print(f'Test size: {test.shape}')

from sklearn.svm import SVC

x_train = train.drop('label', axis = 1)
y_train = train['label']

x_train.reset_index(drop=True)
y_train.reset_index(drop=True)
test.reset_index(drop=True)

svclassifier = SVC(kernel = 'linear')
svclassifier.fit(x_train,y_train)

y_pred_SVM = svclassifier.predict(test)

