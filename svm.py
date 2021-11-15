from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataImportCleaning import dataTool
import numpy as np


dataTool = dataTool('data/train.csv', 'data/test.csv')

train, test =  dataTool.retrieveData

x_train, x_test, y_train, y_test = dataTool.split_data(train, 'label', True, 0.99)

x_train, x_test = dataTool.normalize(train.drop('label', axis = 1), [x_train, x_test])

#see the shae for train and test dataset
print(f'Train size: {x_train.shape}')
print(f'Test size: {x_test.shape}')

from sklearn.svm import SVC

# add svm model classifier
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(x_train, y_train)

#predict values
y_pred_SVM = svclassifier.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy:    ", accuracy_score(y_pred_SVM,y_test))


SVM_matrix = confusion_matrix(y_pred_SVM,y_test)
print(SVM_matrix)

#(def is to identify the confusion matrix so keep it)

def plot_cnf_matirx(cnf_matrix,description):
    class_names = [0,1]
    fig,ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks,class_names)
    plt.yticks(tick_marks,class_names)

# (so you need to change the pd.datafram  ...matix (the name))
import seaborn as sns
import matplotlib.axes as ax
sns.heatmap(pd.DataFrame(SVM_matrix), annot = True, cmap = 'OrRd',fmt = 'g')
#ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('confusion matrix for SVM model')  #(change title too)
plt.ylabel('actual value 0/1',fontsize=12)
plt.xlabel('predict value 0/1',fontsize=12)
plt.show()