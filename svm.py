#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train = pd.read_csv('data/train.csv')
train


# In[3]:


test = pd.read_csv('data/test.csv')
test


# In[4]:


print(f'Train size: {train.shape}')
print(f'Test size: {test.shape}')


# In[5]:


from sklearn.svm import SVC


# In[6]:


x_train = train.drop('label', axis = 1)
y_train = train['label']


# In[ ]:


svclassifier = SVC(kernel = 'linear')
svclassifier.fit(x_train,y_train)

y_pred_SVM = svclassifier.predict(test)

