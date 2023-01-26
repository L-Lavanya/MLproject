#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer


# In[18]:


df=pd.read_csv("C:/Users/lavanya lakkakula/Downloads/spam.csv",encoding="latin-1")


# In[20]:


df.head(10)


# In[25]:


df.shape


# In[27]:


df.ndim


# In[38]:


#for training and testing data split
5572-5572/4


# In[31]:


#to check target attribute is binary or not
np.unique(df["class"])


# In[33]:


np.unique(df["message"])
#it is not binary,so create sparse matrix and count vectorizer object


# In[60]:


#creating sparse matrix
x=df["message"].values
#y=df["class"].values
#create count vectorizer obj
cv=CountVectorizer()
x=cv.fit_transform(x)
v=x.toarray()
print(v)


# In[48]:


#to make easier way to calculate last column make class as last column and message as first
first_col=df.pop("message")
df.insert(0,"message",first_col)
df


# In[49]:


#splitting train+test 3:1
train_x=x[:4179]
train_y=y[:4179]
test_x=x[4179:]
test_y=y[4179:]


# In[50]:


bnb=BernoulliNB(binarize=0.0)
model=bnb.fit(train_x,train_y)
y_pred_train=bnb.predict(train_x)
y_pred_test=bnb.predict(test_x)


# In[52]:


#training score
print(bnb.score(train_x,train_y)*100)
#testing score
print(bnb.score(test_x,test_y)*100)


# In[55]:


from sklearn.metrics import classification_report
print(classification_report(train_y,y_pred_train))


# In[57]:


print(classification_report(test_y,y_pred_test))

