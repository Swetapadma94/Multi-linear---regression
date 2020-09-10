#!/usr/bin/env python
# coding: utf-8

# In[398]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[399]:


data=pd.read_csv(r'E:\Udemy\dataset\Startups.csv',encoding='latin1')


# In[400]:


data.head()


# In[402]:


data.isna().mean()


# In[405]:


X=data.iloc[:,:-1]
Y=data.iloc[:,-1]
X,Y


# In[406]:


states=pd.get_dummies(X['State'],drop_first=True)


# In[407]:


X=X.drop('State',axis=1)
X=pd.concat([X,states],axis=1)


# In[408]:


from sklearn.model_selection import train_test_split


# In[409]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


# In[410]:


X_train


# In[411]:


Y_train


# In[412]:



X_train.shape,Y_train.shape


# In[416]:


from sklearn.linear_model import LinearRegression


# In[417]:


sl=LinearRegression()


# In[418]:


sl.fit(X_train,Y_train)


# In[419]:


y_predict=sl.predict(X_test)


# In[420]:


from sklearn.metrics import r2_score
score=r2_score(Y_test,y_predict)


# In[421]:


score


# In[423]:


dt = pd.DataFrame({'Actual': Y_test, 'Predicted': y_predict})
dt


# In[ ]:




