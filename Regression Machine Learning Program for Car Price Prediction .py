#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Starting a Practical use case of Regression Car Price Prediction


# In[1]:


import numpy as np


# In[10]:


import pandas as pd
import seaborn as sb


# In[3]:


import sklearn as skl


# In[14]:


#skl.show_versions()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[52]:


df=pd.read_csv('car_data.csv')


# In[53]:


#df.head


# In[6]:


df


# In[55]:


#df.isnull().sum()


# In[87]:


#df.drop(['Car_Name','Selling_Price'],axis=1)


# In[57]:


y=df['Selling_Price']
y


# In[74]:


#df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}})


# In[88]:


df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
df.replace ({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
df.replace({'Transmission':{'Automatic':0,'Manual':1}},inplace=True)
X=df.drop(['Car_Name','Selling_Price'],axis=1)


# In[72]:


#df.replace ({'Seller_Type':{'Dealer':0,'Individual':1}})


# In[73]:


#df.replace({'Transmission':{'Automatic':1,'Manual':0}})


# In[ ]:





# In[95]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)


# In[96]:


lr=Lasso()


# In[97]:


lr.fit(X_train,y_train)


# In[98]:


pred=lr.predict(X_test)


# In[99]:


metrics.r2_score(y_test,pred)


# In[ ]:




